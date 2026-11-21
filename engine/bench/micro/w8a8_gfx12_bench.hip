// w8a8_bench.hip - gfx12 microbenchmark for Strata's dense prompt projections.
//
//   A) today's Strata path:  Q6_K weight -> FP16 scratch (dequant) + hipBLAS GemmEx f16 x f16 -> f32
//   B) radiance's path:      Q6_K weight -> int8 fragment-order plane + f16 scale per (row, GRP k)
//                            (requantized per call, like A's dequant), activation -> int8 per row,
//                            r4d_gemm_w8a8_tiled (int8 WMMA), f32 out
//
// The W8A8 GEMM body is radiance's r4d_gemm_w8a8_tiled_kernel (libr4d/r4d_gemm_w8a8_tiled.hip,
// codeberg.org/StillDeadcode/radiance, Apache-2.0) with the scale group as a template argument
// and an f32 (ldy, beta) epilogue instead of bf16. The activation quantizer is r4d_quant_act_i8's
// byte order on FP16 input.
//
// Build: hipcc -O3 --offload-arch=gfx1200 -std=c++17 w8a8_gfx12_bench.hip -lhipblas -o w8a8_gfx12_bench
// Run:   ./w8a8_gfx12_bench [T ...]        (default T = 1024 2048 4096)
//        W8A8_OUTLIERS=0 turns off the activation outlier channels
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <hipblas/hipblas.h>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>
#include <algorithm>

#define CK(x) do { hipError_t e_ = (x); if (e_ != hipSuccess) { \
  std::fprintf(stderr, "%s:%d %s: %s\n", __FILE__, __LINE__, #x, hipGetErrorString(e_)); std::exit(1); } } while (0)
#define CKB(x) do { hipblasStatus_t s_ = (x); if (s_ != HIPBLAS_STATUS_SUCCESS) { \
  std::fprintf(stderr, "%s:%d %s: hipblas %d\n", __FILE__, __LINE__, #x, (int)s_); std::exit(1); } } while (0)

// ------------------------------------------------------------------ Q6_K (ggml block_q6_K)
struct BlockQ6K {
  uint8_t ql[128];
  uint8_t qh[64];
  int8_t sc[16];
  uint16_t d;   // f16
};
static_assert(sizeof(BlockQ6K) == 210, "block_q6_K is 210 bytes");

__device__ __forceinline__ float q6k_val(const BlockQ6K* __restrict__ b, int i) {
  const int half = i >> 7, r = i & 127, quad = r >> 5, l = r & 31;
  const uint8_t* ql = b->ql + half * 64;
  const uint8_t* qh = b->qh + half * 32;
  const int8_t* sc = b->sc + half * 8;
  const int lo = (quad & 1) ? ql[l + 32] : ql[l];
  const int nib = (quad & 2) ? (lo >> 4) : (lo & 15);
  const int hi = (qh[l] >> (2 * quad)) & 3;
  const int q = (nib | (hi << 4)) - 32;
  const float d = __half2float(__ushort_as_half(b->d));
  return d * (float)sc[(l >> 4) + 2 * quad] * (float)q;
}

// A: dequant to FP16, row-major [N][K]. One workgroup = one super-block (256 values).
__global__ __launch_bounds__(256) void dequant_q6k_f16(const BlockQ6K* __restrict__ W, __half* __restrict__ out,
                                                       int64_t nblocks) {
  const int64_t b = blockIdx.x;
  if (b >= nblocks) return;
  out[b * 256 + threadIdx.x] = __float2half(q6k_val(W + b, threadIdx.x));
}

// B1: requantize to radiance's w8 fragment plane. One workgroup = 16 rows x GRP k.
// Byte j of lane l in a (tile t, k-block kb) 512-byte run is
//   W[t*16 + (l & 15)][kb*32 + 16*(j >> 3) + 8*((j & 7) >> 2) + 4*(l >> 4) + (j & 3)]
template <int GRP>
__global__ __launch_bounds__(16 * GRP / 8) void requant_q6k_w8(const BlockQ6K* __restrict__ W, int K,
                                                               uint8_t* __restrict__ Wq,
                                                               __half* __restrict__ Ws) {
  constexpr int TPR = GRP / 8;                       // threads per row, 8 k each
  const int t = blockIdx.x, g = blockIdx.y;          // n-tile, scale group
  const int r = threadIdx.x / TPR, c8 = threadIdx.x % TPR;
  const int n = t * 16 + r;
  const int k0 = g * GRP + c8 * 8;
  const int bpr = K / 256;
  float v[8];
  float m = 0.f;
#pragma unroll
  for (int e = 0; e < 8; e++) {
    const int k = k0 + e;
    v[e] = q6k_val(W + (int64_t)n * bpr + (k >> 8), k & 255);
    m = fmaxf(m, fabsf(v[e]));
  }
  __shared__ unsigned amax[16];
  if (threadIdx.x < 16) amax[threadIdx.x] = 0u;
  __syncthreads();
  atomicMax(&amax[r], __float_as_uint(m));           // non-negative floats order as their bits
  __syncthreads();
  const float mx = __uint_as_float(amax[r]);
  const float s = __half2float(__float2half(fmaxf(mx, 1e-12f) / 127.f));   // the f16 scale the GEMM reads
  const float inv = 1.f / s;
  const int nsz = K / GRP, kpb = K / 32;
  if (c8 == 0) Ws[((int64_t)t * nsz + g) * 16 + r] = __float2half(s);
#pragma unroll
  for (int e = 0; e < 8; e++) {
    const int k = k0 + e;
    const int kb = k >> 5, kk = k & 31;
    const int u = kk >> 4, rr = kk & 15;
    const int a = rr >> 3, h = (rr >> 2) & 1, bb = rr & 3;
    const int j = u * 8 + a * 4 + bb, l = h * 16 + r;
    int q = __float2int_rn(v[e] * inv);
    q = q < -127 ? -127 : (q > 127 ? 127 : q);
    Wq[(((int64_t)t * kpb + kb) * 32 + l) * 16 + j] = (uint8_t)(int8_t)q;
  }
}

// B2: per-row symmetric int8 activation in r4d_quant_act_i8's byte order (FP16 input).
__global__ __launch_bounds__(256) void quant_act_i8_f16(const __half* __restrict__ X, int8_t* __restrict__ Q,
                                                        float* __restrict__ S, int K) {
  const int row = blockIdx.x;
  const __half* a = X + (int64_t)row * K;
  int8_t* q = Q + (int64_t)row * K;
  float m = 0.f;
  for (int i = threadIdx.x; i < K; i += 256) m = fmaxf(m, fabsf(__half2float(a[i])));
  for (int o = 16; o > 0; o >>= 1) m = fmaxf(m, __shfl_xor(m, o, 32));
  __shared__ float red[8];
  if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = m;
  __syncthreads();
  m = red[0];
#pragma unroll
  for (int i = 1; i < 8; i++) m = fmaxf(m, red[i]);
  const float scale = fmaxf(m, 1e-8f) / 127.f, inv = 1.f / scale;
  if (threadIdx.x == 0) S[row] = scale;
  for (int i = threadIdx.x; i < K; i += 256) {
    const int s = i >> 4, j = i & 15;
    const int src = (s << 4) + 8 * ((j & 7) >> 2) + 4 * (j >> 3) + (j & 3);
    const int t = __float2int_rn(__half2float(a[src]) * inv);
    q[i] = (int8_t)(t < -128 ? -128 : (t > 127 ? 127 : t));
  }
}

// B3: radiance r4d_gemm_w8a8_tiled_kernel, GRP templated, f32 (ldy, beta) epilogue.
#define TILED8_KPB 32
#define TILED8_BK 32
typedef int v2i8_t __attribute__((ext_vector_type(2)));
typedef int v8i8_t __attribute__((ext_vector_type(8)));
typedef float v8f8_t __attribute__((ext_vector_type(8)));
typedef unsigned u4v8_t __attribute__((ext_vector_type(4)));

template <int WM, int WN, int TM, int TN, int GRP>
__global__ __launch_bounds__(WM * WN * 32, 8) void w8a8_tiled_kernel(
    const signed char* __restrict__ A, const float* __restrict__ Ascale, const unsigned* __restrict__ Wq,
    const unsigned* __restrict__ Ws, float* __restrict__ Y, int M, int K, int N, int ldy, float beta) {
  constexpr int BK = TILED8_BK;
  constexpr int UNR = GRP / BK;
  constexpr int BM = WM * TM * 16, BN = WN * TN * 16;
  constexpr int NTILE = BN / 16;
  constexpr int AW = BM * BK / 4;
  constexpr int WW = NTILE * BK * 4;
  static_assert(GRP % BK == 0, "group must be whole k steps");

  __shared__ unsigned lds_a[2][AW];
  __shared__ unsigned lds_w[2][WW];

  const int tid = threadIdx.x, lane = tid & 31, wave = tid >> 5;
  const int wm = wave / WN, wn = wave % WN;
  const int m0 = blockIdx.y * BM, n0 = blockIdx.x * BN;
  const int kpb = K / TILED8_KPB, nsz = K / GRP;
  constexpr int NT_ = WM * WN * 32;

  v8f8_t acc[TN][TM];
#pragma unroll
  for (int j = 0; j < TN; j++)
#pragma unroll
    for (int i = 0; i < TM; i++) acc[j][i] = (v8f8_t)(0.f);
  v8i8_t g[TN][TM];
  float csum[TN];
#pragma unroll
  for (int j = 0; j < TN; j++) csum[j] = 0.f;
  unsigned scv[TN];

#define ASW_M2 ((BK / 8) - 1)
#define ASW(r, c) ((r) * (BK / 4) + ((c) ^ ((((r) >> 1) & ASW_M2) << 1)))
  constexpr int APT = (AW + NT_ - 1) / NT_;
  constexpr bool APAD = (AW % NT_) != 0;
  unsigned ra[APT];
  constexpr int WWP = NTILE * BK;
  constexpr int WPTU = (WWP + NT_ - 1) / NT_;
  constexpr bool WPAD = (WWP % NT_) != 0;
  u4v8_t rwu[WPTU];
  auto issue = [&](int kk) {
#pragma unroll
    for (int x = 0; x < WPTU; x++) {
      const int t = tid + x * NT_;
      if (WPAD && t >= WWP) continue;
      const int tl = t / BK, rem = t % BK, kh = rem / 32, ln = rem % 32;
      rwu[x] = *((const u4v8_t*)Wq + (size_t)(((n0 / 16 + tl) * kpb + kk / TILED8_KPB + kh) * 32 + ln));
    }
#pragma unroll
    for (int x = 0; x < APT; x++) {
      const int t = tid + x * NT_, r = t / (BK / 4), c = t % (BK / 4);
      if (APAD && t >= AW) continue;
      const int mm = m0 + r < M ? m0 + r : M - 1;
      ra[x] = *(const unsigned*)(A + (unsigned)(mm * K + kk + c * 4));
    }
  };
  auto commit = [&](int buf) {
#pragma unroll
    for (int x = 0; x < WPTU; x++) {
      const int t = tid + x * NT_;
      if (WPAD && t >= WWP) continue;
      const int tl = t / BK, rem = t % BK, kh = rem / 32, ln = rem % 32;
      *(u4v8_t*)(lds_w[buf] + (size_t)tl * (BK * 4) + ln * (BK / 8) + kh * 4) = rwu[x];
    }
#pragma unroll
    for (int x = 0; x < APT; x++) {
      const int t = tid + x * NT_, r = t / (BK / 4), c = t % (BK / 4);
      if (APAD && t >= AW) continue;
      lds_a[buf][ASW(r, c)] = ra[x];
    }
  };

  issue(0);
  commit(0);
  __syncthreads();

  for (int kg = 0, it = 0; kg < K; kg += BK * UNR) {
#pragma unroll
    for (int h = 0; h < UNR; h++, it++) {
      const int k0 = kg + h * BK;
      const int buf = it & 1;
      if (k0 + BK < K) issue(k0 + BK);
      const bool first_half = (k0 % GRP) == 0;
      if ((k0 % GRP) == 0) {
#pragma unroll
        for (int j = 0; j < TN; j++)
          scv[j] = (unsigned)((const unsigned short*)Ws)[(unsigned)(((n0 / 16 + wn * TN + j) * nsz + k0 / GRP) * 16 +
                                                                    (lane & 15))];
      }
#pragma unroll
      for (int u = 0; u < BK / 16; u++) {
        v2i8_t af[TM];
#pragma unroll
        for (int i = 0; i < TM; i++) {
          const int row = (wm * TM + i) * 16 + (lane & 15);
          const unsigned* pa = lds_a[buf] + ASW(row, u * 4 + 2 * (lane >> 4));
          af[i][0] = (int)pa[0];
          af[i][1] = (int)pa[1];
        }
#pragma unroll
        for (int j = 0; j < TN; j++) {
          const v2i8_t bf = *(const v2i8_t*)(lds_w[buf] + (size_t)(wn * TN + j) * (BK * 4) + lane * (BK / 8) + u * 2);
#pragma unroll
          for (int i = 0; i < TM; i++)
            g[j][i] = (first_half && u == 0)
                          ? __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(true, af[i], true, bf,
                                                                              (v8i8_t)(0x4B400000), false)
                          : __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(true, af[i], true, bf, g[j][i], false);
        }
      }
      if ((k0 + BK) % GRP == 0) {
        float sc[TN];
#pragma unroll
        for (int j = 0; j < TN; j++) sc[j] = __half2float(__ushort_as_half((unsigned short)(scv[j] & 0xFFFFu)));
#pragma unroll
        for (int j = 0; j < TN; j++) csum[j] += sc[j];
#pragma unroll
        for (int i = 0; i < TM; i++)
#pragma unroll
          for (int j = 0; j < TN; j++)
#pragma unroll
            for (int e = 0; e < 8; e++)
              acc[j][i][e] = fmaf(__uint_as_float((unsigned)g[j][i][e]), sc[j], acc[j][i][e]);
      }
      if (k0 + BK < K) commit(buf ^ 1);
      __syncthreads();
    }
  }
#undef ASW
#undef ASW_M2
#pragma unroll
  for (int j = 0; j < TN; j++) {
    const float d = -12582912.0f * csum[j];
#pragma unroll
    for (int i = 0; i < TM; i++)
#pragma unroll
      for (int e = 0; e < 8; e++) acc[j][i][e] += d;
  }
#pragma unroll
  for (int j = 0; j < TN; j++)
#pragma unroll
    for (int i = 0; i < TM; i++)
#pragma unroll
      for (int e = 0; e < 8; e++) {
        const int n = n0 + (wn * TN + j) * 16 + (lane & 15);
        const int m = m0 + (wm * TM + i) * 16 + 8 * (lane >> 4) + e;
        if (n < N && m < M) {
          float* y = Y + (size_t)m * ldy + n;
          const float v = acc[j][i][e] * Ascale[m];
          *y = beta == 0.f ? v : fmaf(beta, *y, v);
        }
      }
}

// ------------------------------------------------------------------ host side
struct Shape { const char* name; int N, K; };
// Qwen3.8-Flash-Next trunk (GGUF ne0 = K, ne1 = N); attn_qkv / ssm_out are the Q6_K majority
static const Shape kShapes[] = {
  {"attn_qkv  ", 10240, 2560}, {"attn_q    ", 12288, 2560}, {"attn_gate ", 6144, 2560},
  {"ssm_out   ", 2560, 6144},  {"shexp_up  ", 640, 2560},
};

struct Variant { const char* name; int BM, BN; void (*launch)(const int8_t*, const float*, const uint8_t*, const __half*, float*, int, int, int, hipStream_t); };

template <int WM, int WN, int TM, int TN, int GRP>
static void launch_v(const int8_t* a, const float* as, const uint8_t* wq, const __half* ws, float* y, int M, int K,
                     int N, hipStream_t st) {
  constexpr int BM = WM * TM * 16, BN = WN * TN * 16;
  dim3 grid(N / BN, (M + BM - 1) / BM), block(WM * WN * 32);
  hipLaunchKernelGGL((w8a8_tiled_kernel<WM, WN, TM, TN, GRP>), grid, block, 0, st, (const signed char*)a, as,
                     (const unsigned*)wq, (const unsigned*)ws, y, M, K, N, N, 0.f);
}

#define VARIANTS(G) \
  {"128x256", 128, 256, launch_v<2, 8, 4, 2, G>}, {"64x256 ", 64, 256, launch_v<2, 8, 2, 2, G>}, \
  {"96x256 ", 96, 256, launch_v<2, 8, 3, 2, G>},  {"128x128", 128, 128, launch_v<2, 8, 4, 1, G>}, \
  {"64x128 ", 64, 128, launch_v<1, 8, 4, 1, G>},  {"112x256", 112, 256, launch_v<1, 16, 7, 1, G>}, \
  {"160x128", 160, 128, launch_v<2, 8, 5, 1, G>}
static const Variant kV128[] = {VARIANTS(128)};
static const Variant kV32[] = {VARIANTS(32)};

static uint16_t f2h(float f) { __half h = __float2half(f); uint16_t u; std::memcpy(&u, &h, 2); return u; }

template <class F>
static float time_ms(hipStream_t st, int iters, F&& f) {
  hipEvent_t a, b;
  CK(hipEventCreate(&a)); CK(hipEventCreate(&b));
  for (int i = 0; i < 3; i++) f();
  CK(hipEventRecord(a, st));
  for (int i = 0; i < iters; i++) f();
  CK(hipEventRecord(b, st));
  CK(hipEventSynchronize(b));
  float ms = 0.f;
  CK(hipEventElapsedTime(&ms, a, b));
  CK(hipEventDestroy(a)); CK(hipEventDestroy(b));
  return ms / iters;
}

static void err_vs(const std::vector<float>& ref, const std::vector<float>& got, double& rel_l2, double& max_rel) {
  double num = 0, den = 0, mx = 0, refmax = 0;
  for (size_t i = 0; i < ref.size(); i++) refmax = std::max(refmax, (double)std::fabs(ref[i]));
  for (size_t i = 0; i < ref.size(); i++) {
    const double d = (double)got[i] - ref[i];
    num += d * d; den += (double)ref[i] * ref[i];
    mx = std::max(mx, std::fabs(d));
  }
  rel_l2 = std::sqrt(num / std::max(den, 1e-30));
  max_rel = mx / std::max(refmax, 1e-30);
}

int main(int argc, char** argv) {
  std::vector<int> Ts;
  for (int i = 1; i < argc; i++) Ts.push_back(std::atoi(argv[i]));
  if (Ts.empty()) Ts = {1024, 2048, 4096};
  const bool outliers = !(std::getenv("W8A8_OUTLIERS") && std::getenv("W8A8_OUTLIERS")[0] == '0');
  const int iters = std::getenv("W8A8_ITERS") ? std::atoi(std::getenv("W8A8_ITERS")) : 20;

  hipDeviceProp_t prop;
  CK(hipGetDeviceProperties(&prop, 0));
  std::printf("device: %s (%s), %d CUs; outliers %s, %d iters\n", prop.name, prop.gcnArchName,
              prop.multiProcessorCount, outliers ? "on" : "off", iters);

  hipStream_t st;
  CK(hipStreamCreate(&st));
  hipblasHandle_t hb;
  CKB(hipblasCreate(&hb));
  CKB(hipblasSetStream(hb, st));

  const int Tmax = *std::max_element(Ts.begin(), Ts.end());
  int64_t maxNK = 0;
  for (const Shape& s : kShapes) maxNK = std::max<int64_t>(maxNK, (int64_t)s.N * s.K);
  int Kmax = 0;
  for (const Shape& s : kShapes) Kmax = std::max(Kmax, s.K);

  BlockQ6K* dW; __half* dW16; uint8_t* dWq; __half* dWs; __half* dX; int8_t* dXq; float* dXs; float* dYa; float* dYb;
  CK(hipMalloc(&dW, maxNK / 256 * sizeof(BlockQ6K)));
  CK(hipMalloc(&dW16, maxNK * 2));
  CK(hipMalloc(&dWq, maxNK));
  CK(hipMalloc(&dWs, maxNK / 32 * 2));
  CK(hipMalloc(&dX, (int64_t)Tmax * Kmax * 2));
  CK(hipMalloc(&dXq, (int64_t)Tmax * Kmax));
  CK(hipMalloc(&dXs, (int64_t)Tmax * 4));
  int64_t maxTN = 0;
  for (const Shape& s : kShapes) maxTN = std::max<int64_t>(maxTN, (int64_t)Tmax * s.N);
  CK(hipMalloc(&dYa, maxTN * 4));
  CK(hipMalloc(&dYb, maxTN * 4));

  std::mt19937 rng(1234);
  std::printf("\n%-10s %5s | %-26s | %-46s | %s\n", "shape", "T", "A: dequant + hipBLAS f16 ms", "B: requant + actq + w8a8 ms (best tile)",
              "B/A speed, err vs A");
  for (const Shape& s : kShapes) {
    const int N = s.N, K = s.K;
    const int64_t nb = (int64_t)N * K / 256;
    // random Q6_K with realistic magnitudes: d ~ 1e-3..3e-3, sub-scales mostly positive
    std::vector<BlockQ6K> hW(nb);
    std::uniform_int_distribution<int> byte(0, 255), scd(8, 60);
    std::uniform_real_distribution<float> dd(1e-3f, 3e-3f);
    for (auto& b : hW) {
      for (auto& v : b.ql) v = (uint8_t)byte(rng);
      for (auto& v : b.qh) v = (uint8_t)byte(rng);
      for (auto& v : b.sc) v = (int8_t)scd(rng);
      b.d = f2h(dd(rng));
    }
    CK(hipMemcpy(dW, hW.data(), nb * sizeof(BlockQ6K), hipMemcpyHostToDevice));

    for (int T : Ts) {
      std::vector<uint16_t> hX((size_t)T * K);
      std::normal_distribution<float> nd(0.f, 1.f);
      for (size_t i = 0; i < hX.size(); i++) {
        float v = nd(rng);
        if (outliers && (i % K) % 97 == 3) v *= 20.f;   // a few hot channels, as RMS-normed hidden states have
        hX[i] = f2h(v);
      }
      CK(hipMemcpy(dX, hX.data(), hX.size() * 2, hipMemcpyHostToDevice));

      const float one = 1.f, zero = 0.f;
      auto pathA = [&] {
        hipLaunchKernelGGL(dequant_q6k_f16, dim3((unsigned)nb), dim3(256), 0, st, dW, dW16, nb);
        CKB(hipblasGemmEx(hb, HIPBLAS_OP_T, HIPBLAS_OP_N, N, T, K, &one, dW16, HIP_R_16F, K, dX, HIP_R_16F, K, &zero,
                          dYa, HIP_R_32F, N, HIPBLAS_COMPUTE_32F, HIPBLAS_GEMM_DEFAULT));
      };
      const float msA = time_ms(st, iters, pathA);
      const float msA_dq = time_ms(st, iters, [&] {
        hipLaunchKernelGGL(dequant_q6k_f16, dim3((unsigned)nb), dim3(256), 0, st, dW, dW16, nb);
      });
      std::vector<float> yA((size_t)T * N), yB((size_t)T * N);
      pathA();
      CK(hipStreamSynchronize(st));
      CK(hipMemcpy(yA.data(), dYa, yA.size() * 4, hipMemcpyDeviceToHost));

      for (int gi = 0; gi < 2; gi++) {
        const int GRP = gi == 0 ? 128 : 32;
        const Variant* V = gi == 0 ? kV128 : kV32;
        const int nV = gi == 0 ? (int)(sizeof(kV128) / sizeof(kV128[0])) : (int)(sizeof(kV32) / sizeof(kV32[0]));
        auto requant = [&] {
          if (GRP == 128)
            hipLaunchKernelGGL(requant_q6k_w8<128>, dim3(N / 16, K / 128), dim3(16 * 128 / 8), 0, st, dW, K, dWq, dWs);
          else
            hipLaunchKernelGGL(requant_q6k_w8<32>, dim3(N / 16, K / 32), dim3(16 * 32 / 8), 0, st, dW, K, dWq, dWs);
        };
        auto actq = [&] { hipLaunchKernelGGL(quant_act_i8_f16, dim3(T), dim3(256), 0, st, dX, dXq, dXs, K); };
        const float msRq = time_ms(st, iters, requant);
        const float msAq = time_ms(st, iters, actq);
        requant(); actq();
        float best = 1e30f; int bi = -1;
        for (int v = 0; v < nV; v++) {
          if (N % V[v].BN) continue;
          const float ms = time_ms(st, iters, [&] { V[v].launch(dXq, dXs, dWq, dWs, dYb, T, K, N, st); });
          if (ms < best) { best = ms; bi = v; }
        }
        if (bi < 0) { std::printf("%-10s %5d | no tile fits N=%d\n", s.name, T, N); continue; }
        V[bi].launch(dXq, dXs, dWq, dWs, dYb, T, K, N, st);
        CK(hipStreamSynchronize(st));
        CK(hipMemcpy(yB.data(), dYb, yB.size() * 4, hipMemcpyDeviceToHost));
        double rl2, mr;
        err_vs(yA, yB, rl2, mr);
        const float msB = msRq + msAq + best;
        const double tflopsA = 2.0 * T * N * K / (msA - msA_dq) / 1e9;
        const double tflopsB = 2.0 * T * N * K / best / 1e9;
        std::printf("%-10s %5d | %6.3f (dq %5.3f, %5.1f TF) | g%-3d %6.3f (rq %5.3f aq %5.3f gemm %6.3f %s %5.1f TF) | %4.2fx  relL2 %.2e max %.2e\n",
                    s.name, T, msA, msA_dq, tflopsA, GRP, msB, msRq, msAq, best, V[bi].name, tflopsB, msA / msB, rl2, mr);
      }
    }
  }
  CKB(hipblasDestroy(hb));
  return 0;
}
