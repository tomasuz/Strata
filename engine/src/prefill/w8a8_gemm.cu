// src/prefill/w8a8_gemm.cu - see include/strata/prefill/w8a8_gemm.hpp.
//
// The int8 byte order (radiance's w8 fragment plane, gfx12 v_wmma_i32_16x16x16_iu8 operands): byte j (0..15) of
// lane l (0..31) in the 512-byte run of a (16-row tile t, 32-value k-block kb) is
//     W[16 t + (l & 15)][32 kb + 16 (j >> 3) + 8 ((j & 7) >> 2) + 4 (l >> 4) + (j & 3)]
// and an activation row is stored per 16 values with the same in-block permutation, so a lane's two dwords of A
// (LDS row, column 4 u + 2 (lane >> 4)) and of B line up on the same k.
#include "strata/prefill/w8a8_gemm.hpp"

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

#include "strata/kernels/gfx_arch.hpp"

#include <atomic>

#if defined(__gfx1200__) || defined(__gfx1201__)
#define STRATA_W8A8_DEVICE 1
#endif

namespace strata::prefill::w8a8 {
namespace {

constexpr int BK = 32;                  // k per pipeline stage (two 16-deep WMMA steps)
constexpr int MAGIC = 0x4B400000;       // the bits of 1.5 * 2^23
constexpr float MAGICF = 12582912.0f;

typedef int v2i __attribute__((ext_vector_type(2)));
typedef int v8i __attribute__((ext_vector_type(8)));
typedef float v8f __attribute__((ext_vector_type(8)));
typedef unsigned u4 __attribute__((ext_vector_type(4)));

// the source index (within its 16-value block) of activation byte j
__device__ __forceinline__ int act_src(int j) { return 8 * ((j & 7) >> 2) + 4 * (j >> 3) + (j & 3); }

__device__ __forceinline__ int q8(float v) {
    const int q = __float2int_rn(v);
    return q < -127 ? -127 : (q > 127 ? 127 : q);
}

// ---- weights: FP16 rows -> int8 fragment plane + FP16 scale per (row, GRP).  A workgroup: 16 rows x GRP values,
// 8 values a thread (TPR threads a row, adjacent lanes, so the row's absmax is a shuffle reduction).
template <int GRP>
__global__ __launch_bounds__(16 * GRP / 8) void requant_f16_kernel(const __half* __restrict__ W, int64_t ldw, int K,
                                                                   uint8_t* __restrict__ Wq, __half* __restrict__ Ws) {
#if defined(STRATA_W8A8_DEVICE)
    constexpr int TPR = GRP / 8;
    static_assert(TPR <= 32 && (32 % TPR) == 0, "a row's threads within one wave");
    const int t = blockIdx.x, g = blockIdx.y;
    const int r = threadIdx.x / TPR, c8 = threadIdx.x % TPR;
    const int64_t n = (int64_t) t * 16 + r;
    const int k0 = g * GRP + c8 * 8;
    const u4 raw = *(const u4*) (W + n * ldw + k0);
    const __half* h = (const __half*) &raw;
    float v[8], m = 0.f;
#pragma unroll
    for (int e = 0; e < 8; e++) {
        v[e] = __half2float(h[e]);
        m = fmaxf(m, fabsf(v[e]));
    }
#pragma unroll
    for (int o = TPR / 2; o > 0; o >>= 1) m = fmaxf(m, __shfl_xor(m, o, 32));
    const float s = __half2float(__float2half(fmaxf(m, 1e-12f) / 127.f));   // the FP16 scale the GEMM reads
    const float inv = 1.f / s;
    const int nsz = K / GRP, kpb = K / 32;
    if (c8 == 0) Ws[((int64_t) t * nsz + g) * 16 + r] = __float2half(s);
    // k0..k0+3 and k0+4..k0+7 are 4 consecutive bytes of lanes r and 16 + r (k0 a multiple of 8)
    const int kb = k0 >> 5, kk = k0 & 31;
    const int u = kk >> 4, a = (kk & 15) >> 3;
    const int j = u * 8 + a * 4;
    uint32_t lo = 0, hi = 0;
#pragma unroll
    for (int e = 0; e < 4; e++) {
        lo |= (uint32_t) (uint8_t) (int8_t) q8(v[e] * inv) << (8 * e);
        hi |= (uint32_t) (uint8_t) (int8_t) q8(v[4 + e] * inv) << (8 * e);
    }
    uint8_t* run = Wq + ((int64_t) t * kpb + kb) * 512;
    *(uint32_t*) (run + r * 16 + j) = lo;
    *(uint32_t*) (run + (16 + r) * 16 + j) = hi;
#endif
}

// ---- activations: FP16 rows -> int8 (the in-block permutation) + FP32 scale per row (AG 0) or (row, AG).
// A workgroup per row (Ms of them: rows past T only zero their scales); a thread per 16-value block at a time.
template <int AG>
__global__ __launch_bounds__(256) void quant_act_kernel(const __half* __restrict__ X, int64_t ldx, int T, int K, int Ms,
                                                        int8_t* __restrict__ Q, float* __restrict__ S) {
#if defined(STRATA_W8A8_DEVICE)
    const int row = blockIdx.x, tid = threadIdx.x;
    const int nb = K / 16;
    if (row >= T) {
        const int ns = AG > 0 ? K / AG : 1;
        for (int g = tid; g < ns; g += 256) S[(int64_t) g * Ms + row] = 0.f;
        return;
    }
    const __half* x = X + (int64_t) row * ldx;
    int8_t* q = Q + (int64_t) row * K;
    auto quant_block = [&](int b, const float (&v)[16], float inv) {
        uint32_t w[4];
#pragma unroll
        for (int d = 0; d < 4; d++) {
            uint32_t acc = 0;
#pragma unroll
            for (int e = 0; e < 4; e++) acc |= (uint32_t) (uint8_t) (int8_t) q8(v[act_src(4 * d + e)] * inv) << (8 * e);
            w[d] = acc;
        }
        *(u4*) (q + (int64_t) b * 16) = u4{w[0], w[1], w[2], w[3]};
    };
    auto load_block = [&](int b, float (&v)[16]) {
        const u4 r0 = *(const u4*) (x + b * 16), r1 = *(const u4*) (x + b * 16 + 8);
        const __half* h0 = (const __half*) &r0;
        const __half* h1 = (const __half*) &r1;
#pragma unroll
        for (int e = 0; e < 8; e++) {
            v[e] = __half2float(h0[e]);
            v[8 + e] = __half2float(h1[e]);
        }
    };
    if constexpr (AG == 0) {
        float m = 0.f;
        for (int b = tid; b < nb; b += 256) {
            float v[16];
            load_block(b, v);
#pragma unroll
            for (int e = 0; e < 16; e++) m = fmaxf(m, fabsf(v[e]));
        }
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) m = fmaxf(m, __shfl_xor(m, o, 32));
        __shared__ float red[8];
        if ((tid & 31) == 0) red[tid >> 5] = m;
        __syncthreads();
        m = red[0];
#pragma unroll
        for (int i = 1; i < 8; i++) m = fmaxf(m, red[i]);
        const float s = fmaxf(m, 1e-8f) / 127.f, inv = 1.f / s;
        if (tid == 0) S[row] = s;
        for (int b = tid; b < nb; b += 256) {
            float v[16];
            load_block(b, v);
            quant_block(b, v, inv);
        }
    } else {
        constexpr int TPG = AG / 16;   // adjacent threads of a group; K % AG == 0, so a group never straddles the loop's end
        static_assert(TPG >= 1 && TPG <= 32 && (256 % TPG) == 0, "group threads within one wave");
        for (int b = tid; b < nb; b += 256) {
            float v[16], m = 0.f;
            load_block(b, v);
#pragma unroll
            for (int e = 0; e < 16; e++) m = fmaxf(m, fabsf(v[e]));
#pragma unroll
            for (int o = TPG / 2; o > 0; o >>= 1) m = fmaxf(m, __shfl_xor(m, o, 32));
            const float s = fmaxf(m, 1e-8f) / 127.f;
            if (b % TPG == 0) S[(int64_t) (b / TPG) * Ms + row] = s;
            quant_block(b, v, 1.f / s);
        }
    }
#endif
}

// ---- the GEMM.  WM x WN waves; a wave TM x TN 16 x 16 tiles (M = tokens, N = weight rows).  The weight scale
// changes every GRP values of K, the activation scale (AG > 0) every AG: the int32 accumulators are flushed into the
// FP32 ones every FL = min(GRP, AG) values (AG 0: every GRP, the activation scale applied once at the end).
template <int WM, int WN, int TM, int TN, int GRP, int AG>
__global__ __launch_bounds__(WM * WN * 32, 8) void gemm_kernel(const signed char* __restrict__ A,
                                                               const float* __restrict__ As, int Ms,
                                                               const unsigned* __restrict__ Wq,
                                                               const unsigned short* __restrict__ Ws,
                                                               float* __restrict__ Y, int M, int K, int N, int64_t ldy,
                                                               float beta) {
#if defined(STRATA_W8A8_DEVICE)
    constexpr int UNR = GRP / BK;
    constexpr int FL = (AG > 0 && AG < GRP) ? AG : GRP;
    constexpr int BM = WM * TM * 16, BN = WN * TN * 16;
    constexpr int NTILE = BN / 16;
    constexpr int AW = BM * BK / 4;
    constexpr int WW = NTILE * BK * 4;
    static_assert(GRP % BK == 0 && FL % BK == 0 && GRP % FL == 0, "scale windows of whole k steps");

    __shared__ unsigned lds_a[2][AW];
    __shared__ unsigned lds_w[2][WW];

    const int tid = threadIdx.x, lane = tid & 31, wave = tid >> 5;
    const int wm = wave / WN, wn = wave % WN;
    const int m0 = blockIdx.y * BM, n0 = blockIdx.x * BN;
    const int kpb = K / 32, nsz = K / GRP;
    constexpr int NT_ = WM * WN * 32;

    v8f acc[TN][TM];
#pragma unroll
    for (int j = 0; j < TN; j++)
#pragma unroll
        for (int i = 0; i < TM; i++) acc[j][i] = (v8f) (0.f);
    v8i g[TN][TM];
    float csum[TN];
#pragma unroll
    for (int j = 0; j < TN; j++) csum[j] = 0.f;
    float sw[TN];

#define ASW_M2 ((BK / 8) - 1)
#define ASW(r, c) ((r) * (BK / 4) + ((c) ^ ((((r) >> 1) & ASW_M2) << 1)))
    constexpr int APT = (AW + NT_ - 1) / NT_;
    constexpr bool APAD = (AW % NT_) != 0;
    unsigned ra[APT];
    constexpr int WWP = NTILE * BK;
    constexpr int WPTU = (WWP + NT_ - 1) / NT_;
    constexpr bool WPAD = (WWP % NT_) != 0;
    u4 rwu[WPTU];
    auto issue = [&](int kk) {
#pragma unroll
        for (int x = 0; x < WPTU; x++) {
            const int t = tid + x * NT_;
            if (WPAD && t >= WWP) continue;
            const int tl = t / BK, rem = t % BK, kh = rem / 32, ln = rem % 32;
            rwu[x] = *((const u4*) Wq + (size_t) (((n0 / 16 + tl) * kpb + kk / 32 + kh) * 32 + ln));
        }
#pragma unroll
        for (int x = 0; x < APT; x++) {
            const int t = tid + x * NT_, r = t / (BK / 4), c = t % (BK / 4);
            if (APAD && t >= AW) continue;
            const int mm = m0 + r < M ? m0 + r : M - 1;
            ra[x] = *(const unsigned*) (A + ((size_t) mm * K + kk + c * 4));
        }
    };
    auto commit = [&](int buf) {
#pragma unroll
        for (int x = 0; x < WPTU; x++) {
            const int t = tid + x * NT_;
            if (WPAD && t >= WWP) continue;
            const int tl = t / BK, rem = t % BK, kh = rem / 32, ln = rem % 32;
            *(u4*) (lds_w[buf] + (size_t) tl * (BK * 4) + ln * (BK / 8) + kh * 4) = rwu[x];
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
            const bool fresh = (k0 % FL) == 0;
            if ((k0 % GRP) == 0) {
#pragma unroll
                for (int j = 0; j < TN; j++)
                    sw[j] = __half2float(__ushort_as_half(
                        Ws[(unsigned) (((n0 / 16 + wn * TN + j) * nsz + k0 / GRP) * 16 + (lane & 15))]));
            }
#pragma unroll
            for (int u = 0; u < BK / 16; u++) {
                v2i af[TM];
#pragma unroll
                for (int i = 0; i < TM; i++) {
                    const int row = (wm * TM + i) * 16 + (lane & 15);
                    const unsigned* pa = lds_a[buf] + ASW(row, u * 4 + 2 * (lane >> 4));
                    af[i][0] = (int) pa[0];
                    af[i][1] = (int) pa[1];
                }
#pragma unroll
                for (int j = 0; j < TN; j++) {
                    const v2i bf = *(const v2i*) (lds_w[buf] + (size_t) (wn * TN + j) * (BK * 4) + lane * (BK / 8) + u * 2);
#pragma unroll
                    for (int i = 0; i < TM; i++)
                        g[j][i] = (fresh && u == 0)
                                      ? __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(true, af[i], true, bf, (v8i) (MAGIC), false)
                                      : __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(true, af[i], true, bf, g[j][i], false);
                }
            }
            if ((k0 + BK) % FL == 0) {
                if constexpr (AG > 0) {
                    // the activation scales of this window, read here (not held across it: registers): this lane's
                    // 8 rows (8 (lane >> 4) + e of each 16-row tile) are consecutive, two float4 a tile
#pragma unroll
                    for (int i = 0; i < TM; i++) {
                        const float* p = As + (size_t) (k0 / AG) * Ms + m0 + (wm * TM + i) * 16 + 8 * (lane >> 4);
                        const float4 a0 = *(const float4*) p, a1 = *(const float4*) (p + 4);
                        const float sa[8] = {a0.x, a0.y, a0.z, a0.w, a1.x, a1.y, a1.z, a1.w};
#pragma unroll
                        for (int j = 0; j < TN; j++)
#pragma unroll
                            for (int e = 0; e < 8; e++)
                                acc[j][i][e] = fmaf(__uint_as_float((unsigned) g[j][i][e]) - MAGICF, sw[j] * sa[e], acc[j][i][e]);
                    }
                } else {
#pragma unroll
                    for (int j = 0; j < TN; j++) csum[j] += sw[j];
#pragma unroll
                    for (int i = 0; i < TM; i++)
#pragma unroll
                        for (int j = 0; j < TN; j++)
#pragma unroll
                            for (int e = 0; e < 8; e++)
                                acc[j][i][e] = fmaf(__uint_as_float((unsigned) g[j][i][e]), sw[j], acc[j][i][e]);
                }
            }
            if (k0 + BK < K) commit(buf ^ 1);
            __syncthreads();
        }
    }
#undef ASW
#undef ASW_M2
    if constexpr (AG == 0) {
#pragma unroll
        for (int j = 0; j < TN; j++) {
            const float d = -MAGICF * csum[j];
#pragma unroll
            for (int i = 0; i < TM; i++)
#pragma unroll
                for (int e = 0; e < 8; e++) acc[j][i][e] += d;
        }
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
                    float* y = Y + (size_t) m * ldy + n;
                    const float v = AG > 0 ? acc[j][i][e] : acc[j][i][e] * As[m];
                    *y = beta == 0.f ? v : fmaf(beta, *y, v);
                }
            }
#endif
}

// ---- host side
bool device_ok() {
    static std::atomic<int> cache[64];   // 0 unknown, 1 yes, 2 no
    int dev = 0;
    if (hipGetDevice(&dev) != hipSuccess || dev < 0 || dev >= 64) return false;
    int c = cache[dev].load(std::memory_order_acquire);
    if (c == 0) {
        hipDeviceProp_t p{};
        const bool ok = hipGetDeviceProperties(&p, dev) == hipSuccess && strata::kernels::gfx_arch_is_gfx12_wmma(p.gcnArchName);
        c = ok ? 1 : 2;
        cache[dev].store(c, std::memory_order_release);
    }
    return c == 1;
}

using Launch = void (*)(const void*, const float*, int, const void*, const void*, float*, int, int, int, int64_t, float,
                        hipStream_t);

template <int WM, int WN, int TM, int TN, int AG>
void launch(const void* xq, const float* xs, int ms, const void* wq, const void* ws, float* y, int M, int K, int N,
            int64_t ldy, float beta, hipStream_t st) {
    constexpr int BM = WM * TM * 16, BN = WN * TN * 16;
    const dim3 grid((unsigned) (N / BN), (unsigned) ((M + BM - 1) / BM)), block(WM * WN * 32);
    hipLaunchKernelGGL((gemm_kernel<WM, WN, TM, TN, kGroup, AG>), grid, block, 0, st, (const signed char*) xq, xs, ms,
                       (const unsigned*) wq, (const unsigned short*) ws, y, M, K, N, ldy, beta);
}

struct Tile {
    const char* name;
    int BN;
    Launch by_ag[3];   // ag 0, 32, 128
};
#define STRATA_W8A8_TILE(name, WM, WN, TM, TN) \
    {name, WN * TN * 16, {launch<WM, WN, TM, TN, 0>, launch<WM, WN, TM, TN, 32>, launch<WM, WN, TM, TN, 128>}}
const Tile kTiles[] = {
    STRATA_W8A8_TILE("128x256", 2, 8, 4, 2),
    STRATA_W8A8_TILE("64x256", 2, 8, 2, 2),
    STRATA_W8A8_TILE("128x128", 2, 8, 4, 1),
    STRATA_W8A8_TILE("64x128", 1, 8, 4, 1),
};
#undef STRATA_W8A8_TILE
constexpr int kNTiles = (int) (sizeof(kTiles) / sizeof(kTiles[0]));

int ag_index(int ag) { return ag == 0 ? 0 : ag == 32 ? 1 : ag == 128 ? 2 : -1; }

}  // namespace

bool available() { return device_ok(); }

int tile_count() { return kNTiles; }
const char* tile_name(int tile) { return tile >= 0 && tile < kNTiles ? kTiles[tile].name : "?"; }

int default_tile(int64_t T, int64_t N) {
    if (N % 256 == 0 && N >= 2048) return T >= 1024 ? 0 : 1;
    if (N % 128 == 0) return T >= 1024 ? 2 : 3;
    return -1;
}

bool requant_f16(const uint16_t* W16, int64_t ldw, int64_t N, int64_t K, void* Wq, void* Ws, void* stream) {
    if (!device_ok() || N <= 0 || N % 16 != 0 || K <= 0 || K % kGroup != 0 || ldw < K || ldw % 8 != 0) return false;
    hipLaunchKernelGGL(requant_f16_kernel<kGroup>, dim3((unsigned) (N / 16), (unsigned) (K / kGroup)), dim3(16 * kGroup / 8), 0,
                       (hipStream_t) stream, (const __half*) W16, ldw, (int) K, (uint8_t*) Wq, (__half*) Ws);
    return true;
}

bool quant_act_f16(const uint16_t* X, int64_t ldx, int64_t T, int64_t K, int ag, void* Xq, float* Xs, void* stream) {
    if (!device_ok() || T <= 0 || K <= 0 || K % kGroup != 0 || ldx < K || ldx % 8 != 0 || ag_index(ag) < 0) return false;
    const int ms = (int) xs_rows(T);
    const dim3 grid((unsigned) ms), block(256);
    hipStream_t st = (hipStream_t) stream;
    const __half* x = (const __half*) X;
    if (ag == 0) hipLaunchKernelGGL(quant_act_kernel<0>, grid, block, 0, st, x, ldx, (int) T, (int) K, ms, (int8_t*) Xq, Xs);
    else if (ag == 32) hipLaunchKernelGGL(quant_act_kernel<32>, grid, block, 0, st, x, ldx, (int) T, (int) K, ms, (int8_t*) Xq, Xs);
    else hipLaunchKernelGGL(quant_act_kernel<128>, grid, block, 0, st, x, ldx, (int) T, (int) K, ms, (int8_t*) Xq, Xs);
    return true;
}

bool gemm(const void* Xq, const float* Xs, int ag, const void* Wq, const void* Ws, float* Y, int64_t T, int64_t N,
          int64_t K, int64_t ldy, float beta, void* stream, int tile) {
    if (!device_ok() || T <= 0 || N <= 0 || K <= 0 || K % kGroup != 0 || ag_index(ag) < 0) return false;
    if (beta != 0.f && beta != 1.f) return false;
    if (ldy <= 0) ldy = N;
    if (ldy < N) return false;
    if (T > INT32_MAX / 2 || T * K > (int64_t) UINT32_MAX) return false;
    if (tile < 0) tile = default_tile(T, N);
    if (tile < 0 || tile >= kNTiles || N % kTiles[tile].BN != 0) return false;
    kTiles[tile].by_ag[ag_index(ag)](Xq, Xs, (int) xs_rows(T), Wq, Ws, Y, (int) T, (int) K, (int) N, ldy, beta,
                                      (hipStream_t) stream);
    return true;
}

}  // namespace strata::prefill::w8a8
