// src/prefill/w8a8_gemm_parity.cu - strata::prefill::w8a8 (gfx12 int8 GEMM) against an FP64 reference on the same FP16
// weights and activations.  GPU, synthetic, no model; exit 77 off gfx12 / without a card.
//
//     w8a8_gemm_parity                      the checks (a ctest): Qwen3.8-Flash-Next's dense prompt shapes, T 300 and 2048
//     w8a8_gemm_parity --bench [T ...]      + timings: hipBLAS FP16 GEMM vs requant / activation quant / int8 GEMM per tile
//
// Per shape, T, activation distribution and activation scale mode (ag 0 = per row, 32, 128): the relative L2 error
// on sampled rows against FP64, every output finite, the ldy padding untouched, beta = 1 accumulating, the same bits
// from every tile and from a second run.  Distributions: plain N(0, 1); hot8 - 8 fixed channels x 30 (the few hot
// channels of RMS-normed hidden states); every97 - every 97th channel x 20 (the case a per-group scale cannot help).
#include "strata/prefill/w8a8_gemm.hpp"

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <hipblas/hipblas.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

namespace w = strata::prefill::w8a8;

namespace {

int g_fail = 0;

#define CK(x) do { hipError_t e_ = (x); if (e_ != hipSuccess) { \
    std::fprintf(stderr, "%s:%d %s: %s\n", __FILE__, __LINE__, #x, hipGetErrorString(e_)); std::exit(1); } } while (0)

void check(bool ok, const char* what, const std::string& where) {
    if (!ok) {
        ++g_fail;
        std::printf("FAIL %s: %s\n", where.c_str(), what);
    }
}

uint16_t f2h(float f) { __half h = __float2half(f); uint16_t u; std::memcpy(&u, &h, 2); return u; }
float h2f(uint16_t u) { __half h; std::memcpy(&h, &u, 2); return __half2float(h); }

struct Shape { const char* name; int N, K; };
// GGUF ne1 = N, ne0 = K
const Shape kShapes[] = {
    {"attn_qkv", 10240, 2560}, {"attn_q", 12288, 2560}, {"attn_gate", 6144, 2560},
    {"ssm_out", 2560, 6144},   {"shexp_up", 640, 2560},
};
const char* kDist[] = {"plain", "hot8", "every97"};
const int kAg[] = {0, 32, 128};

void make_x(int dist, int T, int K, std::mt19937& rng, std::vector<uint16_t>& x) {
    x.resize((size_t) T * K);
    std::normal_distribution<float> nd(0.f, 1.f);
    const int hot[8] = {3, 77, 500, 901, 1300, 1777, 2048, 2400};
    std::vector<float> mul(K, 1.f);
    if (dist == 1) for (int c : hot) mul[c % K] = 30.f;
    if (dist == 2) for (int c = 0; c < K; c++) if (c % 97 == 3) mul[c] = 20.f;
    for (int t = 0; t < T; t++)
        for (int k = 0; k < K; k++) x[(size_t) t * K + k] = f2h(nd(rng) * mul[k]);
}

template <class F> float time_ms(hipStream_t st, int iters, F&& f) {
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

}  // namespace

int main(int argc, char** argv) {
    bool bench = false;
    std::vector<int> Ts;
    for (int i = 1; i < argc; i++) {
        if (std::strcmp(argv[i], "--bench") == 0) bench = true;
        else Ts.push_back(std::atoi(argv[i]));
    }
    int ndev = 0;
    if (hipGetDeviceCount(&ndev) != hipSuccess || ndev == 0) { std::printf("no GPU: skipped\n"); return 77; }
    if (!w::available()) { std::printf("not gfx12: skipped\n"); return 77; }
    if (Ts.empty()) Ts = bench ? std::vector<int>{2048, 4096, 16384} : std::vector<int>{300, 2048};
    const int iters = std::getenv("W8A8_ITERS") ? std::atoi(std::getenv("W8A8_ITERS")) : 20;

    hipStream_t st;
    CK(hipStreamCreate(&st));
    hipblasHandle_t hb = nullptr;
    if (bench) {
        if (hipblasCreate(&hb) != HIPBLAS_STATUS_SUCCESS) { std::printf("hipblasCreate failed\n"); return 1; }
        hipblasSetStream(hb, st);
    }
    std::mt19937 rng(1234);
    const int Tmax = *std::max_element(Ts.begin(), Ts.end());

    for (const Shape& s : kShapes) {
        const int N = s.N, K = s.K;
        const int64_t ldy = N + 16;   // padding columns that must stay untouched
        std::vector<uint16_t> hw((size_t) N * K);
        {
            std::normal_distribution<float> nd(0.f, 0.02f);
            std::uniform_real_distribution<float> rf(0.5f, 2.f);
            for (int n = 0; n < N; n++) {
                const float f = rf(rng);
                for (int k = 0; k < K; k++) hw[(size_t) n * K + k] = f2h(nd(rng) * f);
            }
        }
        std::vector<float> wf(hw.size());
        for (size_t i = 0; i < hw.size(); i++) wf[i] = h2f(hw[i]);

        uint16_t *dW16, *dX;
        void *dWq, *dWs, *dXq;
        float *dXs, *dY, *dY2;
        CK(hipMalloc(&dW16, hw.size() * 2));
        CK(hipMalloc(&dWq, w::wq_bytes(N, K)));
        CK(hipMalloc(&dWs, w::ws_bytes(N, K)));
        CK(hipMalloc(&dX, (size_t) Tmax * K * 2));
        CK(hipMalloc(&dXq, w::xq_bytes(Tmax, K)));
        CK(hipMalloc(&dXs, w::xs_bytes(Tmax, K, 32)));
        CK(hipMalloc(&dY, (size_t) Tmax * ldy * 4));
        CK(hipMalloc(&dY2, (size_t) Tmax * ldy * 4));
        CK(hipMemcpy(dW16, hw.data(), hw.size() * 2, hipMemcpyHostToDevice));
        check(w::requant_f16(dW16, K, N, K, dWq, dWs, st), "requant_f16 refused", s.name);

        for (int T : Ts) {
            const std::string at = std::string(s.name) + " T=" + std::to_string(T);
            if (!bench) {
                std::vector<int> rows = {0, 1, T - 1};
                std::uniform_int_distribution<int> rr(0, T - 1);
                while ((int) rows.size() < 24) rows.push_back(rr(rng));
                std::vector<uint16_t> hx;
                std::vector<float> hy((size_t) T * ldy), hy2((size_t) T * ldy), y0((size_t) T * ldy);
                for (int d = 0; d < 3; d++) {
                    make_x(d, T, K, rng, hx);
                    CK(hipMemcpy(dX, hx.data(), hx.size() * 2, hipMemcpyHostToDevice));
                    std::vector<double> ref(rows.size() * N);
                    for (size_t r = 0; r < rows.size(); r++) {
                        const uint16_t* xr = hx.data() + (size_t) rows[r] * K;
                        std::vector<double> xd(K);
                        for (int k = 0; k < K; k++) xd[k] = h2f(xr[k]);
                        for (int n = 0; n < N; n++) {
                            const float* wr = wf.data() + (size_t) n * K;
                            double a = 0;
                            for (int k = 0; k < K; k++) a += xd[k] * wr[k];
                            ref[r * N + n] = a;
                        }
                    }
                    std::printf("%-9s T=%-5d %-8s", s.name, T, kDist[d]);
                    for (int ag : kAg) {
                        const std::string where = at + " " + kDist[d] + " ag=" + std::to_string(ag);
                        check(w::quant_act_f16(dX, K, T, K, ag, dXq, dXs, st), "quant_act_f16 refused", where);
                        CK(hipMemsetD32Async((hipDeviceptr_t) dY, 0x7FC00000, (size_t) T * ldy, st));   // NaN
                        const int def = w::default_tile(T, N);
                        check(w::gemm(dXq, dXs, ag, dWq, dWs, dY, T, N, K, ldy, 0.f, st), "gemm refused", where);
                        CK(hipStreamSynchronize(st));
                        CK(hipMemcpy(hy.data(), dY, hy.size() * 4, hipMemcpyDeviceToHost));
                        bool finite = true, pad = true;
                        for (int t = 0; t < T; t++) {
                            for (int n = 0; n < N; n++) finite &= std::isfinite(hy[(size_t) t * ldy + n]);
                            for (int n = N; n < ldy; n++) pad &= std::isnan(hy[(size_t) t * ldy + n]);
                        }
                        check(finite, "non-finite or unwritten output", where);
                        check(pad, "ldy padding written", where);
                        double num = 0, den = 0;
                        for (size_t r = 0; r < rows.size(); r++)
                            for (int n = 0; n < N; n++) {
                                const double dd = hy[(size_t) rows[r] * ldy + n] - ref[r * N + n];
                                num += dd * dd;
                                den += ref[r * N + n] * ref[r * N + n];
                            }
                        const double rel = std::sqrt(num / std::max(den, 1e-30));
                        std::printf("  ag%-3d %.2e", ag, rel);
                        check(rel < (d == 0 ? 2e-2 : 8e-2), "relative L2 error too large", where);
                        // every tile that covers N: the same k order, so the same bits
                        for (int tl = 0; tl < w::tile_count(); tl++) {
                            if (tl == def) continue;
                            if (!w::gemm(dXq, dXs, ag, dWq, dWs, dY2, T, N, K, ldy, 0.f, st, tl)) continue;
                            CK(hipStreamSynchronize(st));
                            CK(hipMemcpy(hy2.data(), dY2, hy2.size() * 4, hipMemcpyDeviceToHost));
                            bool same = true;
                            for (int t = 0; t < T && same; t++)
                                same = std::memcmp(&hy[(size_t) t * ldy], &hy2[(size_t) t * ldy], (size_t) N * 4) == 0;
                            check(same, (std::string("tile ") + w::tile_name(tl) + " differs from the default").c_str(), where);
                        }
                        // a second run: the same bits
                        check(w::gemm(dXq, dXs, ag, dWq, dWs, dY2, T, N, K, ldy, 0.f, st), "gemm refused", where);
                        CK(hipStreamSynchronize(st));
                        CK(hipMemcpy(hy2.data(), dY2, hy2.size() * 4, hipMemcpyDeviceToHost));
                        bool same = true;
                        for (int t = 0; t < T && same; t++)
                            same = std::memcmp(&hy[(size_t) t * ldy], &hy2[(size_t) t * ldy], (size_t) N * 4) == 0;
                        check(same, "second run differs", where);
                        // beta = 1: Y0 + product
                        std::uniform_real_distribution<float> u(-1.f, 1.f);
                        for (auto& v : y0) v = u(rng);
                        CK(hipMemcpy(dY2, y0.data(), y0.size() * 4, hipMemcpyHostToDevice));
                        check(w::gemm(dXq, dXs, ag, dWq, dWs, dY2, T, N, K, ldy, 1.f, st), "gemm beta 1 refused", where);
                        CK(hipStreamSynchronize(st));
                        CK(hipMemcpy(hy2.data(), dY2, hy2.size() * 4, hipMemcpyDeviceToHost));
                        double worst = 0;
                        for (int t = 0; t < T; t++)
                            for (int n = 0; n < ldy; n++) {
                                const size_t i = (size_t) t * ldy + n;
                                const double want = n < N ? (double) y0[i] + hy[i] : (double) y0[i];
                                worst = std::max(worst, std::fabs(hy2[i] - want) / (1.0 + std::fabs(want)));
                            }
                        check(worst < 1e-5, "beta = 1 does not accumulate", where);
                    }
                    std::printf("\n");
                }
            } else {
                std::vector<uint16_t> hx;
                make_x(1, T, K, rng, hx);
                CK(hipMemcpy(dX, hx.data(), hx.size() * 2, hipMemcpyHostToDevice));
                const float one = 1.f, zero = 0.f;
                const double fl = 2.0 * T * N * K;
                const float ms_hb = time_ms(st, iters, [&] {
                    hipblasGemmEx(hb, HIPBLAS_OP_T, HIPBLAS_OP_N, N, T, K, &one, dW16, HIP_R_16F, K, dX, HIP_R_16F, K, &zero,
                                  dY, HIP_R_32F, (int) ldy, HIPBLAS_COMPUTE_32F, HIPBLAS_GEMM_DEFAULT);
                });
                const float ms_rq = time_ms(st, iters, [&] { w::requant_f16(dW16, K, N, K, dWq, dWs, st); });
                std::printf("%-9s T=%-5d hipBLAS f16 %7.3f ms %5.1f TF | requant %.3f ms\n", s.name, T, ms_hb, fl / ms_hb / 1e9, ms_rq);
                for (int ag : kAg) {
                    const float ms_aq = time_ms(st, iters, [&] { w::quant_act_f16(dX, K, T, K, ag, dXq, dXs, st); });
                    std::printf("    ag%-3d actq %.3f ms |", ag, ms_aq);
                    for (int tl = 0; tl < w::tile_count(); tl++) {
                        if (!w::gemm(dXq, dXs, ag, dWq, dWs, dY, T, N, K, ldy, 0.f, st, tl)) continue;
                        const float ms = time_ms(st, iters, [&] { w::gemm(dXq, dXs, ag, dWq, dWs, dY, T, N, K, ldy, 0.f, st, tl); });
                        std::printf(" %s%s %.3f ms %5.1f TF", w::tile_name(tl), tl == w::default_tile(T, N) ? "*" : "", ms, fl / ms / 1e9);
                    }
                    std::printf("\n");
                }
            }
        }
        CK(hipFree(dW16)); CK(hipFree(dWq)); CK(hipFree(dWs)); CK(hipFree(dX)); CK(hipFree(dXq)); CK(hipFree(dXs));
        CK(hipFree(dY)); CK(hipFree(dY2));
    }
    if (hb) hipblasDestroy(hb);
    std::printf(g_fail ? "w8a8_gemm_parity: %d FAILED\n" : "w8a8_gemm_parity: all passed\n", g_fail);
    return g_fail ? 1 : 0;
}
