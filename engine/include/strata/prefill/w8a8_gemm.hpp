// include/strata/prefill/w8a8_gemm.hpp - gfx12 (RDNA4) int8 x int8 GEMM for the dense prompt projections.
//
//     Y[t, n] = beta * Y[t, n] + sum_k W[n, k] X[t, k]          (beta 0 or 1, Y FP32 with row stride ldy >= N)
//
// W is held as an int8 plane in the gfx12 WMMA fragment order with an FP16 scale per (row, kGroup values of K); X as
// int8 rows in the same byte order with an FP32 scale per row (ag 0) or per (row, ag values of K) (ag 32 / 128).
// Within a scale window the int32 dot starts from 0x4B400000 (1.5 * 2^23), so its bits read as a float are
// 1.5 * 2^23 + dot exactly (|dot| <= 128 * 127 * 127 < 2^22) and one subtraction turns it into the dot.
//
// The GEMM body is radiance's r4d_gemm_w8a8_tiled_kernel (codeberg.org/StillDeadcode/radiance, libr4d/
// r4d_gemm_w8a8_tiled.hip, Apache-2.0; see NOTICE) with the scale group a template argument, the activation scale
// per group (ag > 0) and an FP32 (ldy, beta) epilogue.
//
// Every entry point returns false and launches nothing when the current device is not gfx1200 / gfx1201 or the
// shape is not covered (K a multiple of kGroup and of ag, N a multiple of 16 for the planes, of the tile's width
// for the GEMM), so a caller keeps its FP16 path as the fallback.
#pragma once

#include <cstddef>
#include <cstdint>

namespace strata::prefill::w8a8 {

constexpr int kGroup = 128;   // weight scale group along K (g32 measured slower for no accuracy gain on Q6_K)

/// The current device runs these kernels (gfx1200 / gfx1201), cached per device.
bool available();

/// Bytes of the int8 weight plane (N x K) and of its FP16 scale plane (N x K / kGroup).
inline size_t wq_bytes(int64_t N, int64_t K) { return (size_t) N * (size_t) K; }
inline size_t ws_bytes(int64_t N, int64_t K) { return (size_t) N * (size_t) (K / kGroup) * 2; }
/// Row stride of the activation scale plane: T rounded up to 128, the tallest tile (the GEMM reads a lane's 8 rows
/// as two float4, the last tile's rows past T included; those scales are 0).
inline int64_t xs_rows(int64_t T) { return (T + 127) / 128 * 128; }
/// Bytes of the int8 activation rows (T x K) and of their FP32 scales (one per row, or per row and ag values).
inline size_t xq_bytes(int64_t T, int64_t K) { return (size_t) T * (size_t) K; }
inline size_t xs_bytes(int64_t T, int64_t K, int ag) { return (size_t) xs_rows(T) * (size_t) (ag > 0 ? K / ag : 1) * 4; }

/// W16 (N x K FP16, row stride ldw >= K, a multiple of 8) -> Wq (wq_bytes) + Ws (ws_bytes): symmetric int8 per
/// (row, kGroup).  N a multiple of 16, K of kGroup.
bool requant_f16(const uint16_t* W16, int64_t ldw, int64_t N, int64_t K, void* Wq, void* Ws, void* stream);

/// X (T x K FP16, row stride ldx >= K, a multiple of 8) -> Xq (xq_bytes) + Xs (xs_bytes): symmetric int8 per row
/// (ag 0) or per (row, ag values) (ag 32 / 128).  K a multiple of kGroup.  Xs rows past T get scale 0.
bool quant_act_f16(const uint16_t* X, int64_t ldx, int64_t T, int64_t K, int ag, void* Xq, float* Xs, void* stream);

/// Y = beta * Y + Xq . Wq^T with the scales above.  tile < 0: the default for the shape; else an index below
/// tile_count() (benchmarks).  False (nothing launched) when the tile does not divide N.
bool gemm(const void* Xq, const float* Xs, int ag, const void* Wq, const void* Ws, float* Y, int64_t T, int64_t N,
          int64_t K, int64_t ldy, float beta, void* stream, int tile = -1);

int tile_count();
const char* tile_name(int tile);
int default_tile(int64_t T, int64_t N);

}  // namespace strata::prefill::w8a8
