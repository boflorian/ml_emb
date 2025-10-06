#ifndef IMU_FFT_H
#define IMU_FFT_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stddef.h>

/* simple complex type used by the FFT */
typedef struct { float re; float im; } c32;

/* Apply an in-place Hamming window to an array of length n */
void hamming_window(float* x, int n);

/* In-place radix-2 FFT
 * x: complex buffer length n (power of two)
 * dir: +1 = FFT, -1 = IFFT
 * returns 0 on success, non-zero on error (e.g. n not power of two)
 */
int fft_radix2(c32* x, int n, int dir);

/* Compute magnitude array (length n) from complex spectrum X */
void fft_mag(const c32* X, int n, float* mag);

/* Peak picking helper: find top-K peaks in single-sided magnitude array
 * - mag: magnitude array
 * - n_half: number of bins to consider (typically N/2+1 for single-sided)
 * - k_exclude_dc: starting index to consider (use 1 to skip DC)
 * - K: number of peaks to find
 * - out_idx/out_val: output arrays of length >= K
 * - out_count: actual number of peaks found
 */
void top_k_peaks(const float* mag, int n_half, int k_exclude_dc, int K,
                 int* out_idx, float* out_val, int* out_count);

#ifdef __cplusplus
}
#endif

#endif /* IMU_FFT_H */
