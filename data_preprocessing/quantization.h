#ifndef QUANTIZATION_H
#define QUANTIZATION_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

static inline float clampf(float v, float lo, float hi) {
    return (v < lo) ? lo : (v > hi) ? hi : v;
}

void quantize_q15(const float* x, int n, int16_t* y, int* clip_count);
void dequantize_q15(const int16_t* x, int n, float* y);


void snr_and_error(const float* ref, const float* test, int n,
                   float* snr_db, float* max_abs_err, float* rms_err);

#ifdef __cplusplus
}
#endif

#endif // QUANTIZATION_H
