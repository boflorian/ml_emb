#include "quantization.h"
#include <math.h>
#include <stdint.h>

// This function is declared static inline in the header and is available there.
// No need to have a duplicate definition here.

void quantize_q15(const float* x, int n, int16_t* y, int* clip_count) {
    int clips = 0;
    for (int i = 0; i < n; i++) {
        // Clamp input to valid Q15 range [-1.0, 0.999969]
        float s = clampf(x[i], -1.0f, 0.999969f);
        if (s != x[i]) clips++;
        
        // Use roundf() for proper rounding: round(x * 32768)
        // Q15 format: multiply by 2^15 = 32768
        float scaled = s * 32768.0f;
        int32_t q = (int32_t)roundf(scaled);
        
        // Clip to int16_t range [-32768, 32767]
        if (q >  32767) q =  32767;
        if (q < -32768) q = -32768;
        
        y[i] = (int16_t)q;
    }
    if (clip_count) *clip_count = clips;
}

void dequantize_q15(const int16_t* x, int n, float* y) {
    for (int i = 0; i < n; i++) y[i] = (float)x[i] / 32768.0f;
}
