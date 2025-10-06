#include <stdio.h>
#include <math.h>
#include <stdint.h>
#include <stdbool.h>
#include "pico/stdlib.h"

// ---------- Simple complex type ----------
typedef struct { float re, im; } c32;

// ---------- Small utilities ----------
static inline float fsqrtf(float x){ return sqrtf(x); }
static inline float fclampf(float v, float lo, float hi){ return (v<lo)?lo:((v>hi)?hi:v); }

/* static removed from most functions here to make them available 
outside the current file */
// ---------- Hamming window (in-place) ----------
void hamming_window(float* x, int n){
    for(int i=0;i<n;i++){
        x[i] *= 0.54f - 0.46f * cosf(2.0f*(float)M_PI*(float)i/(float)(n-1));
    }
}

// ---------- Bit helpers ----------
int is_power_of_two(int n){ return (n>0) && ((n & (n-1))==0); }
unsigned reverse_bits(unsigned v, int nbits){
    unsigned r = 0u;
    for(int i=0;i<nbits;i++){ r = (r<<1) | (v & 1u); v >>= 1u; }
    return r;
}

// ---------- In-place radix-2 Cooley–Tukey FFT ----------
// dir = +1 for FFT, -1 for IFFT
int fft_radix2(c32* x, int n, int dir){
    if(!is_power_of_two(n)) return -1;
    int logn = 0; while((1<<logn) < n) logn++;

    // Bit-reversal permutation
    for(unsigned i=0;i<(unsigned)n;i++){
        unsigned j = reverse_bits(i, logn);
        if(j>i){ c32 t = x[i]; x[i]=x[j]; x[j]=t; }
    }

    const float sgn = (dir >= 0) ? -1.0f : 1.0f;
    for(int s=1; s<=logn; s++){
        int m = 1<<s;
        int m2 = m>>1;
        float theta = sgn * (float)M_PI / (float)m2;
        float wpr = -2.0f * sinf(0.5f*theta) * sinf(0.5f*theta);
        float wpi = sinf(theta);
        for(int k=0; k<n; k+=m){
            float wr = 1.0f, wi = 0.0f;
            for(int j=0; j<m2; j++){
                int t = k + j + m2;
                int u = k + j;
                float tr = wr*x[t].re - wi*x[t].im;
                float ti = wr*x[t].im + wi*x[t].re;
                float ur = x[u].re, ui = x[u].im;
                x[t].re = ur - tr; x[t].im = ui - ti;
                x[u].re = ur + tr; x[u].im = ui + ti;
                // twiddle update (CORDIC-free recurrence)
                float tmp = wr;
                wr = wr + (wr*wpr - wi*wpi);
                wi = wi + (wi*wpr + tmp*wpi);
            }
        }
    }

    if(dir < 0){
        float inv = 1.0f/(float)n;
        for(int i=0;i<n;i++){ x[i].re *= inv; x[i].im *= inv; }
    }
    return 0;
}

// ---------- Magnitude spectrum ----------
void fft_mag(const c32* X, int n, float* mag){
    for(int i=0;i<n;i++){
        mag[i] = fsqrtf(X[i].re*X[i].re + X[i].im*X[i].im);
    }
}

// ---------- Peak picking (top-K, single-sided) ----------
void top_k_peaks(const float* mag, int n_half, int k_exclude_dc, int K,
                        int* out_idx, float* out_val, int* out_count){
    // simple selection without sorting the full array
    int count = 0;
    for(int k=0; k<K; k++){
        int best_i = -1; float best_v = -1.0f;
        for(int i=k_exclude_dc; i<n_half; i++){
            // skip already taken
            bool taken = false;
            for(int j=0;j<count;j++) if(out_idx[j]==i){ taken = true; break; }
            if(taken) continue;
            if(mag[i] > best_v){ best_v = mag[i]; best_i = i; }
        }
        if(best_i >= 0){
            out_idx[count] = best_i;
            out_val[count] = best_v;
            count++;
        }
    }
    *out_count = count;
/* Demo/test main omitted here; see fft_demo.c in the same directory for a
 * standalone fft test program */
