#include <stdio.h>
#include <math.h>
#include <stdint.h>
#include <stdbool.h>
#include "pico/stdlib.h"
#include "fft.h"

int main(void){

    stdio_init_all();
    sleep_ms(1500); 

    // ---- Settings ----
    const float fs = 2200.0f;   // IMU sample rate (Hz)
    enum { N = 256 };           // power-of-two FFT size
    const float df = fs / (float)N;

    if(!is_power_of_two(N)){
        printf("N must be power of two\n");
        while(true) tight_loop_contents();
    }

    // ---- Build a synthetic signal: 120 Hz and 440 Hz
    float x[N];
    for(int n=0; n<N; n++){
        float t = (float)n / fs;
        x[n] = 0.7f*sinf(2.0f*(float)M_PI*120.0f*t)
             + 0.3f*sinf(2.0f*(float)M_PI*440.0f*t);
    }

    // ---- Window and pack into complex buffer
    hamming_window(x, N);
    c32 X[N];
    for(int i=0;i<N;i++){ X[i].re = x[i]; X[i].im = 0.0f; }

    // ---- FFT
    if(fft_radix2(X, N, +1) != 0){
        printf("FFT error\n");
        while(true) tight_loop_contents();
    }

    // ---- Magnitude spectrum (single-sided bins 0..N/2)
    float mag[N];
    fft_mag(X, N, mag);

    const float scale = (2.0f / (float)N) / 0.54f;
    for(int k=0;k<=N/2;k++) mag[k] *= scale;

    // ---- Report
    printf("\n=== IMU FFT Demo ===\n");
    printf("fs=%.1f Hz, N=%d, resolution df=%.3f Hz\n", fs, N, df);
    printf("Looking for top 5 peaks (excluding DC):\n");

    int idx[5]; float val[5]; int found=0;
    top_k_peaks(mag, N/2+1, /*exclude up to index*/1, /*K*/5, idx, val, &found);

    for(int i=0;i<found;i++){
        float fk = df * (float)idx[i];
        printf("  Peak %d: bin=%d  freq=%.2f Hz  amplitude≈%.4f\n",
               i+1, idx[i], fk, val[i]);
    }

    printf("\n");
    printf("TO COPY ONTO A TEXT FILE FOR COMPARSION WITH PC RUN FFT\n");

    // Print the raw values onto the serial in one line
    for (int n = 0; n < N; n++) { if (n) printf(","); printf("%.6f", (double)x[n]); } printf("\n");
    printf("\n");
    printf("TO COPY ONTO A TEXT FILE FOR COMPARSION WITH PC RUN FFT\n");
    
    // Print to the serial the output of the fft to compare with the host run python implmentation of the fft filter
    printf("bin,freq,amp\n");
    for(int i=0;i<found;i++)
    {
        float fk = df * (float)idx[i];
        printf("%d,%.6f,%.6f\n", idx[i], fk, val[i]);
    }

    while(true) tight_loop_contents();
    return 0;
}
