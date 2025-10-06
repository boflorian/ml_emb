#include "statistic.h"
#include <math.h>

// arithmetic mean
float mean_f32(const float* x, int n) {
    double acc = 0.0;
    for (int i = 0; i < n; i++) acc += x[i];
    return (float)(acc / (double)n);
}

// sample variance (denominator n-1)
float variance_f32(const float* x, int n, float mean) {
    if (n <= 1) return 0.0f;
    double acc = 0.0;
    for (int i = 0; i < n; i++) {
        double d = (double)x[i] - (double)mean;
        acc += d * d;
    }
    return (float)(acc / (double)(n - 1));
}

// standard deviation from variance
float stddev_f32(float variance) {
    return sqrtf(variance);
}

// min & max
void min_max_f32(const float* x, int n, float* mn, float* mx) {
    float a = x[0], b = x[0];
    for (int i = 1; i < n; i++) {
        if (x[i] < a) a = x[i];
        if (x[i] > b) b = x[i];
    }
    *mn = a; *mx = b;
}

// median (sorts a local copy; insertion sort — fine for small N)
float median_f32(const float* x, int n) {
    float tmp[n];
    for (int i = 0; i < n; i++) tmp[i] = x[i];

    // insertion sort (ascending)
    for (int i = 1; i < n; i++) {
        float key = tmp[i];
        int j = i - 1;
        while (j >= 0 && tmp[j] > key) {
            tmp[j + 1] = tmp[j];
            j--;
        }
        tmp[j + 1] = key;
    }

    if (n & 1) return tmp[n / 2];
    return 0.5f * (tmp[n / 2 - 1] + tmp[n / 2]);
}

// mode (for discrete/repeated values). If all counts are 1, no mode.
// eps lets you treat near-equal floats as equal (use 0 for exact).
float mode_f32(const float* x, int n, int* count_out, float eps) {
    int best_count = 0;
    float best_val = NAN;

    for (int i = 0; i < n; i++) {
        int cnt = 1;
        for (int j = i + 1; j < n; j++) {
            if (fabsf(x[j] - x[i]) <= eps) cnt++;
        }
        if (cnt > best_count) {
            best_count = cnt;
            best_val = x[i];
        }
    }

    if (count_out) *count_out = best_count;
    return best_val; // if best_count==1, there is no mode
}
