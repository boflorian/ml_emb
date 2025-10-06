#ifndef STATISTIC_H
#define STATISTIC_H

#ifdef __cplusplus
extern "C" {
#endif

// arithmetic mean
float mean_f32(const float* x, int n);

// sample variance (denominator n-1)
float variance_f32(const float* x, int n, float mean);

// standard deviation from variance
float stddev_f32(float variance);

// min & max
void min_max_f32(const float* x, int n, float* mn, float* mx);

// median (sorts a local copy; insertion sort)
float median_f32(const float* x, int n);

// mode (for discrete/repeated values)
// eps lets you treat near-equal floats as equal (use 0 for exact)
float mode_f32(const float* x, int n, int* count_out, float eps);

#ifdef __cplusplus
}
#endif

#endif // STATISTIC_H
