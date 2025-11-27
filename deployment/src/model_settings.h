#ifndef TFLITE_INFERENCE_TEST_MODEL_SETTINGS_H_
#define TFLITE_INFERENCE_TEST_MODEL_SETTINGS_H_

constexpr int kNumCols = 96;
constexpr int kNumRows = 96;
constexpr int kNumChannels = 1;

constexpr int kMaxImageSize = kNumCols * kNumRows * kNumChannels;

// Gesture model outputs 4 classes: negative, ring, slope, wing
constexpr int kCategoryCount = 4;
extern const char* kCategoryLabels[kCategoryCount];

constexpr int image_col_size = 28;
constexpr int image_row_size = 28;

constexpr int arena_size = 200 * 1024; // Increased for CNN model
#endif  // TFLITE_INFERENCE_TEST_MODEL_SETTINGS_H_
