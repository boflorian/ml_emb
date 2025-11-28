#include <cmath> 
#include<iostream>
#include <cstdlib> 
#include <iostream>
#include <cstring>
#include <stdio.h>

#include "pico/stdlib.h"
#include "pico/multicore.h"
#include "pico/sync.h"
#include "hardware/watchdog.h"
#include "hardware/sync.h"

#include "LCD_Driver.h"
#include "LCD_Touch.h"
#include "LCD_GUI.h"
#include "DEV_Config.h"

#include "model.h"
#include "model_settings.h"

// For IMU
#include "icm20948.h"

// Extern declarations for LCD
extern LCD_DIS sLCD_DIS;

using namespace std; 
#define HALT_CORE_1() while (1) { tight_loop_contents(); }

// Define recording and inference parameters
#define RECORDING_SAMPLES 300  // 3 seconds at 100 Hz
#define INFERENCE_WINDOW 64    // Model input size
#define IMU_BYTES_PER_SAMPLE 6 // ax, ay, az, gx, gy, gz

INFERENCE inference;
Model ml_model;

mutex_t mutex;  // Declare a mutex
static bool lcd_ready = false;

enum State { STARTING, RECORDING, INFERING, WAITING };
State current_state = STARTING;
int state_counter = 0;

// Preprocessing functions
void lowpass_filter(float* x, int length, int window) {
    float* temp = new float[length];
    for(int i = 0; i < length; i++) {
        float sum = 0.0f;
        int count = 0;
        for(int j = i - window/2; j <= i + window/2; j++) {
            if(j >= 0 && j < length) {
                sum += x[j];
                count++;
            }
        }
        temp[i] = sum / count;
    }
    memcpy(x, temp, length * sizeof(float));
    delete[] temp;
}

void apply_lowpass_filter(float* buffer, int window_size, int window, int features) {
    for(int axis = 0; axis < features; axis++) {
        float* axis_data = new float[window_size];
        for(int t = 0; t < window_size; t++) {
            axis_data[t] = buffer[t * features + axis];
        }
        lowpass_filter(axis_data, window_size, window);
        for(int t = 0; t < window_size; t++) {
            buffer[t * features + axis] = axis_data[t];
        }
        delete[] axis_data;
    }
}

void normalize_clip(float* buffer, int window_size, int features) {
    // Clip to [-80, 80]
    for(int i = 0; i < window_size * features; i++) {
        if(buffer[i] < -80.0f) buffer[i] = -80.0f;
        if(buffer[i] > 80.0f) buffer[i] = 80.0f;
    }
    // Z-score normalization per axis
    for(int axis = 0; axis < features; axis++) {
        float sum = 0.0f;
        for(int t = 0; t < window_size; t++) {
            sum += buffer[t * features + axis];
        }
        float mean = sum / window_size;
        float sum_sq = 0.0f;
        for(int t = 0; t < window_size; t++) {
            float val = buffer[t * features + axis] - mean;
            sum_sq += val * val;
        }
        float std = sqrt(sum_sq / window_size) + 1e-6f;
        for(int t = 0; t < window_size; t++) {
            buffer[t * features + axis] = (buffer[t * features + axis] - mean) / std;
        }
    }
}

// Simple fatal handler that blinks LED and shows an error message forever
static void fatal_error(const char* msg) {
    printf("FATAL: %s\n", msg);
    fflush(stdout);
#ifdef PICO_DEFAULT_LED_PIN
    gpio_init(PICO_DEFAULT_LED_PIN);
    gpio_set_dir(PICO_DEFAULT_LED_PIN, GPIO_OUT);
#endif
    while (1) {
#ifdef PICO_DEFAULT_LED_PIN
        static bool led_on = false;
        led_on = !led_on;
        gpio_put(PICO_DEFAULT_LED_PIN, led_on ? 1 : 0);
#endif
        if (lcd_ready) {
            GUI_DisString_EN(10, 10, "ERROR:", &Font16, WHITE, RED);
            GUI_DisString_EN(10, 30, msg, &Font16, WHITE, BLACK);
        }
        sleep_ms(250);
    }
}


int count_digits(int number) {
    if (number == 0) {
        return 1; // Special case for 0, which has 1 digit
    }

    number = abs(number); // Handle negative numbers
    return std::floor(std::log10(number)) + 1;
}


// Add a countdown before the recording starts
void countdown(int seconds) {
    for (int i = seconds; i > 0; --i) {
        printf("Starting in %d...\n", i);
        GUI_DisString_EN(10, 60, "Starting in...", &Font16, WHITE, BLACK);
        char countdown_str[16];
        snprintf(countdown_str, sizeof(countdown_str), "%d", i);
        GUI_DisString_EN(10, 80, countdown_str, &Font24, WHITE, BLACK);
        sleep_ms(1000);
    }
    GUI_DisString_EN(10, 60, "                ", &Font16, WHITE, BLACK); 
    GUI_DisString_EN(10, 80, "                ", &Font24, WHITE, BLACK);
}

// run core0 loop that displays UI and handle user interaction
void core1_entry() {
    uint16_t cnt=0;
    while(true) {
      
      for(cnt=1000;cnt>2;cnt--)
      {
        LCD_SetBackLight(1000);
        
        // Removed TP_DrawBoard() as we're using IMU for input, not touch
        // TP_DrawBoard();
      }
    }
}

int main(void) 
{
    System_Init();
    setvbuf(stdout, NULL, _IONBF, 0); // unbuffered stdout for USB/UART
    printf("System initialized\n");
    mutex_init(&mutex);  // Initialize the mutex

    sleep_ms(5000);
    printf("Sleep done\n");

    // Initialize IMU
    IMU_EN_SENSOR_TYPE imu_type;
    imuInit(&imu_type);
    if (imu_type != IMU_EN_SENSOR_TYPE_ICM20948) {
        fatal_error("IMU not detected");
    }
    printf("IMU initialized\n");

    // initialize LCD display
    LCD_SCAN_DIR  lcd_scan_dir = SCAN_DIR_DFT;
    LCD_Init(lcd_scan_dir,1000);
    TP_Init(lcd_scan_dir);
    TP_GetAdFac();
    printf("LCD initialized\n");
    reset_inference(&inference);
	init_gui();
    lcd_ready = true;

    // Keep the white background from init_gui()


    // run core1 loop that handles user interface
    multicore_launch_core1(core1_entry);
    printf("Core1 launched\n");

        // initialize ML model
    if (!ml_model.setup()) {
        fatal_error("Model init failed");
    }
    printf("Model initialized\n");
    
    float input_scale = ml_model.input_scale();
    int input_zero_point = ml_model.input_zero_point();
    printf("Input quantization: scale=%f, zero_point=%d\n", input_scale, input_zero_point);
    
    uint8_t* test_image_input = ml_model.input_data();
    if (test_image_input == nullptr) {
        fatal_error("Cannot get model input");
    }

    int byte_size = ml_model.byte_size();
    if (!byte_size) {
        fatal_error("Byte size not found");
    }

    // Buffer for IMU data: RECORDING_SAMPLES samples * 6 features
    const int imu_buffer_bytes = RECORDING_SAMPLES * IMU_BYTES_PER_SAMPLE;
    const int imu_float_elements = imu_buffer_bytes;
    uint8_t imu_buffer[RECORDING_SAMPLES * IMU_BYTES_PER_SAMPLE];  // For inference input
    float imu_buffer_float[imu_float_elements];
    float magnitudes[RECORDING_SAMPLES];  // Store motion magnitudes
    int buffer_index = 0;
    int loop_counter = 0;
    bool heartbeat_on = false;
    uint16_t backlight_level = 1000;

#ifdef PICO_DEFAULT_LED_PIN
    gpio_init(PICO_DEFAULT_LED_PIN);
    gpio_set_dir(PICO_DEFAULT_LED_PIN, GPIO_OUT);
    gpio_put(PICO_DEFAULT_LED_PIN, 0);
#endif

    while (true) {
        // printf("Entering main loop iteration, state: %d\n", current_state);
        
        switch (current_state) {
            case STARTING:
                countdown(3); // 3-second countdown
                GUI_DisString_EN(10, 80, "Recording...", &Font16, WHITE, BLACK);
                buffer_index = 0;
                current_state = RECORDING;
                state_counter = 0;
                break;
                
            case RECORDING: {
                // Read IMU data
                IMU_ST_SENSOR_DATA gyro, accel;
                imuDataAccGyrGet(&gyro, &accel);

                // Print or process the data
                //printf("Accel: X=%d, Y=%d, Z=%d | Gyro: X=%d, Y=%d, Z=%d\n",
                //       accel.s16X, accel.s16Y, accel.s16Z,
                //       gyro.s16X, gyro.s16Y, gyro.s16Z);

                // Collect data in buffer
                const int write_offset = buffer_index * IMU_BYTES_PER_SAMPLE;
                imu_buffer_float[write_offset + 0] = (float)accel.s16X;
                imu_buffer_float[write_offset + 1] = (float)accel.s16Y;
                imu_buffer_float[write_offset + 2] = (float)accel.s16Z;
                imu_buffer_float[write_offset + 3] = (float)gyro.s16X;
                imu_buffer_float[write_offset + 4] = (float)gyro.s16Y;
                imu_buffer_float[write_offset + 5] = (float)gyro.s16Z;

                // Compute motion magnitude (using acceleration)
                float ax = (float)accel.s16X / 32768.0f;  // Normalize to -1 to 1
                float ay = (float)accel.s16Y / 32768.0f;
                float az = (float)accel.s16Z / 32768.0f;
                magnitudes[buffer_index] = sqrt(ax*ax + ay*ay + az*az);

                buffer_index++;
                state_counter++;
                
                if (state_counter >= RECORDING_SAMPLES) {
                    current_state = INFERING;
                    state_counter = 0;
                }
                break;
            }
                
            case INFERING: {
                // Find the sample with maximum motion magnitude
                int max_magnitude_index = 0;
                float max_magnitude = magnitudes[0];
                for(int i = 1; i < RECORDING_SAMPLES; i++) {
                    if(magnitudes[i] > max_magnitude) {
                        max_magnitude = magnitudes[i];
                        max_magnitude_index = i;
                    }
                }
                printf("Max magnitude at sample %d: %.2f\n", max_magnitude_index, max_magnitude);
                
                // Extract INFERENCE_WINDOW samples centered on max_magnitude_index
                const int half_window = INFERENCE_WINDOW / 2;
                int start_sample = max_magnitude_index - half_window;
                if(start_sample < 0) start_sample = 0;
                int end_sample = start_sample + INFERENCE_WINDOW - 1;
                if(end_sample >= RECORDING_SAMPLES) {
                    end_sample = RECORDING_SAMPLES - 1;
                    start_sample = end_sample - INFERENCE_WINDOW + 1;
                    if(start_sample < 0) start_sample = 0;
                }
                
                const int inference_bytes = INFERENCE_WINDOW * IMU_BYTES_PER_SAMPLE;
                float inference_buffer_float[inference_bytes];
                uint8_t inference_buffer[inference_bytes];
                
                // Copy the selected window
                const int start_offset = start_sample * IMU_BYTES_PER_SAMPLE;
                memcpy(inference_buffer_float, imu_buffer_float + start_offset, inference_bytes * sizeof(float));
                
                // Apply preprocessing to the inference window
                apply_lowpass_filter(inference_buffer_float, INFERENCE_WINDOW, 7, IMU_BYTES_PER_SAMPLE);
                normalize_clip(inference_buffer_float, INFERENCE_WINDOW, IMU_BYTES_PER_SAMPLE);
                
                // Debug: print some preprocessed values
                printf("Preprocessed samples: ");
                for(int i = 0; i < 10 && i < inference_bytes; i++) {
                    printf("%.2f ", inference_buffer_float[i]);
                }
                printf("\n");
                
                // Quantize to uint8 using 
                
                for(int i = 0; i < inference_bytes; i++) {
                    float val = inference_buffer_float[i];
                    int quantized = round(val / input_scale) + input_zero_point;
                    inference_buffer[i] = (uint8_t)std::max(0, std::min(255, quantized));
                }
                
                // Copy to model input
                const int bytes_to_copy = byte_size < inference_bytes ? byte_size : inference_bytes;
                memcpy(test_image_input, inference_buffer, bytes_to_copy);
                if (bytes_to_copy < byte_size) {
                    // Zero any remaining input bytes so we don't leave stale data
                    memset(test_image_input + bytes_to_copy, 0, byte_size - bytes_to_copy);
                }

                // Run inference
                int result = ml_model.predict();
                printf("Inference result: %d\n", result);
                
                // If classified as negative, take the next most likely gesture
                //if (result == 0) {
                //float* outputs = ml_model.output_data();
                //    if (outputs != nullptr) {
                //        float max_prob = outputs[1];
                //        int best_non_neg = 1;
                //        for (int i = 2; i < kCategoryCount; ++i) {
                //            if (outputs[i] > max_prob) {
                //                max_prob = outputs[i];
                //                best_non_neg = i;
                //            }
                //        }
                //        result = best_non_neg;
                //        printf("Reclassified from negative to: %d\n", result);
                //    }
                //}
                
                if (result == -1) {
                    printf("Failed to run inference\n");
                    GUI_DisString_EN(10, 80, "Inference failed", &Font16, WHITE, BLACK);
                } else {
                    const char* label = (result >= 0 && result < kCategoryCount) ? kCategoryLabels[result] : "unknown";
                    printf("Predicted Gesture: %d (%s)\n", result, label);
                    // Display gesture on LCD
                    char str[32];
                    snprintf(str, sizeof(str), "Gesture: %s", label);
                    GUI_DisString_EN(10, 100, str, &Font24, WHITE, BLACK);
                    // Clear inference indicator
                    GUI_DisString_EN(10, 80, "           ", &Font16, WHITE, BLACK);
                }

                current_state = WAITING;
                state_counter = 0;
                break;
            }
                
            case WAITING:
                state_counter++;
                if (state_counter >= 200) {  // 2 seconds at 100Hz
                    current_state = STARTING;
                    state_counter = 0;
                }
                break;
        }

        // Heartbeat: toggle LED and a small on-screen marker (loop alive ???)
        if ((loop_counter++ % 50) == 0) {
            heartbeat_on = !heartbeat_on;
#ifdef PICO_DEFAULT_LED_PIN
            gpio_put(PICO_DEFAULT_LED_PIN, heartbeat_on ? 1 : 0);
#endif
            // Pulse the LCD backlight as a visual heartbeat even if text is not visible
            backlight_level = heartbeat_on ? 1000 : 100;
            LCD_SetBackLight(backlight_level);
            // Leave a small marker on-screen if the LCD is on
            GUI_DisString_EN(10, 200, heartbeat_on ? "*" : " ", &Font16, WHITE, BLACK);
        }

        sleep_ms(10); 
    }
    return 0;
}
