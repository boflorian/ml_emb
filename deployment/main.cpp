#include <cmath> 
#include<iostream>
#include <cstdlib> 
#include <iostream>
#include <string>
#include <cstring>
#include <stdio.h>

#include "pico/stdlib.h"
#include "pico/multicore.h"
#include "pico/sync.h"
#include "pico/cyw43_arch.h"
#include "hardware/watchdog.h"
#include "hardware/sync.h"

#include "LCD_Driver.h"
#include "LCD_Touch.h"
#include "LCD_GUI.h"
#include "DEV_Config.h"

#include "model.h"
#include "model_settings.h"
#include "request.hpp"

// For IMU
#include "icm20948.h"

// Extern declarations for LCD
extern LCD_DIS sLCD_DIS;

using namespace std; 
#define HALT_CORE_1() while (1) { tight_loop_contents(); }

// Define recording and inference parameters
#define RECORDING_SAMPLES 300   // 3 seconds at 100 Hz
#define INFERENCE_WINDOW 286     // Model input size per gesture window
#define IMU_FEATURES 3          // ax, ay, az (match lab_0_1 data flow)
#define ACCEL_SCALE (1.0f / 16384.0f)

// App2 configuration
#include "wifi_config.h"
static const char SERVER_HOST[] = WIFI_SERVER_HOST;
static const uint16_t SERVER_PORT = WIFI_SERVER_PORT;
static const char SERVER_PATH[] = WIFI_SERVER_PATH;

// Trigger configuration (simple version)
static const int TRIGGER_CLASS = 1;
static const uint32_t COOLDOWN_MS = 0;
static const uint32_t NET_TIMEOUT_MS = 8000;
static const uint32_t SERVER_SEND_INTERVAL_MS = 1500;
static const int SERVER_WINDOW_SAMPLES = 8;



// maybe remove later 
#define MOTION_THRESHOLD_G 0.12f   // minimum acceleration magnitude to treat as motion
#define DEBUG_SAMPLE_PRINTS 6      // number of samples to dump for debugging

INFERENCE inference;
Model ml_model;

mutex_t mutex;  
static bool lcd_ready = false;

enum State { STARTING, RECORDING, INFERING, WAITING };
State current_state = STARTING;
int state_counter = 0;

enum AppMode { MODE_GESTURE, MODE_SERVER };
static AppMode mode = MODE_GESTURE;

enum NetState { NET_IDLE, NET_WIFI, NET_SEND, NET_DONE, NET_ERR };
static NetState net_state = NET_IDLE;

static uint32_t last_trigger_time_ms = 0;
static int last_trigger_class = -1;
static const char* last_trigger_label = "unknown";

static bool wifi_ready = false;
static bool armed = false;
static bool send_arm_event = false;

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
    // Skip normalization for small window sizes
    if (window_size <= 1) {
        return;
    }

    // Clip to [-80, 80]
    for (int i = 0; i < window_size * features; i++) {
        if (buffer[i] < -80.0f) buffer[i] = -80.0f;
        if (buffer[i] > 80.0f) buffer[i] = 80.0f;
    }

    // Z-score normalization per axis
    for (int axis = 0; axis < features; axis++) {
        float sum = 0.0f;
        for (int t = 0; t < window_size; t++) {
            sum += buffer[t * features + axis];
        }
        float mean = sum / window_size;

        float sum_sq = 0.0f;
        for (int t = 0; t < window_size; t++) {
            float val = buffer[t * features + axis] - mean;
            sum_sq += val * val;
        }
        float std = sqrt(sum_sq / window_size);
        if (std < 1e-6f) {
            continue;
        }

        for (int t = 0; t < window_size; t++) {
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

static inline uint32_t now_ms(void) {
    return to_ms_since_boot(get_absolute_time());
}

static bool ensure_wifi_connected(void) {
    if (wifi_ready) {
        return true;
    }
    if (cyw43_arch_init()) {
        printf("[APP2] Wi-Fi init failed\n");
        return false;
    }
    cyw43_arch_enable_sta_mode();
    printf("[APP2] connecting to Wi-Fi...\n");
    int rc = cyw43_arch_wifi_connect_timeout_ms(
        WIFI_SSID, WIFI_PASSWORD, CYW43_AUTH_WPA2_AES_PSK, 30000);
    if (rc != 0) {
        printf("[APP2] Wi-Fi connect failed: %d\n", rc);
        return false;
    }
    printf("[APP2] Wi-Fi connected\n");
    wifi_ready = true;
    return true;
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
    mutex_init(&mutex); 

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

    // Pre-connect Wi-Fi so App2 can send immediately.
    if (!ensure_wifi_connected()) {
        printf("[APP2] Wi-Fi pre-connect failed (will retry on trigger)\n");
    }


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
    // Sanity check: model input size vs our prepared window (features x window)
    const int model_input_bytes = ml_model.byte_size();
    const int prepared_input_bytes = INFERENCE_WINDOW * IMU_FEATURES; // uint8 after quantization
    if (model_input_bytes != prepared_input_bytes) {
         printf("Warning: model expects %d bytes but pipeline prepares %d bytes (INFERENCE_WINDOW=%d, FEATURES=%d)\n",
             model_input_bytes, prepared_input_bytes, INFERENCE_WINDOW, IMU_FEATURES);
         // We still proceed and zero-fill the remaining bytes below, but adjust either the
         // model input shape or INFERENCE_WINDOW/IMU_FEATURES so they match.
    }
    
    uint8_t* test_image_input = ml_model.input_data();
    if (test_image_input == nullptr) {
        fatal_error("Cannot get model input");
    }

    int byte_size = model_input_bytes;
    if (!byte_size) {
        fatal_error("Byte size not found");
    }

    // Buffer for IMU data: RECORDING_SAMPLES samples * 3 accel features
    const int imu_float_elements = RECORDING_SAMPLES * IMU_FEATURES;
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

    // Hardcoding the model size instead of including the header
    constexpr size_t arena_size = 81920; // Tensor size is 67876 -> safety margin 80 * 1024


    while (true) {
        // printf("Entering main loop iteration, state: %d\n", current_state);
        if (mode == MODE_GESTURE) {
        switch (current_state) {
            case STARTING:
                countdown(3); // 3-second countdown
                buffer_index = 0;
                current_state = RECORDING;
                state_counter = 0;
                break;
                
            case RECORDING: {
                // Print recording indicator only once at the start
                if (buffer_index == 0) {
                    printf("Recording...\n");
                }
                
                // Read IMU data
                IMU_ST_SENSOR_DATA gyro, accel;
                imuDataAccGyrGet(&gyro, &accel);

                // Collect data in buffer - convert to g-units to match training data
                // Training data uses ACCEL_SCALE = 1/16384 (same as lab_0_1.cpp)
                const int write_offset = buffer_index * IMU_FEATURES;
                imu_buffer_float[write_offset + 0] = (float)accel.s16X * ACCEL_SCALE;  // Convert to g-units
                imu_buffer_float[write_offset + 1] = (float)accel.s16Y * ACCEL_SCALE;
                imu_buffer_float[write_offset + 2] = (float)accel.s16Z * ACCEL_SCALE;

                // Compute motion magnitude (using acceleration in g-units)
                float ax = imu_buffer_float[write_offset + 0];
                float ay = imu_buffer_float[write_offset + 1];
                float az = imu_buffer_float[write_offset + 2];
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

                // Activity recognition: clip leading idle samples
                int active_start = 0;
                while (active_start < RECORDING_SAMPLES && magnitudes[active_start] < MOTION_THRESHOLD_G) {
                    active_start++;
                }
                if (active_start >= RECORDING_SAMPLES) {
                    active_start = 0; // all quiet, fall back to start
                }

                //] Max magnitude idx=%d val=%.2f | active_start=%d (thr=%.2fg)\n",
                  //     max_magnitude_index, max_magnitude, active_start, MOTION_THRESHOLD_G);
                
                // Extract INFERENCE_WINDOW samples prioritizing active region
                const int half_window = INFERENCE_WINDOW / 2;
                int start_sample = max_magnitude_index - half_window;
                if(start_sample < 0) start_sample = 0;
                if(start_sample < active_start) start_sample = active_start;
                int end_sample = start_sample + INFERENCE_WINDOW - 1;
                if(end_sample >= RECORDING_SAMPLES) {
                    end_sample = RECORDING_SAMPLES - 1;
                    start_sample = end_sample - INFERENCE_WINDOW + 1;
                    if(start_sample < 0) start_sample = 0;
                }
                //printf("[DBG] Window start=%d end=%d (size=%d)\n", start_sample, end_sample, INFERENCE_WINDOW);
                
                const int inference_elements = INFERENCE_WINDOW * IMU_FEATURES;
                float inference_buffer_float[inference_elements];
                
                // Copy the selected window
                const int start_offset = start_sample * IMU_FEATURES;
                memcpy(inference_buffer_float, imu_buffer_float + start_offset, inference_elements * sizeof(float));

                // Apply preprocessing to match training pipeline
                apply_lowpass_filter(inference_buffer_float, INFERENCE_WINDOW, 5, IMU_FEATURES); // window=5 for lowpass
                
                // Clip to [-80, 80] first
                for (int i = 0; i < inference_elements; i++) {
                    if (inference_buffer_float[i] < -80.0f) inference_buffer_float[i] = -80.0f;
                    if (inference_buffer_float[i] > 80.0f) inference_buffer_float[i] = 80.0f;
                }
                
                // Z-score normalization per axis (matches training normalize_clip)
                for (int axis = 0; axis < IMU_FEATURES; axis++) {
                    float sum = 0.0f;
                    for (int t = 0; t < INFERENCE_WINDOW; t++) {
                        sum += inference_buffer_float[t * IMU_FEATURES + axis];
                    }
                    float mean = sum / INFERENCE_WINDOW;
                    
                    float sum_sq = 0.0f;
                    for (int t = 0; t < INFERENCE_WINDOW; t++) {
                        float val = inference_buffer_float[t * IMU_FEATURES + axis] - mean;
                        sum_sq += val * val;
                    }
                    float std = sqrt(sum_sq / INFERENCE_WINDOW) + 1e-6f;
                    
                    for (int t = 0; t < INFERENCE_WINDOW; t++) {
                        inference_buffer_float[t * IMU_FEATURES + axis] = 
                            (inference_buffer_float[t * IMU_FEATURES + axis] - mean) / std;
                    }
                }
                
                // NOTE: Scaling disabled - testing with direct Z-score normalized values
                // If this works, the model expects normalized input directly
                // If not, we need to re-quantize the model properly

                // Debug: print some preprocessed values (now Z-score normalized, ~[-3, 3])
                //printf("[DBG] Window floats: ");
                //for(int i = 0; i < 10 && i < inference_elements; i++) {
                //    printf("%.2f ", inference_buffer_float[i]);
                //}
                printf("\n");
                
                // Quantize to int8 (signed) - model uses INT8 quantization with zero_point around -1
                int8_t inference_buffer[inference_elements];
                for(int i = 0; i < inference_elements; i++) {
                    float val = inference_buffer_float[i];
                    int quantized = round(val / input_scale) + input_zero_point;
                    inference_buffer[i] = (int8_t)std::max(-128, std::min(127, quantized));
                }

                //printf("[DBG] Window quantized: ");
                //for(int i = 0; i < 10 && i < inference_elements; i++) {
                //    printf("%d ", inference_buffer[i]);
                //}
                //printf("\n");
                
                // Copy to model input (cast to uint8_t* for memcpy, but data is int8)
                const int bytes_to_copy = byte_size < inference_elements ? byte_size : inference_elements;
                memcpy(test_image_input, inference_buffer, bytes_to_copy);
                if (bytes_to_copy < byte_size) {
                    // Zero any remaining input bytes so we don't leave stale data
                    memset(test_image_input + bytes_to_copy, 0, byte_size - bytes_to_copy);
                }

                // Run inference
                int result = ml_model.predict();
                // printf("Inference result: %d\n", result);
                
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
                    // printf("Predicted Gesture: %d (%s)\n", result, label);
                    printf("[APP1] class=%d conf=%.2f\n", result, 0.0f);
                    // Display gesture on LCD
                    char str[32];
                    snprintf(str, sizeof(str), "Gesture: %s", label);
                    GUI_DisString_EN(10, 100, str, &Font24, WHITE, BLACK);
                    // Clear inference indicator
                    GUI_DisString_EN(10, 80, "           ", &Font16, WHITE, BLACK);

                    uint32_t now = now_ms();
                    bool in_cooldown = (now - last_trigger_time_ms) < COOLDOWN_MS;
                    if (!in_cooldown && result == TRIGGER_CLASS) {
                        last_trigger_time_ms = now;
                        last_trigger_class = result;
                        last_trigger_label = label;
                        printf("[TRIGGER] fired\n");
                        armed = true;
                        send_arm_event = true;
                        mode = MODE_SERVER;
                        net_state = NET_IDLE;
                    }
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
        } else {
            static float stream_buffer[SERVER_WINDOW_SAMPLES * IMU_FEATURES];
            static int stream_index = 0;
            static uint32_t last_send_ms = 0;

            if (!armed) {
                mode = MODE_GESTURE;
                current_state = STARTING;
                net_state = NET_IDLE;
            } else {
                if (net_state == NET_IDLE) {
                    printf("[APP2] streaming to server ...\n");
                    net_state = NET_WIFI;
                }

                if (net_state == NET_WIFI) {
                    if (!ensure_wifi_connected()) {
                        printf("[APP2] Wi-Fi not ready\n");
                        net_state = NET_ERR;
                    } else {
                        net_state = NET_SEND;
                    }
                }

                if (net_state == NET_SEND) {
                    IMU_ST_SENSOR_DATA gyro, accel;
                    imuDataAccGyrGet(&gyro, &accel);
                    const int write_offset = stream_index * IMU_FEATURES;
                    stream_buffer[write_offset + 0] = (float)accel.s16X * ACCEL_SCALE;
                    stream_buffer[write_offset + 1] = (float)accel.s16Y * ACCEL_SCALE;
                    stream_buffer[write_offset + 2] = (float)accel.s16Z * ACCEL_SCALE;
                    stream_index++;

                    uint32_t now = now_ms();
                    if (stream_index >= SERVER_WINDOW_SAMPLES &&
                        (now - last_send_ms) >= SERVER_SEND_INTERVAL_MS) {
                        static char json[2048];
                        const char* trig = send_arm_event ? "ring" : "activity";
                        int n = snprintf(
                            json, sizeof(json),
                            "{\"device_id\":\"pico_w\",\"t_ms\":%lu,"
                            "\"trigger_class\":\"%s\",\"trigger_conf\":1.0,"
                            "\"samples\":[",
                            (unsigned long)now, trig);
                        if (n <= 0 || n >= (int)sizeof(json)) {
                            printf("[APP2] payload build failed\n");
                            net_state = NET_ERR;
                        } else {
                            int pos = n;
                            for (int i = 0; i < SERVER_WINDOW_SAMPLES * IMU_FEATURES; i++) {
                                int written = snprintf(
                                    json + pos, sizeof(json) - pos,
                                    "%s%.4f",
                                    (i == 0 ? "" : ","),
                                    stream_buffer[i]);
                                if (written <= 0 || written >= (int)(sizeof(json) - pos)) {
                                    break;
                                }
                                pos += written;
                            }
                            int tail = snprintf(
                                json + pos, sizeof(json) - pos,
                                "],\"meta\":{\"fs\":100,\"win\":%d,\"armed\":true}}",
                                SERVER_WINDOW_SAMPLES);
                            if (tail <= 0 || tail >= (int)(sizeof(json) - pos)) {
                                printf("[APP2] payload build failed\n");
                                net_state = NET_ERR;
                            } else {
                                HttpRequest request(std::string(SERVER_PATH),
                                                    std::string(SERVER_HOST),
                                                    SERVER_PORT);
                                request.set_timeout_ms(NET_TIMEOUT_MS);
                                request.set_body_raw(json, strlen(json));
                                ResponseDataStr response = request.post();
                                if (response.status_ok && response.y) {
                                    printf("[APP2] response message: %s\n", response.y);
                                } else {
                                    printf("[APP2] request failed\n");
                                }
                                send_arm_event = false;
                                stream_index = 0;
                                last_send_ms = now;
                            }
                        }
                    }
                }

                if (net_state == NET_ERR) {
                    printf("[APP2] timeout/error -> back to APP1\n");
                    armed = false;
                    mode = MODE_GESTURE;
                    current_state = STARTING;
                    net_state = NET_IDLE;
                }
            }
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

        if (wifi_ready) {
            cyw43_arch_poll();
        }
        sleep_ms(10); 
    }
    return 0;
}
