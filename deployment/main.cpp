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

// Define gesture window size (number of IMU samples per inference)
#define GESTURE_WINDOW_SIZE 100  // Adjust based on model input size
#define IMU_BYTES_PER_SAMPLE 6   // ax, ay, az, gx, gy, gz

INFERENCE inference;
Model ml_model;

mutex_t mutex;  // Declare a mutex


int count_digits(int number) {
    if (number == 0) {
        return 1; // Special case for 0, which has 1 digit
    }

    number = abs(number); // Handle negative numbers
    return std::floor(std::log10(number)) + 1;
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
    printf("System initialized\n");
    mutex_init(&mutex);  // Initialize the mutex

    sleep_ms(5000);
    printf("Sleep done\n");

    // Initialize IMU
    IMU_EN_SENSOR_TYPE imu_type;
    imuInit(&imu_type);
    if (imu_type != IMU_EN_SENSOR_TYPE_ICM20948) {
        printf("IMU not detected\n");
        HALT_CORE_1();
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

    // Keep the white background from init_gui()


    // run core1 loop that handles user interface
    multicore_launch_core1(core1_entry);
    printf("Core1 launched\n");

        // initialize ML model
    if (!ml_model.setup()) {
        printf("Failed to initialize ML model!\n");
        HALT_CORE_1();
    }
    printf("Model initialized\n");
    
    uint8_t* test_image_input = ml_model.input_data();
    if (test_image_input == nullptr) {
        printf("Cannot set input\n");
        HALT_CORE_1();
    }

    int byte_size = ml_model.byte_size();
    if (!byte_size) {
        printf("Byte size not found\n");
        HALT_CORE_1();
    }

    // Buffer for IMU data: GESTURE_WINDOW_SIZE samples * 6 features (ax,ay,az,gx,gy,gz)
    const int imu_buffer_bytes = GESTURE_WINDOW_SIZE * IMU_BYTES_PER_SAMPLE;
    uint8_t imu_buffer[imu_buffer_bytes];
    int buffer_index = 0;
    int loop_counter = 0;
    bool heartbeat_on = false;

#ifdef PICO_DEFAULT_LED_PIN
    gpio_init(PICO_DEFAULT_LED_PIN);
    gpio_set_dir(PICO_DEFAULT_LED_PIN, GPIO_OUT);
    gpio_put(PICO_DEFAULT_LED_PIN, 0);
#endif

    while (true) {
        printf("Entering main loop iteration\n");
        // Read IMU data
        IMU_ST_SENSOR_DATA gyro, accel;
        imuDataAccGyrGet(&gyro, &accel);

        // Print or process the data
        printf("Accel: X=%d, Y=%d, Z=%d | Gyro: X=%d, Y=%d, Z=%d\n",
               accel.s16X, accel.s16Y, accel.s16Z,
               gyro.s16X, gyro.s16Y, gyro.s16Z);

        // Collect data in buffer, quantize to uint8 (assuming model expects 0-255 for -1 to 1)
        float ax = (float)accel.s16X / 32768.0f;
        float ay = (float)accel.s16Y / 32768.0f;
        float az = (float)accel.s16Z / 32768.0f;
        float gx = (float)gyro.s16X / 32768.0f;
        float gy = (float)gyro.s16Y / 32768.0f;
        float gz = (float)gyro.s16Z / 32768.0f;

        const int write_offset = buffer_index * IMU_BYTES_PER_SAMPLE;
        imu_buffer[write_offset + 0] = (uint8_t)((ax + 1.0f) * 127.5f);
        imu_buffer[write_offset + 1] = (uint8_t)((ay + 1.0f) * 127.5f);
        imu_buffer[write_offset + 2] = (uint8_t)((az + 1.0f) * 127.5f);
        imu_buffer[write_offset + 3] = (uint8_t)((gx + 1.0f) * 127.5f);
        imu_buffer[write_offset + 4] = (uint8_t)((gy + 1.0f) * 127.5f);
        imu_buffer[write_offset + 5] = (uint8_t)((gz + 1.0f) * 127.5f);

        buffer_index++;

        // When buffer is full, run inference
        if (buffer_index >= GESTURE_WINDOW_SIZE) {
            printf("Buffer full, running inference\n");
            // Copy buffer to model input
            const int bytes_to_copy = byte_size < imu_buffer_bytes ? byte_size : imu_buffer_bytes;
            memcpy(test_image_input, imu_buffer, bytes_to_copy);
            if (bytes_to_copy < byte_size) {
                // Zero any remaining input bytes so we don't leave stale data
                memset(test_image_input + bytes_to_copy, 0, byte_size - bytes_to_copy);
            }

            // Run inference
            int result = ml_model.predict();
            printf("Inference result: %d\n", result);
            if (result == -1) {
                printf("Failed to run inference\n");
            } else {
                printf("Predicted Gesture: %d\n", result);
                // Display gesture on LCD
                char str[32];
                sprintf(str, "Gesture: %d", result);
                GUI_DisString_EN(10, 100, str, &Font24, WHITE, BLACK);
            }

            // Reset buffer
            buffer_index = 0;
        }

        // Heartbeat: toggle LED and a small on-screen marker so we know the loop is alive
        if ((loop_counter++ % 50) == 0) {
            heartbeat_on = !heartbeat_on;
#ifdef PICO_DEFAULT_LED_PIN
            gpio_put(PICO_DEFAULT_LED_PIN, heartbeat_on ? 1 : 0);
#endif
            GUI_DisString_EN(10, 200, heartbeat_on ? "*" : " ", &Font16, WHITE, BLACK);
        }

        sleep_ms(10); // Adjust sampling rate (e.g., 100 Hz)
    }
    return 0;
}
