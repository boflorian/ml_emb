#include <cmath> 
#include<iostream>
#include <cstdlib> 
#include <iostream>
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

using namespace std; 
#define HALT_CORE_1() while (1) { tight_loop_contents(); }

// Define gesture window size (number of IMU samples per inference)
#define GESTURE_WINDOW_SIZE 100  // Adjust based on model input size

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
        
        // pass the ML inputs, output, and semaphore
        TP_DrawBoard();
      }
    }
}

int main(void) 
{
    System_Init();
    mutex_init(&mutex);  // Initialize the mutex

    sleep_ms(5000);

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
    LCD_SCAN_DIR bmp_scan_dir = D2U_R2L;
    TP_GetAdFac();
    reset_inference(&inference);
	init_gui();


    // run core1 loop that handles user interface
    multicore_launch_core1(core1_entry);

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
    float imu_buffer[GESTURE_WINDOW_SIZE * 6];
    int buffer_index = 0;

    while (true) {
        // Read IMU data
        IMU_ST_SENSOR_DATA gyro, accel;
        imuDataAccGyrGet(&gyro, &accel);

        // Print or process the data
        printf("Accel: X=%d, Y=%d, Z=%d | Gyro: X=%d, Y=%d, Z=%d\n",
               accel.s16X, accel.s16Y, accel.s16Z,
               gyro.s16X, gyro.s16Y, gyro.s16Z);

        // Collect data in buffer (normalize to float if needed)
        imu_buffer[buffer_index * 6 + 0] = (float)accel.s16X / 32768.0f; // Assuming 16-bit signed
        imu_buffer[buffer_index * 6 + 1] = (float)accel.s16Y / 32768.0f;
        imu_buffer[buffer_index * 6 + 2] = (float)accel.s16Z / 32768.0f;
        imu_buffer[buffer_index * 6 + 3] = (float)gyro.s16X / 32768.0f;
        imu_buffer[buffer_index * 6 + 4] = (float)gyro.s16Y / 32768.0f;
        imu_buffer[buffer_index * 6 + 5] = (float)gyro.s16Z / 32768.0f;

        buffer_index++;

        // When buffer is full, run inference
        if (buffer_index >= GESTURE_WINDOW_SIZE) {
            // Copy buffer to model input (assuming float input; adjust if quantized)
            memcpy(test_image_input, imu_buffer, byte_size);

            // Run inference
            int result = ml_model.predict();
            if (result == -1) {
                printf("Failed to run inference\n");
            } else {
                printf("Predicted Gesture: %d\n", result);
                // Display on LCD
                char str[32];
                sprintf(str, "Gesture: %d", result);
                GUI_Clear(BLACK);  // Clear screen
                GUI_DisString_EN(10, 50, str, &Font24, WHITE, BLACK);
                GUI_Show();  // Update display
            }

            // Reset buffer
            buffer_index = 0;
        }

        sleep_ms(10); // Adjust sampling rate (e.g., 100 Hz)
    }
    return 0;
}
