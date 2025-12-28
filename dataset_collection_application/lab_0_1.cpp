#include <stdio.h>
#include <string.h>
#include "pico/stdlib.h"
#include "hardware/pio.h"
#include "pico/multicore.h"
#include "pico/util/queue.h"

#include "icm20948.h"
#include "ws2812.pio.h"
#include "ff.h"         
#include "sd_card.h"     
#include "f_util.h"     
#include "hw_config.h" 

const uint WS2812_PIN = 4;
static FATFS fs;

// Configuration
// Data collection parameters. Edit as needed.
const uint32_t MAX_DATA_COLLECTION_TIME_US = 3 * 1000 * 1000; // duration of session 
uint32_t RECORD_TIMES = 20; // number of sessions 


// Control flags
volatile bool RECORD = false;        
volatile uint32_t SESSION = 0;     
volatile bool STOP_ALL = false;     


char FILE_NAME_PREFIX[32] = "train";

typedef struct __attribute__((packed)) 
{
    int16_t ax, ay, az;          // accel
    uint8_t  _pad[2];            // pad to 8 bytes total
} imu_sample_t;

#define QUEUE_DEPTH  (8192)      // 8192 * 8B = 64 KB ring -> ~8s at 1 kHz
static queue_t sample_q;

#define ACCEL_SCALE (1.0f / 16384.0f)

static void die(FRESULT fr, const char *op) 
{
    printf("%s failed: %s (%d)\n", op, FRESULT_str(fr), fr);
    while (1) tight_loop_contents();
}

static void halt_forever(const char *msg) 
{
    if (msg) printf("%s\n", msg);
    while (1) tight_loop_contents();
}

static inline void put_pixel(uint32_t pixel_grb) 
{
    pio_sm_put_blocking(pio0, 0, pixel_grb << 8u);
}

static inline uint32_t rgb_to_grb_u32(uint8_t r, uint8_t g, uint8_t b) 
{
    return ((uint32_t)g << 16) | ((uint32_t)r << 8) | (uint32_t)b;
}

static inline void show_color_rgb(uint8_t r, uint8_t g, uint8_t b) 
{
    uint32_t grb = rgb_to_grb_u32(r, g, b);
    put_pixel(grb);
}

void init_pio_for_ws2812()
{
    PIO pio = pio0;
    int sm = 0;
    uint offset = pio_add_program(pio, &ws2812_program);

    ws2812_program_init(pio, sm, offset, WS2812_PIN, 800000, true);
}

static const char *sd_mount_or_format(FATFS *pfs, int card_num) 
{
    if (!sd_init_driver()) {
        printf("sd_init_driver() failed\n");
        halt_forever(NULL);
    }
    sd_card_t *sd = sd_get_by_num(card_num);
    if (!sd) {
        printf("No SD config found (sd_get_by_num(%d) == NULL)\n", card_num);
        halt_forever(NULL);
    }
    const char *drive = sd_get_drive_prefix(sd);

    FRESULT fr = f_mount(pfs, drive, 1);
    printf("f_mount -> %s (%d)\n", FRESULT_str(fr), fr);

    if (fr == FR_NO_FILESYSTEM) {
        BYTE work[4096];                 
        MKFS_PARM opt = { FM_FAT | FM_SFD, 0, 0, 0, 0 };
        fr = f_mkfs(drive, &opt, work, sizeof work);
        printf("f_mkfs -> %s (%d)\n", FRESULT_str(fr), fr);
        if (fr != FR_OK) die(fr, "f_mkfs");

        fr = f_mount(pfs, drive, 1);
        printf("f_mount(after mkfs) -> %s (%d)\n", FRESULT_str(fr), fr);
    }
    if (fr != FR_OK) die(fr, "f_mount");

    return drive;
}

static void open_log_file(FIL *f, const char *drive, const char *gesture, const char *prefix) {
    char path[96];
    char dir_path[96];

    // Create directory for the gesture
    snprintf(dir_path, sizeof dir_path, "%s/%s", drive, gesture);
    FRESULT fr = f_mkdir(dir_path);
    if (fr != FR_OK && fr != FR_EXIST) die(fr, "f_mkdir");

    // Create file path within the gesture directory
    uint64_t t = time_us_64();
    snprintf(path, sizeof path, "%s/%s_%llu.txt", dir_path, prefix, (unsigned long long)t);

    fr = f_open(f, path, FA_WRITE | FA_CREATE_ALWAYS);
    if (fr != FR_OK) die(fr, "f_open");
    printf("logging to: %s\n", path);
}


static void core1_entry(void) 
{
    const char *drive = sd_mount_or_format(&fs, 0);

    uint32_t last_seen_session = 0;

    for (;;) {
        // Wait for gesture name from core0
        const char* gesture = (const char*)multicore_fifo_pop_blocking();

        // Open single file for all sessions within the gesture folder
        FIL f;
        open_log_file(&f, drive, gesture, FILE_NAME_PREFIX);

        while (SESSION == last_seen_session && !STOP_ALL) { tight_loop_contents(); }
        if (STOP_ALL) break;

        // Wait for recording to actually start before proceeding
        while (!RECORD && !STOP_ALL) { tight_loop_contents(); }
        if (STOP_ALL) break;

        printf("[Core1] Writing sample%lu to file...\n", SESSION);

        // Write sample header for this session
        char header[96];
        int hdr_len = snprintf(header, sizeof header, "sample%lu\nax,ay,az\n", SESSION);
        UINT bw = 0;
        FRESULT fr = f_write(&f, header, (UINT)hdr_len, &bw);
        if (fr != FR_OK || bw != (UINT)hdr_len) die(fr, "f_write(sample header)");

        uint32_t lines_since_sync = 0;
        imu_sample_t s;

        // Wait for data to start arriving in the queue
        while (RECORD && queue_is_empty(&sample_q)) { tight_loop_contents(); }

        for (;;) 
        {
            // Use non-blocking check to avoid getting stuck
            if (!queue_try_remove(&sample_q, &s)) {
                if (!RECORD && queue_is_empty(&sample_q)) break; // session done
                tight_loop_contents();
                continue;
            }

            char line[96];
            int n = snprintf(line, sizeof line, "%d,%d,%d\n",
                                s.ax, s.ay, s.az);
            bw = 0;
            fr = f_write(&f, line, (UINT)n, &bw);
            if (fr != FR_OK || bw != (UINT)n) die(fr, "f_write(sample)");
            if (++lines_since_sync >= 128) { f_sync(&f); lines_since_sync = 0; }
        }

        // Write 4 empty lines between samples
        const char *separator = "\n\n\n\n";
        bw = 0;
        fr = f_write(&f, separator, (UINT)strlen(separator), &bw);
        if (fr != FR_OK || bw != strlen(separator)) die(fr, "f_write(separator)");

        f_sync(&f);
        printf("[Core1] Sample %lu written\n", SESSION);

        last_seen_session = SESSION;

        f_sync(&f);
        f_close(&f);
    }

    f_unmount(drive);
    while (1) tight_loop_contents();
}

// Move countdown function outside of main
static void countdown(int seconds) {
    for (int i = seconds; i > 0; --i) {
        printf("Starting in %d...\n", i);
        sleep_ms(1000);
    }
}

// Replace preprocessing functions with those from inference
static void lowpass_filter(float* x, int length, int window) {
    float* temp = new float[length];

    // Ensure window size is valid
    if (window <= 0) {
        printf("Invalid window size: %d\n", window);
        return;
    }

    // Apply low-pass filter
    for (int i = 0; i < length; i++) {
        float sum = 0.0f;
        int count = 0;
        for (int j = i - window / 2; j <= i + window / 2; j++) {
            if (j >= 0 && j < length) {
                sum += x[j];
                count++;
            }
        }
        temp[i] = sum / count;
    }
    memcpy(x, temp, length * sizeof(float));
    delete[] temp;
}

static void apply_lowpass_filter(float* buffer, int window_size, int window, int features) {
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

static void normalize_clip(float* buffer, int window_size, int features) {
    // Skip normalization for small window sizes
    if (window_size <= 1) {
        printf("Skipping normalization for window size: %d\n", window_size);
        return;
    }

    // Clip to [-80, 80] and log clipped values
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
            printf("Axis %d std too small, skipping normalization\n", axis);
            continue;
        }

        for (int t = 0; t < window_size; t++) {
            buffer[t * features + axis] = (buffer[t * features + axis] - mean) / std;
        }
    }
}

int main() 
{
    stdio_init_all();
    sleep_ms(6000);

    init_pio_for_ws2812();
    show_color_rgb(255, 0, 0);

    queue_init(&sample_q, sizeof(imu_sample_t), QUEUE_DEPTH);
    multicore_launch_core1(core1_entry);
    printf("Device setup complete on core1\n");

    IMU_EN_SENSOR_TYPE enMotionSensorType;
    IMU_ST_SENSOR_DATA stGyroRawData, stAccelRawData;
    printf("Initializing IMU...\n");
    imuInit(&enMotionSensorType);
    if (enMotionSensorType == IMU_EN_SENSOR_TYPE_NULL) {
        printf("IMU initialization failed. Sensor type: NULL\n");
    } else {
        printf("IMU initialized successfully. Sensor type: ICM20948\n");
    }
    printf("Device setup complete on core0\n");
    show_color_rgb(0, 255, 0);

    // Define gestures and samples per gesture
    const char* GESTURES[] = {"ring", "wave", "slope", "negative"};
    const uint32_t NUM_GESTURES = sizeof(GESTURES) / sizeof(GESTURES[0]);
    const uint32_t SAMPLES_PER_GESTURE = 10; // Configurable number of samples per gesture

    // Update the main loop to pass gesture names to core1
    for (uint32_t gesture_index = 0; gesture_index < NUM_GESTURES; ++gesture_index) {
        const char* current_gesture = GESTURES[gesture_index];
        printf("\n========================================\n");
        printf("Next Gesture: %s\n", current_gesture);
        printf("========================================\n");

        for (uint32_t sample_index = 0; sample_index < SAMPLES_PER_GESTURE; ++sample_index) {
            printf("\nRecording sample %lu of %lu for gesture: %s\n", sample_index + 1, SAMPLES_PER_GESTURE, current_gesture);
            countdown(3); // 3-second countdown

            SESSION++;
            RECORD = true;
            show_color_rgb(0, 255, 0);

            // Pass gesture name to core1 for file handling
            multicore_fifo_push_blocking((uintptr_t)current_gesture);

            uint32_t start_time = time_us_64();
            while (time_us_64() < start_time + MAX_DATA_COLLECTION_TIME_US) {
                imuDataOnlyGet(&stGyroRawData, &stAccelRawData);

                imu_sample_t s = {
                    .ax = stAccelRawData.s16X, .ay = stAccelRawData.s16Y, .az = stAccelRawData.s16Z,
                    ._pad = {0}
                };

                // Convert raw data to float
                float data[3] = {
                    (float)s.ax * ACCEL_SCALE,
                    (float)s.ay * ACCEL_SCALE,
                    (float)s.az * ACCEL_SCALE
                };

                // Apply preprocessing
                apply_lowpass_filter(data, 1, 1, 3); // Example window size of 1
                normalize_clip(data, 1, 3);

                // Debug prints to log raw and preprocessed accelerometer data
                printf("Raw accelerometer data: ax=%d, ay=%d, az=%d\n", s.ax, s.ay, s.az);
                printf("Preprocessed accelerometer data: ax=%.2f, ay=%.2f, az=%.2f\n", data[0], data[1], data[2]);

                // Add preprocessed data to the queue
                if (!queue_try_add(&sample_q, &data)) {
                    show_color_rgb(0, 0, 255); // overflow indicator
                }

                sleep_us(10000); // 100 Hz source rate
            }

            RECORD = false;
            printf("Sample %lu for gesture %s completed.\n", sample_index + 1, current_gesture);
            show_color_rgb(1, 1, 1);
            sleep_ms(250);
        }
    }

    STOP_ALL = true;
    printf("\n========================================\n");
    printf("All gestures and samples completed!\n");
    printf("========================================\n");

    for(;;){ show_color_rgb(0,0,255); sleep_ms(250); show_color_rgb(1,1,1); sleep_ms(250); }
    return 0;
}