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

static void open_log_file(FIL *f, const char *drive, const char *prefix) {
    char path[96];
    uint64_t t = time_us_64();

    snprintf(path, sizeof path, "%s/%s_%llu.txt", drive, prefix, (unsigned long long)t);

    FRESULT fr = f_open(f, path, FA_WRITE | FA_CREATE_ALWAYS);
    if (fr != FR_OK) die(fr, "f_open");
    printf("logging to: %s\n", path);
}


static void core1_entry(void) 
{
    const char *drive = sd_mount_or_format(&fs, 0);

    uint32_t last_seen_session = 0;

    // Open single file for all sessions
    FIL f;
    open_log_file(&f, drive, FILE_NAME_PREFIX);

    for (;;) {
        while (SESSION == last_seen_session && !STOP_ALL) { tight_loop_contents(); }
        if (STOP_ALL) break;

        // Wait for recording to actually start before proceeding
        while (!RECORD && !STOP_ALL) { tight_loop_contents(); }
        if (STOP_ALL) break;

        printf("[Core1] Writing sample%u to file...\n", SESSION);

        // Write sample header for this session
        char header[96];
        int hdr_len = snprintf(header, sizeof header, "sample%u\nax,ay,az\n", SESSION);
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
        printf("[Core1] Sample %u written\n", SESSION);

        last_seen_session = SESSION;
    }

    f_sync(&f);
    f_close(&f);
    f_unmount(drive);
    while (1) tight_loop_contents();
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
    imuInit(&enMotionSensorType);
    printf("Device setup complete on core0\n");
    show_color_rgb(0, 255, 0);

    for (uint32_t i = 0; i < RECORD_TIMES; i++) 
    {     
        SESSION++;
        printf("\n========================================\n");
        printf("Starting Sample %u of %u\n", SESSION, RECORD_TIMES);
        printf("========================================\n");
        
        RECORD = true;   

        show_color_rgb(0, 255, 0);
        
        uint32_t start_time = time_us_64();
        while (time_us_64() < start_time + MAX_DATA_COLLECTION_TIME_US) 
        {
            imuDataOnlyGet(&stGyroRawData, &stAccelRawData);

            imu_sample_t s = {
                .ax = stAccelRawData.s16X, .ay = stAccelRawData.s16Y, .az = stAccelRawData.s16Z,
            };

            if (!queue_try_add(&sample_q, &s)) {
                show_color_rgb(0, 0, 255); // overflow indicator
            }

            // sleep_us(1000); // 1 kHz source rate
            // sleep_us(20000); // 50 Hz source rate
            // sleep_us(40000); // 25 Hz source rate
            sleep_us(10000); // 100 Hz source rate
        }

        RECORD = false;
        printf("Sample %u completed. Waiting for file write...\n", SESSION);

        // End of session
        (show_color_rgb(1,1,1), sleep_ms(250),show_color_rgb(0,0,255), sleep_ms(250),show_color_rgb(1,1,1), sleep_ms(250),show_color_rgb(0,0,255), sleep_ms(250)); 
    }

    STOP_ALL = true;
    printf("\n========================================\n");
    printf("All %u samples completed!\n", RECORD_TIMES);
    printf("========================================\n");

    for(;;){ show_color_rgb(0,0,255); sleep_ms(250); show_color_rgb(1,1,1); sleep_ms(250); }
    return 0;
}