#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include "pico/stdlib.h"
#include "pico/multicore.h"
#include "pico/util/queue.h"
#include "pico/time.h"

#include "ff.h"
#include "sd_card.h"
#include "f_util.h"
#include "hw_config.h"

// FFT API (in data_preprocessing)
#include "fft.h"
#include "quantization.h"
#include "statistic.h"

// For IMU Logging 
# include "icm20948.h"
#include "pico/util/queue.h"


// ============================================================================
// CONFIGURATION SETTINGS - Adjust these to change system behavior
// ============================================================================
#define SAMPLE_RATE_HZ 200      // Desired sampling rate in Hz (must be integer)
                                 // Common values: 50, 100, 200
                                 // Sleep time = 1000 / SAMPLE_RATE_HZ milliseconds
#define MAX_SAMPLES 1000        // Maximum number of samples to collect before stopping
                                 // Total duration = MAX_SAMPLES / SAMPLE_RATE_HZ seconds
                                 // Example: 1000 samples @ 100 Hz = 10 seconds

// ============================================================================
// System Constants (usually don't need to change these)
// ============================================================================
#define PATH_MAX_LEN 256
#define QUEUE_DEPTH 1024
#define N_FFT 256               // FFT window size (must be power of 2)
#define FFT_PEAKS 3             // Number of spectral peaks to extract per axis

// ICM-20948 sensor normalization constants
// Step 1: Convert raw int16 to physical units
#define ACCEL_SCALE (1.0f / 16384.0f)  // Converts LSB to g (±2g range, 16384 LSB/g)
#define GYRO_SCALE (1.0f / 131.0f)     // Converts LSB to dps (±250 dps range, 131 LSB/dps)
#define MAG_SCALE 0.15f                 // Converts LSB to µT (±4900 µT range, 0.15 µT/LSB)

// Step 2: Scale physical units to [-1, 1) range for Q15 quantization
// Use normalization ranges that avoid mathematical cancellation and provide headroom
// Math check: (1/16384) × 4.0 × 32768 = 8.0 (NOT 1.0, so cancellation avoided!)
#define ACCEL_RANGE 4.0f                // ±4g normalization (2x sensor range for headroom)
#define GYRO_RANGE 500.0f               // ±500 dps normalization (2x sensor range for headroom)
#define MAG_RANGE 10000.0f              // ±10000 µT normalization (2x sensor range for headroom)


// --------- Globals (FatFs requires the FS to outlive the mount) ----------
static FATFS fs;                 // must be static/global (lives as long as the mount)
static sd_card_t *g_sd = NULL;   // active SD card
static const char *g_drive = NULL; // typically "0:"

// ------------------------- Utility / Error -------------------------------
static void die(FRESULT fr, const char *op) {
    printf("%s failed: %s (%d)\n", op, FRESULT_str(fr), fr);
    while (1) tight_loop_contents();
}

static void loop_forever_msg(const char *msg) {
    printf("%s\n", msg);
    while (1) tight_loop_contents();
}

static void join_path(char *out, size_t out_sz, const char *drive, const char *rel) {
    // drive = "0:" or "0:/", ensure exactly one slash when joining
    if (rel && rel[0] == '/') rel++; // avoid double slashes
    if (drive && drive[strlen(drive) - 1] == '/')
        snprintf(out, out_sz, "%s%s", drive, rel ? rel : "");
    else
        snprintf(out, out_sz, "%s/%s", drive, rel ? rel : "");
}

// ------------------------- 1) Initialization -----------------------------
static bool sd_init_and_mount(void) {
    printf("Initializing SD card\n");
    if (!sd_init_driver()) {
        printf("sd_init_driver() failed\n");
        return false;
    }

    g_sd = sd_get_by_num(0);
    if (!g_sd) {
        printf("No SD config found (sd_get_by_num(0) == NULL)\n");
        return false;
    }

    g_drive = sd_get_drive_prefix(g_sd);  
    if (!g_drive) {
        printf("sd_get_drive_prefix() returned NULL\n");
        return false;
    }

    FRESULT fr = f_mount(&fs, g_drive, 1);
    printf("f_mount -> %s (%d)\n", FRESULT_str(fr), fr);

    if (fr == FR_NO_FILESYSTEM) {
        BYTE work[4096]; // >= FF_MAX_SS
        MKFS_PARM opt = { FM_FAT | FM_SFD, 0, 0, 0, 0 };
        fr = f_mkfs(g_drive, &opt, work, sizeof work);
        printf("f_mkfs -> %s (%d)\n", FRESULT_str(fr), fr);
        if (fr == FR_OK) {
            fr = f_mount(&fs, g_drive, 1);
            printf("f_mount(after mkfs) -> %s (%d)\n", FRESULT_str(fr), fr);
        }
    }

    if (fr != FR_OK) {
        printf("Mount failed: %s (%d)\n", FRESULT_str(fr), fr);
        return false;
    }

    return true;
}

// ------------------------- 2) File creation ------------------------------
static FRESULT create_file(const char *abs_path, FIL *out_file) {
    // Creates/truncates a file and opens it for writing
    printf("Creating file: %s\n", abs_path);
    return f_open(out_file, abs_path, FA_WRITE | FA_CREATE_ALWAYS);

    //return f_open(out_file, abs_path, FA_WRITE | FA_OPEN_ALWAYS);
}

// ------------------------- 3) File writing -------------------------------
static FRESULT write_to_file(FIL *file, const void *data, UINT len, UINT *bytes_written) {
    *bytes_written = 0;
    FRESULT fr = f_write(file, data, len, bytes_written);
    if (fr == FR_OK) {
        fr = f_sync(file); // ensure data hits the card
    }
    return fr;
}

// ------------------------- 4) File checking/listing ----------------------
typedef struct {
    uint32_t files;
    uint32_t dirs;
    uint64_t total_bytes;
} list_stats_t;

static bool is_dot_or_dotdot(const char *name) {
    return (name[0] == '.' && (name[1] == '\0' || (name[1] == '.' && name[2] == '\0')));
}

static FRESULT list_dir_recursive(const char *path, list_stats_t *stats) {
    DIR dir;
    FILINFO fno;
    FRESULT fr = f_opendir(&dir, path);
    if (fr != FR_OK) {
        printf("f_opendir('%s') -> %s (%d)\n", path, FRESULT_str(fr), fr);
        return fr;
    }

    for (;;) {
        fr = f_readdir(&dir, &fno);
        if (fr != FR_OK) {
            printf("f_readdir('%s') -> %s (%d)\n", path, FRESULT_str(fr), fr);
            break;
        }
        if (fno.fname[0] == '\0') break; // end of directory

        if (is_dot_or_dotdot(fno.fname)) continue;

        if (fno.fattrib & AM_DIR) {
            stats->dirs++;
            char subpath[PATH_MAX_LEN];
            snprintf(subpath, sizeof subpath, "%s/%s", path, fno.fname);
            printf("[DIR]  %s\n", subpath);
            fr = list_dir_recursive(subpath, stats);
            if (fr != FR_OK) break;
        } else {
            stats->files++;
            stats->total_bytes += (uint64_t)fno.fsize;
            printf("[FILE] %s/%s  (%lu bytes)\n", path, fno.fname, (unsigned long)fno.fsize);
        }
    }

    FRESULT frc = f_closedir(&dir);
    if (fr == FR_OK && frc != FR_OK) fr = frc;
    return fr;
}

// Public checker: lists all files and sizes, and tells if any exist
static FRESULT check_and_list_files(const char *root_drive) {
    // Build root path "0:/"
    char root[PATH_MAX_LEN];
    join_path(root, sizeof root, root_drive, ""); // ensures a trailing slash when we add children

    list_stats_t stats = {0};
    printf("\n--- SD Card File Listing for '%s' ---\n", root_drive);
    FRESULT fr = list_dir_recursive(root_drive, &stats);
    if (fr != FR_OK && fr != FR_NO_PATH) {
        printf("Directory listing aborted due to error.\n");
        return fr;
    }

    if (stats.files == 0 && stats.dirs == 0) {
        printf("No files or directories found on the SD card.\n");
    } else if (stats.files == 0) {
        printf("No files found (but %u director%s present).\n", stats.dirs, (stats.dirs == 1 ? "y" : "ies"));
    } else {
        printf("\nSummary: %u file%s in %u director%s, total %llu bytes.\n",
               stats.files, (stats.files == 1 ? "" : "s"),
               stats.dirs, (stats.dirs == 1 ? "y" : "ies"),
               (unsigned long long)stats.total_bytes);
    }
    return FR_OK;
}


// Debug 

void check_sd_ready(void) {
    FATFS *fs_ptr;
    DWORD free_clusters;
    FRESULT fr;

    fr = f_getfree("0:", &free_clusters, &fs_ptr);
    if (fr == FR_OK) {
        DWORD total_sectors = (fs_ptr->n_fatent - 2) * fs_ptr->csize;
        DWORD free_sectors  = free_clusters * fs_ptr->csize;

        printf("SD card ready!\n");
    } else {
        printf("f_getfree failed: %d\n", fr);
    }
}

typedef struct {
    int16_t ax, ay, az;
    int16_t gx, gy, gz;
    int16_t mx, my, mz;
    uint32_t t_us;
} Sample;

static queue_t sample_q;
static volatile bool imu_ready = false;  // Synchronization flag

static void core0_sampler(void)
{
    IMU_EN_SENSOR_TYPE type;
    imuInit(&type);
    
    // Allow IMU sensor to settle - read and discard samples during this time
    // Settling avoids wrong data in world z direction (gravity)
    printf("Waiting for IMU to settle (5 seconds)...\n");
    uint32_t settle_start = time_us_32();
    while ((time_us_32() - settle_start) < 5000000) {  // 5 seconds in microseconds
        // Read and discard samples during settling period
        IMU_ST_SENSOR_DATA stGyroRawData, stAccelRawData, stMagnRawData;
        imuDataGet(NULL, &stGyroRawData, &stAccelRawData, &stMagnRawData);
        sleep_ms(10);
    }
    
    imu_ready = true;
    printf("IMU ready, starting data collection at %d Hz.\n", SAMPLE_RATE_HZ);
    
    // Calculate sleep time in milliseconds from sample rate
    const uint32_t sleep_ms_val = 1000 / SAMPLE_RATE_HZ;
    printf("Sampling period: %u ms\n", sleep_ms_val);
    
    while (1) {
        IMU_ST_SENSOR_DATA stGyroRawData, stAccelRawData, stMagnRawData;
        // Read only raw sensor data (no angle calculations for speed)
        imuDataGet(NULL, &stGyroRawData, &stAccelRawData, &stMagnRawData);
        uint32_t t_now = (uint32_t)time_us_64();
        Sample s = { stAccelRawData.s16X, stAccelRawData.s16Y, stAccelRawData.s16Z,
                      stGyroRawData.s16X, stGyroRawData.s16Y, stGyroRawData.s16Z,
                      stMagnRawData.s16X, stMagnRawData.s16Y, stMagnRawData.s16Z,
                      t_now };
        queue_add_blocking(&sample_q, &s);
        sleep_ms(sleep_ms_val);
    }
}


// remove ? 
void test_sd_write(void) {
    char path[PATH_MAX_LEN];
    join_path(path, sizeof path, g_drive, "sd_test.txt");
    FIL f;
    FRESULT fr = create_file(path, &f);
    if (fr != FR_OK) {
        printf("Failed to create test file: %d\n", fr);
        return;
    }
    const char *msg = "SD card test successful!\n";
    UINT bw = 0;
    fr = write_to_file(&f, msg, strlen(msg), &bw);
    if (fr == FR_OK && bw == strlen(msg)) {
        printf("Test file written: %s\n", path);
    } else {
        printf("Failed to write test file: %d\n", fr);
    }
    f_close(&f);
}

void core1_writer(void) {
    sleep_ms(1500); // delays generally help to capture all output in serial terminal 

    if (!sd_init_and_mount()) {
        loop_forever_msg("SD init failed");
    }   

    char path[PATH_MAX_LEN];
    join_path(path, sizeof path, g_drive, "imu_log.csv");

    FIL file;
    FRESULT fr = create_file(path, &file);
    if (fr != FR_OK) die(fr, "f_open");

    UINT written = 0;
    // CSV Header with units clarification:
    // *_raw: raw int16 sensor values from IMU (LSB, unprocessed)
    // *_f32: physical units (ax/ay/az in g, gx/gy/gz in dps, mx/my/mz in µT)
    // *_norm: normalized values in [-1,1) range (physical units / sensor range)
    // *_q15: Q15 quantized int16 values (norm * 32768, using full int16 range for [-1,1))
    // *_dq: dequantized float values back to [-1,1) normalized range (q15 / 32768)
    const char *header = "timestamp_us,ax_raw,ay_raw,az_raw,gx_raw,gy_raw,gz_raw,mx_raw,my_raw,mz_raw,ax_f32,ay_f32,az_f32,gx_f32,gy_f32,gz_f32,mx_f32,my_f32,mz_f32,ax_norm,ay_norm,az_norm,gx_norm,gy_norm,gz_norm,mx_norm,my_norm,mz_norm,ax_q15,ay_q15,az_q15,gx_q15,gy_q15,gz_q15,mx_q15,my_q15,mz_q15,ax_dq,ay_dq,az_dq,gx_dq,gy_dq,gz_dq,mx_dq,my_dq,mz_dq\n";
    write_to_file(&file, header, strlen(header), &written);

    // Open a secondary file to store spectral peaks
    char spectrum_path[PATH_MAX_LEN];
    join_path(spectrum_path, sizeof spectrum_path, g_drive, "imu_spectra.csv");
    FIL spec_file;
    FRESULT fr2 = create_file(spectrum_path, &spec_file);
    if (fr2 == FR_OK) {
        const char *spec_hdr = "timestamp_us,axis,bin,freq_hz,amp\n";
        UINT bw; write_to_file(&spec_file, spec_hdr, strlen(spec_hdr), &bw);
    } else {
        printf("Warning: failed to create spectrum file: %d\n", fr2);
    }

    // FFT accumulation buffers (per-axis for all 9 channels)
    float buf_ax[N_FFT], buf_ay[N_FFT], buf_az[N_FFT];
    float buf_gx[N_FFT], buf_gy[N_FFT], buf_gz[N_FFT];
    float buf_mx[N_FFT], buf_my[N_FFT], buf_mz[N_FFT];
    int buf_pos = 0;

    // Sample counter and arrays for statistics (both raw and quantized)
    uint32_t sample_count = 0;
    // Accelerometer
    float *all_raw_ax = (float*)malloc(MAX_SAMPLES * sizeof(float));
    float *all_raw_ay = (float*)malloc(MAX_SAMPLES * sizeof(float));
    float *all_raw_az = (float*)malloc(MAX_SAMPLES * sizeof(float));
    float *all_q_ax = (float*)malloc(MAX_SAMPLES * sizeof(float));
    float *all_q_ay = (float*)malloc(MAX_SAMPLES * sizeof(float));
    float *all_q_az = (float*)malloc(MAX_SAMPLES * sizeof(float));
    // Gyroscope
    float *all_raw_gx = (float*)malloc(MAX_SAMPLES * sizeof(float));
    float *all_raw_gy = (float*)malloc(MAX_SAMPLES * sizeof(float));
    float *all_raw_gz = (float*)malloc(MAX_SAMPLES * sizeof(float));
    float *all_q_gx = (float*)malloc(MAX_SAMPLES * sizeof(float));
    float *all_q_gy = (float*)malloc(MAX_SAMPLES * sizeof(float));
    float *all_q_gz = (float*)malloc(MAX_SAMPLES * sizeof(float));
    // Magnetometer
    float *all_raw_mx = (float*)malloc(MAX_SAMPLES * sizeof(float));
    float *all_raw_my = (float*)malloc(MAX_SAMPLES * sizeof(float));
    float *all_raw_mz = (float*)malloc(MAX_SAMPLES * sizeof(float));
    float *all_q_mx = (float*)malloc(MAX_SAMPLES * sizeof(float));
    float *all_q_my = (float*)malloc(MAX_SAMPLES * sizeof(float));
    float *all_q_mz = (float*)malloc(MAX_SAMPLES * sizeof(float));
    
    if (!all_raw_ax || !all_raw_ay || !all_raw_az || !all_q_ax || !all_q_ay || !all_q_az ||
        !all_raw_gx || !all_raw_gy || !all_raw_gz || !all_q_gx || !all_q_gy || !all_q_gz ||
        !all_raw_mx || !all_raw_my || !all_raw_mz || !all_q_mx || !all_q_my || !all_q_mz) {
        printf("Failed to allocate memory for statistics\n");
        f_close(&file);
        if (fr2 == FR_OK) f_close(&spec_file);
        loop_forever_msg("Memory allocation failed");
    }

    // Wait for IMU to be ready before starting data collection
    printf("Waiting for IMU initialization...\n");
    while (!imu_ready) {
        sleep_ms(100);
    }
    printf("IMU ready signal received, starting data logging.\n");

    // Track timestamps for achieved sampling rate calculation
    uint32_t first_timestamp = 0;
    uint32_t last_timestamp = 0;

    // Dequeue samples, write CSV, accumulate buffers and run FFT when full
    while (sample_count < MAX_SAMPLES) {
        Sample s;
        queue_remove_blocking(&sample_q, &s);
        
        // Capture first and last timestamps
        if (sample_count == 0) {
            first_timestamp = s.t_us;
        }
        last_timestamp = s.t_us;

        // Convert raw int16 to physical units (float32)
        float f32_ax = (float)s.ax * ACCEL_SCALE;  // in g
        float f32_ay = (float)s.ay * ACCEL_SCALE;
        float f32_az = (float)s.az * ACCEL_SCALE;
        float f32_gx = (float)s.gx * GYRO_SCALE;   // in dps
        float f32_gy = (float)s.gy * GYRO_SCALE;
        float f32_gz = (float)s.gz * GYRO_SCALE;
        float f32_mx = (float)s.mx * MAG_SCALE;    // in µT
        float f32_my = (float)s.my * MAG_SCALE;
        float f32_mz = (float)s.mz * MAG_SCALE;
        
        // Normalize to [-1, 1) range for Q15 quantization
        float norm_ax = f32_ax / ACCEL_RANGE;
        float norm_ay = f32_ay / ACCEL_RANGE;
        float norm_az = f32_az / ACCEL_RANGE;
        float norm_gx = f32_gx / GYRO_RANGE;
        float norm_gy = f32_gy / GYRO_RANGE;
        float norm_gz = f32_gz / GYRO_RANGE;
        float norm_mx = f32_mx / MAG_RANGE;
        float norm_my = f32_my / MAG_RANGE;
        float norm_mz = f32_mz / MAG_RANGE;
        
        // Quantize normalized values to Q15 format
        float norm_vals[9] = {norm_ax, norm_ay, norm_az,
                              norm_gx, norm_gy, norm_gz,
                              norm_mx, norm_my, norm_mz};
        int16_t q15_vals[9];
        int clip_count = 0;
        
        quantize_q15(norm_vals, 9, q15_vals, &clip_count);
        
        // Dequantize Q15 back to float for CSV output
        float dequant_vals[9];
        dequantize_q15(q15_vals, 9, dequant_vals);
        
        // Write CSV line with all formats: raw int16, float32, normalized float, quantized int16, dequantized float
        char line[768];
        int len = snprintf(line, sizeof line, 
            "%u,"                                           // timestamp
            "%d,%d,%d,%d,%d,%d,%d,%d,%d,"                  // raw int16 (9 values)
            "%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f," // float32 physical units (9 values)
            "%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f," // normalized float [-1,1) (9 values)
            "%d,%d,%d,%d,%d,%d,%d,%d,%d,"                  // Q15 quantized int16 (9 values)
            "%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f\n", // dequantized float [-1,1) (9 values)
            s.t_us,
            s.ax, s.ay, s.az, s.gx, s.gy, s.gz, s.mx, s.my, s.mz,
            f32_ax, f32_ay, f32_az, f32_gx, f32_gy, f32_gz, f32_mx, f32_my, f32_mz,
            norm_ax, norm_ay, norm_az, norm_gx, norm_gy, norm_gz, norm_mx, norm_my, norm_mz,
            q15_vals[0], q15_vals[1], q15_vals[2], q15_vals[3], q15_vals[4], 
            q15_vals[5], q15_vals[6], q15_vals[7], q15_vals[8],
            dequant_vals[0], dequant_vals[1], dequant_vals[2], dequant_vals[3], dequant_vals[4],
            dequant_vals[5], dequant_vals[6], dequant_vals[7], dequant_vals[8]);
        
        if (len > 0 && len < (int)sizeof line) {
            fr = write_to_file(&file, line, (UINT)len, &written);
            if (fr != FR_OK || written != (UINT)len) {
                printf("IMU log write failed: %d\n", fr);
                break;
            }
        }

        // Store float32 values for statistics
        all_raw_ax[sample_count] = f32_ax;
        all_raw_ay[sample_count] = f32_ay;
        all_raw_az[sample_count] = f32_az;
        all_raw_gx[sample_count] = f32_gx;
        all_raw_gy[sample_count] = f32_gy;
        all_raw_gz[sample_count] = f32_gz;
        all_raw_mx[sample_count] = f32_mx;
        all_raw_my[sample_count] = f32_my;
        all_raw_mz[sample_count] = f32_mz;
        
        // Store dequantized values for statistics (these are in normalized [-1,1) range)
        all_q_ax[sample_count] = dequant_vals[0];
        all_q_ay[sample_count] = dequant_vals[1];
        all_q_az[sample_count] = dequant_vals[2];
        all_q_gx[sample_count] = dequant_vals[3];
        all_q_gy[sample_count] = dequant_vals[4];
        all_q_gz[sample_count] = dequant_vals[5];
        all_q_mx[sample_count] = dequant_vals[6];
        all_q_my[sample_count] = dequant_vals[7];
        all_q_mz[sample_count] = dequant_vals[8];

        // Accumulate DEQUANTIZED values for FFT to simulate Q15-based processing
        // This shows what the FFT would look like after quantization/dequantization
        buf_ax[buf_pos] = dequant_vals[0];
        buf_ay[buf_pos] = dequant_vals[1];
        buf_az[buf_pos] = dequant_vals[2];
        buf_gx[buf_pos] = dequant_vals[3];
        buf_gy[buf_pos] = dequant_vals[4];
        buf_gz[buf_pos] = dequant_vals[5];
        buf_mx[buf_pos] = dequant_vals[6];
        buf_my[buf_pos] = dequant_vals[7];
        buf_mz[buf_pos] = dequant_vals[8];
        buf_pos++;

        // --------------------------------- FFT -------------------------------

        if (buf_pos >= N_FFT) {            
            // Process each axis sequentially (all 9 channels)
            c32 X[N_FFT];
            float mag[N_FFT];
            int idx[FFT_PEAKS]; float val[FFT_PEAKS]; int found = 0;

            for (int axis = 0; axis < 9; axis++) {
                float *buf = (axis == 0) ? buf_ax : (axis == 1) ? buf_ay : (axis == 2) ? buf_az :
                             (axis == 3) ? buf_gx : (axis == 4) ? buf_gy : (axis == 5) ? buf_gz :
                             (axis == 6) ? buf_mx : (axis == 7) ? buf_my : buf_mz;

                // copy and window
                float tmp[N_FFT];
                for (int i = 0; i < N_FFT; i++) tmp[i] = buf[i];
                hamming_window(tmp, N_FFT);
                for (int i = 0; i < N_FFT; i++) { X[i].re = tmp[i]; X[i].im = 0.0f; }

                if (fft_radix2(X, N_FFT, +1) == 0) {
                    fft_mag(X, N_FFT, mag);
                    const float scale = (2.0f / (float)N_FFT) / 0.54f;
                    for (int k = 0; k <= N_FFT/2; k++) mag[k] *= scale;

                    top_k_peaks(mag, N_FFT/2 + 1, 1, FFT_PEAKS, idx, val, &found);

                    if (fr2 == FR_OK) {
                        // write peaks to spec file
                        for (int p = 0; p < found; p++) {
                            float freq = ((float)idx[p]) * ((float)SAMPLE_RATE_HZ / (float)N_FFT);
                            const char *axname = (axis == 0) ? "ax" : (axis == 1) ? "ay" : (axis == 2) ? "az" :
                                                 (axis == 3) ? "gx" : (axis == 4) ? "gy" : (axis == 5) ? "gz" :
                                                 (axis == 6) ? "mx" : (axis == 7) ? "my" : "mz";
                            char spec_line[128];
                            int l2 = snprintf(spec_line, sizeof spec_line, "%u,%s,%d,%.3f,%.6f\n",
                                              s.t_us, axname, idx[p], freq, val[p]);
                            if (l2 > 0 && l2 < (int)sizeof spec_line) {
                                UINT bw; write_to_file(&spec_file, spec_line, (UINT)l2, &bw);
                            }
                        }
                    }
                }
            }

            buf_pos = 0;
        }
        
        sample_count++;
    }
    // -------------------------------- End of sampling loop -----------------------------
    
    printf("\nReached sample limit (%u samples). Computing statistics...\n", sample_count);
    
    // Calculate statistics from RAW values
    float raw_mean_ax = mean_f32(all_raw_ax, sample_count);
    float raw_mean_ay = mean_f32(all_raw_ay, sample_count);
    float raw_mean_az = mean_f32(all_raw_az, sample_count);
    
    float raw_var_ax = variance_f32(all_raw_ax, sample_count, raw_mean_ax);
    float raw_var_ay = variance_f32(all_raw_ay, sample_count, raw_mean_ay);
    float raw_var_az = variance_f32(all_raw_az, sample_count, raw_mean_az);
    
    float raw_std_ax = stddev_f32(raw_var_ax);
    float raw_std_ay = stddev_f32(raw_var_ay);
    float raw_std_az = stddev_f32(raw_var_az);
    
    float raw_min_ax, raw_max_ax, raw_min_ay, raw_max_ay, raw_min_az, raw_max_az;
    min_max_f32(all_raw_ax, sample_count, &raw_min_ax, &raw_max_ax);
    min_max_f32(all_raw_ay, sample_count, &raw_min_ay, &raw_max_ay);
    min_max_f32(all_raw_az, sample_count, &raw_min_az, &raw_max_az);
    
    float raw_med_ax = median_f32(all_raw_ax, sample_count);
    float raw_med_ay = median_f32(all_raw_ay, sample_count);
    float raw_med_az = median_f32(all_raw_az, sample_count);
    
    // Gyroscope raw statistics
    float raw_mean_gx = mean_f32(all_raw_gx, sample_count);
    float raw_mean_gy = mean_f32(all_raw_gy, sample_count);
    float raw_mean_gz = mean_f32(all_raw_gz, sample_count);
    float raw_std_gx = stddev_f32(variance_f32(all_raw_gx, sample_count, raw_mean_gx));
    float raw_std_gy = stddev_f32(variance_f32(all_raw_gy, sample_count, raw_mean_gy));
    float raw_std_gz = stddev_f32(variance_f32(all_raw_gz, sample_count, raw_mean_gz));
    float raw_min_gx, raw_max_gx, raw_min_gy, raw_max_gy, raw_min_gz, raw_max_gz;
    min_max_f32(all_raw_gx, sample_count, &raw_min_gx, &raw_max_gx);
    min_max_f32(all_raw_gy, sample_count, &raw_min_gy, &raw_max_gy);
    min_max_f32(all_raw_gz, sample_count, &raw_min_gz, &raw_max_gz);
    
    // Magnetometer raw statistics
    float raw_mean_mx = mean_f32(all_raw_mx, sample_count);
    float raw_mean_my = mean_f32(all_raw_my, sample_count);
    float raw_mean_mz = mean_f32(all_raw_mz, sample_count);
    float raw_std_mx = stddev_f32(variance_f32(all_raw_mx, sample_count, raw_mean_mx));
    float raw_std_my = stddev_f32(variance_f32(all_raw_my, sample_count, raw_mean_my));
    float raw_std_mz = stddev_f32(variance_f32(all_raw_mz, sample_count, raw_mean_mz));
    float raw_min_mx, raw_max_mx, raw_min_my, raw_max_my, raw_min_mz, raw_max_mz;
    min_max_f32(all_raw_mx, sample_count, &raw_min_mx, &raw_max_mx);
    min_max_f32(all_raw_my, sample_count, &raw_min_my, &raw_max_my);
    min_max_f32(all_raw_mz, sample_count, &raw_min_mz, &raw_max_mz);
    
    // Calculate statistics from QUANTIZED values
    float q_mean_ax = mean_f32(all_q_ax, sample_count);
    float q_mean_ay = mean_f32(all_q_ay, sample_count);
    float q_mean_az = mean_f32(all_q_az, sample_count);
    
    float q_var_ax = variance_f32(all_q_ax, sample_count, q_mean_ax);
    float q_var_ay = variance_f32(all_q_ay, sample_count, q_mean_ay);
    float q_var_az = variance_f32(all_q_az, sample_count, q_mean_az);
    
    float q_std_ax = stddev_f32(q_var_ax);
    float q_std_ay = stddev_f32(q_var_ay);
    float q_std_az = stddev_f32(q_var_az);
    
    float q_min_ax, q_max_ax, q_min_ay, q_max_ay, q_min_az, q_max_az;
    min_max_f32(all_q_ax, sample_count, &q_min_ax, &q_max_ax);
    min_max_f32(all_q_ay, sample_count, &q_min_ay, &q_max_ay);
    min_max_f32(all_q_az, sample_count, &q_min_az, &q_max_az);
    
    float q_med_ax = median_f32(all_q_ax, sample_count);
    float q_med_ay = median_f32(all_q_ay, sample_count);
    float q_med_az = median_f32(all_q_az, sample_count);
    
    // Gyroscope quantized statistics
    float q_mean_gx = mean_f32(all_q_gx, sample_count);
    float q_mean_gy = mean_f32(all_q_gy, sample_count);
    float q_mean_gz = mean_f32(all_q_gz, sample_count);
    float q_std_gx = stddev_f32(variance_f32(all_q_gx, sample_count, q_mean_gx));
    float q_std_gy = stddev_f32(variance_f32(all_q_gy, sample_count, q_mean_gy));
    float q_std_gz = stddev_f32(variance_f32(all_q_gz, sample_count, q_mean_gz));
    float q_min_gx, q_max_gx, q_min_gy, q_max_gy, q_min_gz, q_max_gz;
    min_max_f32(all_q_gx, sample_count, &q_min_gx, &q_max_gx);
    min_max_f32(all_q_gy, sample_count, &q_min_gy, &q_max_gy);
    min_max_f32(all_q_gz, sample_count, &q_min_gz, &q_max_gz);
    
    // Magnetometer quantized statistics
    float q_mean_mx = mean_f32(all_q_mx, sample_count);
    float q_mean_my = mean_f32(all_q_my, sample_count);
    float q_mean_mz = mean_f32(all_q_mz, sample_count);
    float q_std_mx = stddev_f32(variance_f32(all_q_mx, sample_count, q_mean_mx));
    float q_std_my = stddev_f32(variance_f32(all_q_my, sample_count, q_mean_my));
    float q_std_mz = stddev_f32(variance_f32(all_q_mz, sample_count, q_mean_mz));
    float q_min_mx, q_max_mx, q_min_my, q_max_my, q_min_mz, q_max_mz;
    min_max_f32(all_q_mx, sample_count, &q_min_mx, &q_max_mx);
    min_max_f32(all_q_my, sample_count, &q_min_my, &q_max_my);
    min_max_f32(all_q_mz, sample_count, &q_min_mz, &q_max_mz);
    
    // Calculate SNR and error metrics for quantization quality
    float *norm_raw_ax = (float*)malloc(sample_count * sizeof(float));
    float *norm_raw_ay = (float*)malloc(sample_count * sizeof(float));
    float *norm_raw_az = (float*)malloc(sample_count * sizeof(float));
    float *norm_raw_gx = (float*)malloc(sample_count * sizeof(float));
    float *norm_raw_gy = (float*)malloc(sample_count * sizeof(float));
    float *norm_raw_gz = (float*)malloc(sample_count * sizeof(float));
    float *norm_raw_mx = (float*)malloc(sample_count * sizeof(float));
    float *norm_raw_my = (float*)malloc(sample_count * sizeof(float));
    float *norm_raw_mz = (float*)malloc(sample_count * sizeof(float));
    
    // Declare SNR variables outside if block for later use
    float snr_ax = 0.0f, max_err_ax = 0.0f, rms_err_ax = 0.0f;
    float snr_ay = 0.0f, max_err_ay = 0.0f, rms_err_ay = 0.0f;
    float snr_az = 0.0f, max_err_az = 0.0f, rms_err_az = 0.0f;
    float snr_gx = 0.0f, max_err_gx = 0.0f, rms_err_gx = 0.0f;
    float snr_gy = 0.0f, max_err_gy = 0.0f, rms_err_gy = 0.0f;
    float snr_gz = 0.0f, max_err_gz = 0.0f, rms_err_gz = 0.0f;
    float snr_mx = 0.0f, max_err_mx = 0.0f, rms_err_mx = 0.0f;
    float snr_my = 0.0f, max_err_my = 0.0f, rms_err_my = 0.0f;
    float snr_mz = 0.0f, max_err_mz = 0.0f, rms_err_mz = 0.0f;
    bool snr_computed = false;
    
    if (norm_raw_ax && norm_raw_ay && norm_raw_az && 
        norm_raw_gx && norm_raw_gy && norm_raw_gz &&
        norm_raw_mx && norm_raw_my && norm_raw_mz) {
        // Normalize raw values to [-1, 1) range for comparison
        for (uint32_t i = 0; i < sample_count; i++) {
            norm_raw_ax[i] = all_raw_ax[i] / ACCEL_RANGE;
            norm_raw_ay[i] = all_raw_ay[i] / ACCEL_RANGE;
            norm_raw_az[i] = all_raw_az[i] / ACCEL_RANGE;
            norm_raw_gx[i] = all_raw_gx[i] / GYRO_RANGE;
            norm_raw_gy[i] = all_raw_gy[i] / GYRO_RANGE;
            norm_raw_gz[i] = all_raw_gz[i] / GYRO_RANGE;
            norm_raw_mx[i] = all_raw_mx[i] / MAG_RANGE;
            norm_raw_my[i] = all_raw_my[i] / MAG_RANGE;
            norm_raw_mz[i] = all_raw_mz[i] / MAG_RANGE;
        }
        
        // Compute SNR and error metrics for each axis
        
        snr_and_error(norm_raw_ax, all_q_ax, sample_count, &snr_ax, &max_err_ax, &rms_err_ax);
        snr_and_error(norm_raw_ay, all_q_ay, sample_count, &snr_ay, &max_err_ay, &rms_err_ay);
        snr_and_error(norm_raw_az, all_q_az, sample_count, &snr_az, &max_err_az, &rms_err_az);
        snr_and_error(norm_raw_gx, all_q_gx, sample_count, &snr_gx, &max_err_gx, &rms_err_gx);
        snr_and_error(norm_raw_gy, all_q_gy, sample_count, &snr_gy, &max_err_gy, &rms_err_gy);
        snr_and_error(norm_raw_gz, all_q_gz, sample_count, &snr_gz, &max_err_gz, &rms_err_gz);
        snr_and_error(norm_raw_mx, all_q_mx, sample_count, &snr_mx, &max_err_mx, &rms_err_mx);
        snr_and_error(norm_raw_my, all_q_my, sample_count, &snr_my, &max_err_my, &rms_err_my);
        snr_and_error(norm_raw_mz, all_q_mz, sample_count, &snr_mz, &max_err_mz, &rms_err_mz);
        
        snr_computed = true;
        
        // Print SNR and error analysis
        printf("\n========================================\n");
        printf("=== Quantization Quality Analysis ===\n");
        printf("========================================\n");
        printf("Comparing normalized raw vs quantized values\n\n");
        
        printf("Accelerometer X (ax):\n");
        printf("  SNR:         %.2f dB\n", snr_ax);
        printf("  Max Error:   %.6f\n", max_err_ax);
        printf("  RMS Error:   %.6f\n\n", rms_err_ax);
        
        printf("Accelerometer Y (ay):\n");
        printf("  SNR:         %.2f dB\n", snr_ay);
        printf("  Max Error:   %.6f\n", max_err_ay);
        printf("  RMS Error:   %.6f\n\n", rms_err_ay);
        
        printf("Accelerometer Z (az):\n");
        printf("  SNR:         %.2f dB\n", snr_az);
        printf("  Max Error:   %.6f\n", max_err_az);
        printf("  RMS Error:   %.6f\n\n", rms_err_az);
        
        printf("Gyroscope X (gx):\n");
        printf("  SNR:         %.2f dB\n", snr_gx);
        printf("  Max Error:   %.6f\n", max_err_gx);
        printf("  RMS Error:   %.6f\n\n", rms_err_gx);
        
        printf("Gyroscope Y (gy):\n");
        printf("  SNR:         %.2f dB\n", snr_gy);
        printf("  Max Error:   %.6f\n", max_err_gy);
        printf("  RMS Error:   %.6f\n\n", rms_err_gy);
        
        printf("Gyroscope Z (gz):\n");
        printf("  SNR:         %.2f dB\n", snr_gz);
        printf("  Max Error:   %.6f\n", max_err_gz);
        printf("  RMS Error:   %.6f\n\n", rms_err_gz);
        
        printf("Magnetometer X (mx):\n");
        printf("  SNR:         %.2f dB\n", snr_mx);
        printf("  Max Error:   %.6f\n", max_err_mx);
        printf("  RMS Error:   %.6f\n\n", rms_err_mx);
        
        printf("Magnetometer Y (my):\n");
        printf("  SNR:         %.2f dB\n", snr_my);
        printf("  Max Error:   %.6f\n", max_err_my);
        printf("  RMS Error:   %.6f\n\n", rms_err_my);
        
        printf("Magnetometer Z (mz):\n");
        printf("  SNR:         %.2f dB\n", snr_mz);
        printf("  Max Error:   %.6f\n", max_err_mz);
        printf("  RMS Error:   %.6f\n\n", rms_err_mz);
        
        // Free normalized arrays (we'll save SNR values for file writing)
        free(norm_raw_ax);
        free(norm_raw_ay);
        free(norm_raw_az);
        free(norm_raw_gx);
        free(norm_raw_gy);
        free(norm_raw_gz);
        free(norm_raw_mx);
        free(norm_raw_my);
        free(norm_raw_mz);
    } else {
        printf("Failed to allocate memory for SNR analysis\n");
    }
    
    // Print statistics to console
    printf("\n========================================\n");
    printf("=== IMU Data Statistics ===\n");
    printf("========================================\n");
    printf("Total samples: %u\n\n", sample_count);
    
    printf("NOTE: Raw values are in physical units:\n");
    printf("  - Accelerometer: g (1g = 9.81 m/s²)\n");
    printf("  - Gyroscope: dps (degrees per second)\n");
    printf("  - Magnetometer: µT (microtesla)\n");
    printf("  - Quantized values: normalized [-1, 1) range\n\n");
    
    printf("--- RAW Values (Physical Units) ---\n\n");
    printf("Accelerometer X (ax):\n");
    printf("  Mean:     %.6f\n", raw_mean_ax);
    printf("  Median:   %.6f\n", raw_med_ax);
    printf("  Variance: %.6f\n", raw_var_ax);
    printf("  Std Dev:  %.6f\n", raw_std_ax);
    printf("  Min:      %.6f\n", raw_min_ax);
    printf("  Max:      %.6f\n\n", raw_max_ax);
    
    printf("Accelerometer Y (ay):\n");
    printf("  Mean:     %.6f\n", raw_mean_ay);
    printf("  Median:   %.6f\n", raw_med_ay);
    printf("  Variance: %.6f\n", raw_var_ay);
    printf("  Std Dev:  %.6f\n", raw_std_ay);
    printf("  Min:      %.6f\n", raw_min_ay);
    printf("  Max:      %.6f\n\n", raw_max_ay);
    
    printf("Accelerometer Z (az):\n");
    printf("  Mean:     %.6f\n", raw_mean_az);
    printf("  Median:   %.6f\n", raw_med_az);
    printf("  Variance: %.6f\n", raw_var_az);
    printf("  Std Dev:  %.6f\n", raw_std_az);
    printf("  Min:      %.6f\n", raw_min_az);
    printf("  Max:      %.6f\n\n", raw_max_az);
    
    printf("--- QUANTIZED Values (Normalized [-1, 1) range) ---\n\n");
    printf("Accelerometer X (ax):\n");
    printf("  Mean:     %.6f\n", q_mean_ax);
    printf("  Median:   %.6f\n", q_med_ax);
    printf("  Variance: %.6f\n", q_var_ax);
    printf("  Std Dev:  %.6f\n", q_std_ax);
    printf("  Min:      %.6f\n", q_min_ax);
    printf("  Max:      %.6f\n\n", q_max_ax);
    
    printf("Accelerometer Y (ay):\n");
    printf("  Mean:     %.6f\n", q_mean_ay);
    printf("  Median:   %.6f\n", q_med_ay);
    printf("  Variance: %.6f\n", q_var_ay);
    printf("  Std Dev:  %.6f\n", q_std_ay);
    printf("  Min:      %.6f\n", q_min_ay);
    printf("  Max:      %.6f\n\n", q_max_ay);
    
    printf("Accelerometer Z (az):\n");
    printf("  Mean:     %.6f\n", q_mean_az);
    printf("  Median:   %.6f\n", q_med_az);
    printf("  Variance: %.6f\n", q_var_az);
    printf("  Std Dev:  %.6f\n", q_std_az);
    printf("  Min:      %.6f\n", q_min_az);
    printf("  Max:      %.6f\n\n", q_max_az);
    
    printf("Gyroscope X (gx):\n");
    printf("  Mean:     %.6f  Std Dev: %.6f  Min: %.6f  Max: %.6f\n\n", 
           q_mean_gx, q_std_gx, q_min_gx, q_max_gx);
    
    printf("Gyroscope Y (gy):\n");
    printf("  Mean:     %.6f  Std Dev: %.6f  Min: %.6f  Max: %.6f\n\n", 
           q_mean_gy, q_std_gy, q_min_gy, q_max_gy);
    
    printf("Gyroscope Z (gz):\n");
    printf("  Mean:     %.6f  Std Dev: %.6f  Min: %.6f  Max: %.6f\n\n", 
           q_mean_gz, q_std_gz, q_min_gz, q_max_gz);
    
    printf("Magnetometer X (mx):\n");
    printf("  Mean:     %.6f  Std Dev: %.6f  Min: %.6f  Max: %.6f\n\n", 
           q_mean_mx, q_std_mx, q_min_mx, q_max_mx);
    
    printf("Magnetometer Y (my):\n");
    printf("  Mean:     %.6f  Std Dev: %.6f  Min: %.6f  Max: %.6f\n\n", 
           q_mean_my, q_std_my, q_min_my, q_max_my);
    
    printf("Magnetometer Z (mz):\n");
    printf("  Mean:     %.6f  Std Dev: %.6f  Min: %.6f  Max: %.6f\n\n", 
           q_mean_mz, q_std_mz, q_min_mz, q_max_mz);
    
    // Write statistics to file
    char stats_path[PATH_MAX_LEN];
    join_path(stats_path, sizeof stats_path, g_drive, "imu_statistics.txt");
    FIL stats_file;
    FRESULT fr_stats = create_file(stats_path, &stats_file);
    
    if (fr_stats == FR_OK) {
        char stats_buf[512];
        int len;
        UINT bw;
        
        len = snprintf(stats_buf, sizeof stats_buf, 
            "========================================\n"
            "IMU Data Statistics\n"
            "========================================\n");
        write_to_file(&stats_file, stats_buf, len, &bw);
        
        // Write configuration settings
        len = snprintf(stats_buf, sizeof stats_buf,
            "\n--- Configuration Settings ---\n"
            "Configured Sampling Rate: %u Hz\n"
            "Max Samples: %u\n"
            "FFT Window Size: %u samples\n"
            "FFT Peaks Extracted: %u per axis\n"
            "Normalization Ranges:\n"
            "  - Accelerometer: ±%.1f g\n"
            "  - Gyroscope: ±%.1f dps\n"
            "  - Magnetometer: ±%.1f µT\n\n",
            SAMPLE_RATE_HZ, MAX_SAMPLES, N_FFT, FFT_PEAKS,
            ACCEL_RANGE, GYRO_RANGE, MAG_RANGE);
        write_to_file(&stats_file, stats_buf, len, &bw);
        
        // Write actual collection results
        float duration_s = (float)(last_timestamp - first_timestamp) / 1e6f;  // Convert µs to seconds
        float achieved_rate = (sample_count > 1) ? (float)(sample_count - 1) / duration_s : 0.0f;
        len = snprintf(stats_buf, sizeof stats_buf, 
            "--- Collection Results ---\n"
            "Total Samples Collected: %u\n"
            "Achieved Sampling Rate: %.2f Hz\n"
            "Actual Duration: %.3f seconds\n"
            "FFT Window Duration: %.3f seconds\n\n",
            sample_count, achieved_rate, duration_s, (float)N_FFT / SAMPLE_RATE_HZ);
        write_to_file(&stats_file, stats_buf, len, &bw);
        
        // Write RAW statistics
        len = snprintf(stats_buf, sizeof stats_buf,
            "NOTE: Raw values are in physical units:\n"
            "  - Accelerometer: g (1g = 9.81 m/s²)\n"
            "  - Gyroscope: dps (degrees per second)\n"
            "  - Magnetometer: µT (microtesla)\n"
            "  - Quantized values: normalized [-1, 1) range\n\n"
            "--- RAW Values (Physical Units) ---\n\n");
        write_to_file(&stats_file, stats_buf, len, &bw);
        
        len = snprintf(stats_buf, sizeof stats_buf,
            "Accelerometer X (ax):\n"
            "  Mean:     %.6f\n"
            "  Median:   %.6f\n"
            "  Variance: %.6f\n"
            "  Std Dev:  %.6f\n"
            "  Min:      %.6f\n"
            "  Max:      %.6f\n\n",
            raw_mean_ax, raw_med_ax, raw_var_ax, raw_std_ax, raw_min_ax, raw_max_ax);
        write_to_file(&stats_file, stats_buf, len, &bw);
        
        len = snprintf(stats_buf, sizeof stats_buf,
            "Accelerometer Y (ay):\n"
            "  Mean:     %.6f\n"
            "  Median:   %.6f\n"
            "  Variance: %.6f\n"
            "  Std Dev:  %.6f\n"
            "  Min:      %.6f\n"
            "  Max:      %.6f\n\n",
            raw_mean_ay, raw_med_ay, raw_var_ay, raw_std_ay, raw_min_ay, raw_max_ay);
        write_to_file(&stats_file, stats_buf, len, &bw);
        
        len = snprintf(stats_buf, sizeof stats_buf,
            "Accelerometer Z (az):\n"
            "  Mean:     %.6f\n"
            "  Median:   %.6f\n"
            "  Variance: %.6f\n"
            "  Std Dev:  %.6f\n"
            "  Min:      %.6f\n"
            "  Max:      %.6f\n\n",
            raw_mean_az, raw_med_az, raw_var_az, raw_std_az, raw_min_az, raw_max_az);
        write_to_file(&stats_file, stats_buf, len, &bw);
        
        len = snprintf(stats_buf, sizeof stats_buf,
            "Gyroscope (gx, gy, gz):\n"
            "  gx: Mean=%.3f Std=%.3f Min=%.3f Max=%.3f\n"
            "  gy: Mean=%.3f Std=%.3f Min=%.3f Max=%.3f\n"
            "  gz: Mean=%.3f Std=%.3f Min=%.3f Max=%.3f\n\n",
            raw_mean_gx, raw_std_gx, raw_min_gx, raw_max_gx,
            raw_mean_gy, raw_std_gy, raw_min_gy, raw_max_gy,
            raw_mean_gz, raw_std_gz, raw_min_gz, raw_max_gz);
        write_to_file(&stats_file, stats_buf, len, &bw);
        
        len = snprintf(stats_buf, sizeof stats_buf,
            "Magnetometer (mx, my, mz):\n"
            "  mx: Mean=%.3f Std=%.3f Min=%.3f Max=%.3f\n"
            "  my: Mean=%.3f Std=%.3f Min=%.3f Max=%.3f\n"
            "  mz: Mean=%.3f Std=%.3f Min=%.3f Max=%.3f\n\n",
            raw_mean_mx, raw_std_mx, raw_min_mx, raw_max_mx,
            raw_mean_my, raw_std_my, raw_min_my, raw_max_my,
            raw_mean_mz, raw_std_mz, raw_min_mz, raw_max_mz);
        write_to_file(&stats_file, stats_buf, len, &bw);
        
        // Write QUANTIZED statistics
        len = snprintf(stats_buf, sizeof stats_buf,
            "--- QUANTIZED Values (Normalized [-1, 1) range) ---\n\n");
        write_to_file(&stats_file, stats_buf, len, &bw);
        
        len = snprintf(stats_buf, sizeof stats_buf,
            "Accelerometer X (ax):\n"
            "  Mean:     %.6f\n"
            "  Median:   %.6f\n"
            "  Variance: %.6f\n"
            "  Std Dev:  %.6f\n"
            "  Min:      %.6f\n"
            "  Max:      %.6f\n\n",
            q_mean_ax, q_med_ax, q_var_ax, q_std_ax, q_min_ax, q_max_ax);
        write_to_file(&stats_file, stats_buf, len, &bw);
        
        len = snprintf(stats_buf, sizeof stats_buf,
            "Accelerometer Y (ay):\n"
            "  Mean:     %.6f\n"
            "  Median:   %.6f\n"
            "  Variance: %.6f\n"
            "  Std Dev:  %.6f\n"
            "  Min:      %.6f\n"
            "  Max:      %.6f\n\n",
            q_mean_ay, q_med_ay, q_var_ay, q_std_ay, q_min_ay, q_max_ay);
        write_to_file(&stats_file, stats_buf, len, &bw);
        
        len = snprintf(stats_buf, sizeof stats_buf,
            "Accelerometer Z (az):\n"
            "  Mean:     %.6f\n"
            "  Median:   %.6f\n"
            "  Variance: %.6f\n"
            "  Std Dev:  %.6f\n"
            "  Min:      %.6f\n"
            "  Max:      %.6f\n\n",
            q_mean_az, q_med_az, q_var_az, q_std_az, q_min_az, q_max_az);
        write_to_file(&stats_file, stats_buf, len, &bw);
        
        len = snprintf(stats_buf, sizeof stats_buf,
            "Gyroscope (quantized):\n"
            "  gx: Mean=%.6f Std=%.6f Min=%.6f Max=%.6f\n"
            "  gy: Mean=%.6f Std=%.6f Min=%.6f Max=%.6f\n"
            "  gz: Mean=%.6f Std=%.6f Min=%.6f Max=%.6f\n\n",
            q_mean_gx, q_std_gx, q_min_gx, q_max_gx,
            q_mean_gy, q_std_gy, q_min_gy, q_max_gy,
            q_mean_gz, q_std_gz, q_min_gz, q_max_gz);
        write_to_file(&stats_file, stats_buf, len, &bw);
        
        len = snprintf(stats_buf, sizeof stats_buf,
            "Magnetometer (quantized):\n"
            "  mx: Mean=%.6f Std=%.6f Min=%.6f Max=%.6f\n"
            "  my: Mean=%.6f Std=%.6f Min=%.6f Max=%.6f\n"
            "  mz: Mean=%.6f Std=%.6f Min=%.6f Max=%.6f\n\n",
            q_mean_mx, q_std_mx, q_min_mx, q_max_mx,
            q_mean_my, q_std_my, q_min_my, q_max_my,
            q_mean_mz, q_std_mz, q_min_mz, q_max_mz);
        write_to_file(&stats_file, stats_buf, len, &bw);
        
        // Write SNR and error analysis if computed
        if (snr_computed) {
            len = snprintf(stats_buf, sizeof stats_buf,
                "--- Quantization Quality Analysis ---\n"
                "Comparing normalized raw vs quantized values\n\n");
            write_to_file(&stats_file, stats_buf, len, &bw);
            
            len = snprintf(stats_buf, sizeof stats_buf,
                "Accelerometer:\n"
                "  ax: SNR=%.2f dB  Max Error=%.6f  RMS Error=%.6f\n"
                "  ay: SNR=%.2f dB  Max Error=%.6f  RMS Error=%.6f\n"
                "  az: SNR=%.2f dB  Max Error=%.6f  RMS Error=%.6f\n\n",
                snr_ax, max_err_ax, rms_err_ax,
                snr_ay, max_err_ay, rms_err_ay,
                snr_az, max_err_az, rms_err_az);
            write_to_file(&stats_file, stats_buf, len, &bw);
            
            len = snprintf(stats_buf, sizeof stats_buf,
                "Gyroscope:\n"
                "  gx: SNR=%.2f dB  Max Error=%.6f  RMS Error=%.6f\n"
                "  gy: SNR=%.2f dB  Max Error=%.6f  RMS Error=%.6f\n"
                "  gz: SNR=%.2f dB  Max Error=%.6f  RMS Error=%.6f\n\n",
                snr_gx, max_err_gx, rms_err_gx,
                snr_gy, max_err_gy, rms_err_gy,
                snr_gz, max_err_gz, rms_err_gz);
            write_to_file(&stats_file, stats_buf, len, &bw);
            
            len = snprintf(stats_buf, sizeof stats_buf,
                "Magnetometer:\n"
                "  mx: SNR=%.2f dB  Max Error=%.6f  RMS Error=%.6f\n"
                "  my: SNR=%.2f dB  Max Error=%.6f  RMS Error=%.6f\n"
                "  mz: SNR=%.2f dB  Max Error=%.6f  RMS Error=%.6f\n",
                snr_mx, max_err_mx, rms_err_mx,
                snr_my, max_err_my, rms_err_my,
                snr_mz, max_err_mz, rms_err_mz);
            write_to_file(&stats_file, stats_buf, len, &bw);
        }
        
        f_close(&stats_file);
        printf("Statistics written to %s\n", stats_path);
    } else {
        printf("Failed to create statistics file: %d\n", fr_stats);
    }
    
    // Clean up
    free(all_raw_ax);
    free(all_raw_ay);
    free(all_raw_az);
    free(all_raw_gx);
    free(all_raw_gy);
    free(all_raw_gz);
    free(all_raw_mx);
    free(all_raw_my);
    free(all_raw_mz);
    free(all_q_ax);
    free(all_q_ay);
    free(all_q_az);
    free(all_q_gx);
    free(all_q_gy);
    free(all_q_gz);
    free(all_q_mx);
    free(all_q_my);
    free(all_q_mz);

    f_close(&file);
    if (fr2 == FR_OK) f_close(&spec_file);
    f_unmount(g_drive);
    loop_forever_msg("Logging complete. Statistics saved.");
}

int main(void) {
    stdio_init_all();         // only here
    sleep_ms(3000);           
    
    printf("\n========================================\n");
    printf("IMU Data Logger - Starting Up\n");
    printf("========================================\n");
    printf("Configuration:\n");
    printf("  Sample Rate: %d Hz\n", SAMPLE_RATE_HZ);
    printf("  Max Samples: %d\n", MAX_SAMPLES);
    printf("  Duration:    %.1f seconds\n", (float)MAX_SAMPLES / (float)SAMPLE_RATE_HZ);
    printf("  FFT Size:    %d samples\n", N_FFT);
    printf("  FFT Window:  %.2f seconds\n", (float)N_FFT / (float)SAMPLE_RATE_HZ);
    printf("  Nyquist:     %.1f Hz\n", (float)SAMPLE_RATE_HZ / 2.0f);
    printf("========================================\n\n");
    
    // initialize the shared queue before launching core1
    queue_init(&sample_q, sizeof(Sample), QUEUE_DEPTH);

    // start core1 (SD writer + FFT)
    multicore_launch_core1(core1_writer); // core1 will own SD writes

    // run the sampler on core0 (this will block and push samples to the queue)
    core0_sampler();

    // unreachable
    while (true) { sleep_ms(1000); }
}