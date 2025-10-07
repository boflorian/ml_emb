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


#define PATH_MAX_LEN 256
#define QUEUE_DEPTH 1024
#define N_FFT 256
#define FFT_PEAKS 3
#define SAMPLE_RATE_HZ 100.0f // approximate (sleep_ms(10) in sampler)
#define MAX_SAMPLES 1000 // limit for data collection

// ICM-20948 sensor normalization constants
// These convert raw int16 values to normalized [-1, 1) range for Q15 quantization
// Accelerometer: ±2g range, 16384 LSB/g → normalize by dividing by max expected g (e.g., 2g)
#define ACCEL_SCALE (1.0f / 16384.0f)  // Converts LSB to g, then divide by max range
#define ACCEL_RANGE 2.0f                // ±2g, so full range is 4g peak-to-peak
// Gyroscope: ±250 dps range, 131 LSB/dps → normalize by dividing by max expected dps
#define GYRO_SCALE (1.0f / 131.0f)     // Converts LSB to dps
#define GYRO_RANGE 250.0f               // ±250 dps, so full range is 500 dps peak-to-peak
// Magnetometer: ±4900 µT range, 0.15 µT/LSB
#define MAG_SCALE 0.15f                 // Converts LSB to µT
#define MAG_RANGE 4900.0f               // ±4900 µT


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

    g_drive = sd_get_drive_prefix(g_sd);  // usually "0:"
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
    printf("Creating file...\n");
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
    float roll, pitch, yaw;
    uint32_t t_us;
} Sample;

static queue_t sample_q;

static void core0_sampler(void)
{
    IMU_EN_SENSOR_TYPE type;
    imuInit(&type);
    uint32_t t_prev = (uint32_t)time_us_64();
    while (1) {
        IMU_ST_SENSOR_DATA stGyroRawData, stAccelRawData, stMagnRawData;
        IMU_ST_ANGLES_DATA stAngles;
        imuDataGet(&stAngles, &stGyroRawData, &stAccelRawData, &stMagnRawData);
        uint32_t t_now = (uint32_t)time_us_64();
        Sample s = { stAccelRawData.s16X, stAccelRawData.s16Y, stAccelRawData.s16Z,
                      stGyroRawData.s16X, stGyroRawData.s16Y, stGyroRawData.s16Z,
                      stMagnRawData.s16X, stMagnRawData.s16Y, stMagnRawData.s16Z,
                      stAngles.fRoll, stAngles.fPitch, stAngles.fYaw,
                      t_now };
        queue_add_blocking(&sample_q, &s);
        // pace sampling to ~SAMPLE_RATE_HZ to match previous behavior
        sleep_ms(10);
    }
}

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
    sleep_ms(1500); // optional

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
    // *_raw: raw int16 sensor values (LSB)
    // *_f32: physical units (ax/ay/az in g, gx/gy/gz in dps, mx/my/mz in µT)
    // *_q15: Q15 quantized values (normalized to [-1,1) then quantized to int16)
    // roll/pitch/yaw: orientation angles in degrees
    const char *header = "timestamp_us,ax_raw,ay_raw,az_raw,gx_raw,gy_raw,gz_raw,mx_raw,my_raw,mz_raw,ax_f32,ay_f32,az_f32,gx_f32,gy_f32,gz_f32,mx_f32,my_f32,mz_f32,ax_q15,ay_q15,az_q15,gx_q15,gy_q15,gz_q15,mx_q15,my_q15,mz_q15,roll,pitch,yaw\n";
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
    // Angles
    float *all_roll = (float*)malloc(MAX_SAMPLES * sizeof(float));
    float *all_pitch = (float*)malloc(MAX_SAMPLES * sizeof(float));
    float *all_yaw = (float*)malloc(MAX_SAMPLES * sizeof(float));
    
    if (!all_raw_ax || !all_raw_ay || !all_raw_az || !all_q_ax || !all_q_ay || !all_q_az ||
        !all_raw_gx || !all_raw_gy || !all_raw_gz || !all_q_gx || !all_q_gy || !all_q_gz ||
        !all_raw_mx || !all_raw_my || !all_raw_mz || !all_q_mx || !all_q_my || !all_q_mz ||
        !all_roll || !all_pitch || !all_yaw) {
        printf("Failed to allocate memory for statistics\n");
        f_close(&file);
        if (fr2 == FR_OK) f_close(&spec_file);
        loop_forever_msg("Memory allocation failed");
    }

    // Dequeue samples, write CSV, accumulate buffers and run FFT when full
    while (sample_count < MAX_SAMPLES) {
        Sample s;
        queue_remove_blocking(&sample_q, &s);

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
        
        // Write CSV line with all three formats: raw int16, float32, quantized int16
        char line[512];
        int len = snprintf(line, sizeof line, 
            "%u,"                                           // timestamp
            "%d,%d,%d,%d,%d,%d,%d,%d,%d,"                  // raw int16 (9 values)
            "%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f," // float32 (9 values)
            "%d,%d,%d,%d,%d,%d,%d,%d,%d,"                  // quantized int16 (9 values)
            "%.3f,%.3f,%.3f\n",                            // angles (3 values)
            s.t_us,
            s.ax, s.ay, s.az, s.gx, s.gy, s.gz, s.mx, s.my, s.mz,
            f32_ax, f32_ay, f32_az, f32_gx, f32_gy, f32_gz, f32_mx, f32_my, f32_mz,
            q15_vals[0], q15_vals[1], q15_vals[2], q15_vals[3], q15_vals[4], 
            q15_vals[5], q15_vals[6], q15_vals[7], q15_vals[8],
            s.roll, s.pitch, s.yaw);
        
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
        all_roll[sample_count] = s.roll;
        all_pitch[sample_count] = s.pitch;
        all_yaw[sample_count] = s.yaw;
        
        // Dequantize Q15 back to float for statistics
        float dequant_vals[9];
        dequantize_q15(q15_vals, 9, dequant_vals);
        
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

        // Accumulate normalized values for FFT (same range as will be quantized)
        buf_ax[buf_pos] = norm_ax;
        buf_ay[buf_pos] = norm_ay;
        buf_az[buf_pos] = norm_az;
        buf_gx[buf_pos] = norm_gx;
        buf_gy[buf_pos] = norm_gy;
        buf_gz[buf_pos] = norm_gz;
        buf_mx[buf_pos] = norm_mx;
        buf_my[buf_pos] = norm_my;
        buf_mz[buf_pos] = norm_mz;
        buf_pos++;

        if (buf_pos >= N_FFT) {
            // ---- Quantize & dequantize for FFT ----
            int16_t q15_ax[N_FFT], q15_ay[N_FFT], q15_az[N_FFT];
            int16_t q15_gx[N_FFT], q15_gy[N_FFT], q15_gz[N_FFT];
            int16_t q15_mx[N_FFT], q15_my[N_FFT], q15_mz[N_FFT];
            int clips_ax = 0, clips_ay = 0, clips_az = 0;
            int clips_gx = 0, clips_gy = 0, clips_gz = 0;
            int clips_mx = 0, clips_my = 0, clips_mz = 0;

            quantize_q15(buf_ax, N_FFT, q15_ax, &clips_ax);
            quantize_q15(buf_ay, N_FFT, q15_ay, &clips_ay);
            quantize_q15(buf_az, N_FFT, q15_az, &clips_az);
            quantize_q15(buf_gx, N_FFT, q15_gx, &clips_gx);
            quantize_q15(buf_gy, N_FFT, q15_gy, &clips_gy);
            quantize_q15(buf_gz, N_FFT, q15_gz, &clips_gz);
            quantize_q15(buf_mx, N_FFT, q15_mx, &clips_mx);
            quantize_q15(buf_my, N_FFT, q15_my, &clips_my);
            quantize_q15(buf_mz, N_FFT, q15_mz, &clips_mz);

            dequantize_q15(q15_ax, N_FFT, buf_ax);
            dequantize_q15(q15_ay, N_FFT, buf_ay);
            dequantize_q15(q15_az, N_FFT, buf_az);
            dequantize_q15(q15_gx, N_FFT, buf_gx);
            dequantize_q15(q15_gy, N_FFT, buf_gy);
            dequantize_q15(q15_gz, N_FFT, buf_gz);
            dequantize_q15(q15_mx, N_FFT, buf_mx);
            dequantize_q15(q15_my, N_FFT, buf_my);
            dequantize_q15(q15_mz, N_FFT, buf_mz);

            printf("Quantization clips: ax=%d, ay=%d, az=%d, gx=%d, gy=%d, gz=%d, mx=%d, my=%d, mz=%d\n", 
                   clips_ax, clips_ay, clips_az, clips_gx, clips_gy, clips_gz, clips_mx, clips_my, clips_mz);

            // process each axis sequentially (all 9 channels)
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
                            float freq = ((float)idx[p]) * (SAMPLE_RATE_HZ / (float)N_FFT);
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
    
    // Angles statistics
    float mean_roll = mean_f32(all_roll, sample_count);
    float mean_pitch = mean_f32(all_pitch, sample_count);
    float mean_yaw = mean_f32(all_yaw, sample_count);
    float std_roll = stddev_f32(variance_f32(all_roll, sample_count, mean_roll));
    float std_pitch = stddev_f32(variance_f32(all_pitch, sample_count, mean_pitch));
    float std_yaw = stddev_f32(variance_f32(all_yaw, sample_count, mean_yaw));
    float min_roll, max_roll, min_pitch, max_pitch, min_yaw, max_yaw;
    min_max_f32(all_roll, sample_count, &min_roll, &max_roll);
    min_max_f32(all_pitch, sample_count, &min_pitch, &max_pitch);
    min_max_f32(all_yaw, sample_count, &min_yaw, &max_yaw);
    
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
        
        len = snprintf(stats_buf, sizeof stats_buf, 
            "Total samples: %u\n\n", sample_count);
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
        
        len = snprintf(stats_buf, sizeof stats_buf,
            "Orientation Angles:\n"
            "  Roll:  Mean=%.3f° Std=%.3f° Min=%.3f° Max=%.3f°\n"
            "  Pitch: Mean=%.3f° Std=%.3f° Min=%.3f° Max=%.3f°\n"
            "  Yaw:   Mean=%.3f° Std=%.3f° Min=%.3f° Max=%.3f°\n\n",
            mean_roll, std_roll, min_roll, max_roll,
            mean_pitch, std_pitch, min_pitch, max_pitch,
            mean_yaw, std_yaw, min_yaw, max_yaw);
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
            "  mz: Mean=%.6f Std=%.6f Min=%.6f Max=%.6f\n",
            q_mean_mx, q_std_mx, q_min_mx, q_max_mx,
            q_mean_my, q_std_my, q_min_my, q_max_my,
            q_mean_mz, q_std_mz, q_min_mz, q_max_mz);
        write_to_file(&stats_file, stats_buf, len, &bw);
        
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
    free(all_roll);
    free(all_pitch);
    free(all_yaw);

    f_close(&file);
    if (fr2 == FR_OK) f_close(&spec_file);
    f_unmount(g_drive);
    loop_forever_msg("Logging complete. Statistics saved.");
}

int main(void) {
    stdio_init_all();         // only here
    sleep_ms(2000);           // optional USB settle time
    printf("IMU Logger starting...\n");
    // initialize the shared queue before launching core1
    queue_init(&sample_q, sizeof(Sample), QUEUE_DEPTH);

    // start core1 (SD writer + FFT)
    multicore_launch_core1(core1_writer); // core1 will own SD writes

    // run the sampler on core0 (this will block and push samples to the queue)
    core0_sampler();

    // unreachable
    while (true) { sleep_ms(1000); }
}