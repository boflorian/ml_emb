#include <stdio.h>
#include <string.h>
#include "pico/stdlib.h"
#include "pico/multicore.h"
#include "pico/util/queue.h"
#include "pico/time.h"

#include "ff.h"
#include "sd_card.h"
#include "f_util.h"
#include "hw_config.h"

// For IMU Logging 
# include "icm20948.h"
#include "pico/util/queue.h"


#define PATH_MAX_LEN 256


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
    printf("In create file func\n");
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
    uint32_t t_us;
} Sample;

static queue_t sample_q;

static void core1_reader(void)
{
    IMU_EN_SENSOR_TYPE type;
    imuInit(&type);
    uint32_t t_prev = (uint32_t)time_us_64();
    while (1) {
        IMU_ST_SENSOR_DATA stGyroRawData, stAccelRawData;
        imuDataAccGyrGet(&stGyroRawData, &stAccelRawData);
        uint32_t t_now = (uint32_t)time_us_64();
        Sample s = { stAccelRawData.s16X, stAccelRawData.s16Y, stAccelRawData.s16Z,
                      stGyroRawData.s16X, stGyroRawData.s16Y, stGyroRawData.s16Z,
                      t_now };
        queue_add_blocking(&sample_q, &s);
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

void core1(void) {
    sleep_ms(1500); // optional

    if (!sd_init_and_mount()) {
        loop_forever_msg("SD init failed");
    }

    char log_path[PATH_MAX_LEN];
    join_path(log_path, sizeof log_path, g_drive, "imu_log.csv");

    FIL file;
    FRESULT fr = create_file(log_path, &file);
    if (fr != FR_OK) die(fr, "f_open");

    UINT written = 0;
    const char *header = "timestamp_us,ax,ay,az,gx,gy,gz\n";
    write_to_file(&file, header, strlen(header), &written);

    IMU_EN_SENSOR_TYPE imu_type;
    imuInit(&imu_type);
    if (imu_type != IMU_EN_SENSOR_TYPE_ICM20948) {
        printf("IMU not detected\n");
        f_close(&file);
        loop_forever_msg("No IMU");
    }

    while (true) {
        IMU_ST_SENSOR_DATA gyro, accel;
        imuDataAccGyrGet(&gyro, &accel);

        uint32_t t_now = (uint32_t)time_us_64();
        char line[96];
        int len = snprintf(line, sizeof line,
                           "%u,%d,%d,%d,%d,%d,%d\n",
                           t_now,
                           accel.s16X, accel.s16Y, accel.s16Z,
                           gyro.s16X, gyro.s16Y, gyro.s16Z);

        if (len > 0 && len < (int)sizeof line) {
            fr = write_to_file(&file, line, (UINT)len, &written);
            if (fr != FR_OK || written != (UINT)len) {
                printf("IMU log write failed: %d\n", fr);
                break;
            }
        }
        sleep_ms(10); // adjust sampling rate as required
    }

    f_close(&file);
    f_unmount(g_drive);
    loop_forever_msg("Logging halted");
}

int main(void) {
    stdio_init_all();         // only here
    sleep_ms(2000);           // optional USB settle time
    printf("IMU Logger starting...\n");
    multicore_launch_core1(core1); // only core1 can write to sd card 
    while (true) { sleep_ms(1000); }
}