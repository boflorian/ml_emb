# ML for Embedded Systems — Gesture Recognition

Repository for the project in *Machine Learning for Embedded Systems* at **TalTech**.

End-to-end gesture recognition using a Raspberry Pi Pico W and an ICM20948 IMU sensor. Gestures are classified in real-time on-device using TensorFlow Lite Micro and results are sent to a Flask server over WiFi.

---

## Project Structure

```
ml_emb/
├── imu_data_logger/              # Pico firmware: standalone IMU data logging to SD card
├── dataset_collection_application/  # Pico firmware: gesture dataset collection
├── host_data_processing/         # Visualization and analysis of collected IMU data
├── data_preprocessing/           # C signal processing utilities (FFT, stats, quantization)
├── gesture_recog/                # ML training pipeline (Python)
│   ├── dataset_magic_wand/       # Reference dataset (4 classes, 8 participants)
│   ├── dataset_pico_gestures/    # Custom dataset from Pico hardware
│   ├── model_definitions/        # CNN, BiLSTM, 1D-CNN model architectures
│   ├── util/                     # Data loading, augmentation, feature extraction
│   ├── gan/                      # GAN-based data augmentation
│   ├── deployment_models/        # Converted TFLite models ready for Pico
│   ├── main.py                   # Training script
│   └── convert.py                # Keras → TFLite conversion + quantization
├── deployment/                   # Pico firmware: real-time inference + WiFi reporting
│   ├── src/                      # Inference pipeline (TFLite Micro)
│   ├── drivers/                  # LCD, LED, touch drivers
│   └── lib/pico-tflmicro/        # TensorFlow Lite Micro library
├── server/                       # Flask backend: receives gesture events from Pico
└── nn_python/                    # Separate utility ML project
```

---

## Hardware

- **MCU**: Raspberry Pi Pico W (RP2040, 264 KB RAM, 2 MB Flash)
- **IMU**: ICM20948 (accelerometer, gyroscope, magnetometer) via I2C
- **Storage**: SD card (data collection)
- **Display**: LCD with touch input (deployment)
- **LED**: WS2812 RGB

---

## Pipeline

```
IMU Data Collection (Pico → SD card CSV)
        ↓
Data Preprocessing (filter, normalize, window)
        ↓
Model Training (TensorFlow — CNN / BiLSTM)
        ↓
Convert to TFLite + Quantize
        ↓
Deploy on Pico W (TFLite Micro inference)
        ↓
Send Results via HTTP POST → Flask Server
```

### Gesture Classes
- `negative` (idle/no gesture)
- `ring`
- `slope`
- `wave`

### Model Input
- 64–286 accelerometer samples (ax, ay, az) per window
- Low-pass filtered, normalized to [−1, 1]

---

## Getting Started

### 1. IMU Data Collection

Flash `imu_data_logger/main.cpp` onto the Pico to log raw IMU data to SD card:

```bash
cd imu_data_logger
mkdir build && cd build && cmake .. && make
```

Sampling rate and duration are configured at the top of `main.cpp`.

**Outputs:**
- `imu_log.csv` — raw IMU samples
- `imu_spectra.csv` — frequency-domain features
- `imu_statistics.txt` — statistical summary

### 2. Gesture Dataset Collection

Flash `dataset_collection_application/lab_0_1.cpp` for labelled gesture recording. LED colours indicate state: red = startup, green = ready, blue = buffer overflow warning.

### 3. Model Training

```bash
cd gesture_recog
python main.py
```

Trains CNN, BiLSTM, and 1D-CNN models. Hyperparameters are configured at the top of `main.py`.

### 4. Convert to TFLite

```bash
python convert.py
```

Converts the best Keras model to TFLite with post-training quantization.

### 5. Deploy on Pico W

Update `deployment/wifi_config.h` with your WiFi credentials and server address, then build:

```bash
cd deployment
mkdir build && cd build && cmake .. && make
```

### 6. Run the Server

```bash
cd server
python server.py
```

Flask API listens on `0.0.0.0:8000`. Gesture events are received at `POST /api/gesture_event`.

---

## Tech Stack

| Layer | Technology |
|---|---|
| Embedded firmware | C++ (Pico SDK), CMake |
| On-device ML | TensorFlow Lite Micro |
| Model training | TensorFlow 2 / Keras, Python |
| Data processing | NumPy, SciPy, Pandas |
| Server | Flask |
| Data augmentation | GAN (Keras) |

---

## Notes

- See `host_data_processing/TRAJECTORY_FIX.md` for important notes on IMU unit conversion (raw LSB vs. physical units).
- See `gesture_recog/model_definitions/agent.md` for CNN architecture details and hyperparameter tuning notes.
- See `server/agent.md` for the API contract.
