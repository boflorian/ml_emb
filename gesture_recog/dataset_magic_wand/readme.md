# Magic Wand Dataset for MCU Machine Learning

## Overview

This dataset contains accelerometer data collected from magic wand gestures performed by multiple participants. The data is designed for training machine learning models that can classify different wand movements, specifically targeting deployment on microcontroller units (MCU) such as the Raspberry Pi Pico.

## Dataset Structure

The dataset is organized into four main categories representing different gesture types:

```
dataset/
├── negative/     # Control movements (non-gesture data)
├── ring/         # Circular/ring-shaped gestures
├── slope/        # Linear/sloped movements
├── wing/         # Wing-like movements
└── readme.md     # This file
```

Each category contains 8 files (`person0.txt` to `person7.txt`), representing data collected from 8 different participants.

## Data Format

### File Structure
Each `.txt` file contains multiple samples of accelerometer data. The structure follows this pattern:

```
sample1
ax,ay,az
-0.67,12.52,-7.83
-0.94,8.39,-8.59
-1.22,10.48,-8.47
...

sample2
ax,ay,az
0.27,8.34,-6.96
-0.99,7.92,-6.63
-0.50,8.80,-6.76
...
```

### Data Description
- **Sample Header**: Each gesture sample starts with `sampleN` where N is the sample number (1, 2, 3, ...)
- **Column Header**: `ax,ay,az` indicates the three accelerometer axes
- **Data Points**: Comma-separated values representing:
  - `ax`: Acceleration in X-axis (g-force units)
  - `ay`: Acceleration in Y-axis (g-force units) 
  - `az`: Acceleration in Z-axis (g-force units)

### Sampling Characteristics
- **Sampling Rate**: Variable length sequences per gesture
- **Range**: Accelerometer values typically range from -80 to +80 g-force units
- **Precision**: Values recorded with 2 decimal places

## Data Parsing Guidelines

### Python Example
```python
def parse_gesture_file(filepath):
    """
    Parse a gesture file and return samples with their accelerometer data.
    
    Returns:
        List of dictionaries, each containing:
        - 'sample_id': Sample identifier
        - 'data': List of [ax, ay, az] accelerometer readings
    """
    samples = []
    current_sample = None
    
    with open(filepath, 'r') as file:
        for line in file:
            line = line.strip()
            
            if line.startswith('sample'):
                # New sample detected
                if current_sample:
                    samples.append(current_sample)
                current_sample = {
                    'sample_id': line,
                    'data': []
                }
            elif line == 'ax,ay,az':
                # Skip header line
                continue
            elif line and ',' in line:
                # Parse accelerometer data
                try:
                    ax, ay, az = map(float, line.split(','))
                    current_sample['data'].append([ax, ay, az])
                except ValueError:
                    continue
        
        # Add the last sample
        if current_sample:
            samples.append(current_sample)
    
    return samples
```

## Suggested Preprocessing Steps
1. **Normalization**: Scale accelerometer values to [-1, 1] range
2. **Filtering**: Apply low-pass filter to reduce noise
3. **Segmentation**: Create fixed-length windows (e.g., 64-128 samples)
4. **Feature Extraction**: Consider statistical features (mean, std, min, max) if using traditional ML

## Dataset Statistics

| Category | Files | Avg Samples/File | Total Samples |
|----------|-------|------------------|---------------|
| negative | 8     | ~10-26          | ~150          |
| ring     | 8     | ~5-25           | ~120          |
| slope    | 8     | ~5-25           | ~120          |
| wing     | 8     | ~5-25           | ~120          |

## Usage Notes

- **Training Split**: Recommend using 6-7 participants for training, 1-2 for testing
- **Cross-Validation**: Consider person-independent validation for better generalization
- **Data Augmentation**: May be needed due to limited samples per gesture type
- **Baseline Model**: Start with simple models (SVM, Random Forest) before deep learning approaches