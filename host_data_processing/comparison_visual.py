"""
Visual Comparison: Original vs Fixed Analysis

=============================================================================
ORIGINAL (INCORRECT) ANALYSIS
=============================================================================

Raw Sensor → Physical Units → Normalization → Q15 Quantization
   (MCU)         (MCU)           (MCU)            (MCU)
                   ↓                                ↓
              f32 = 0.001953 g              q15 = 32 (int16)
                   ↓                                ↓
                   |                         q15/32768 = 0.0009765
                   |                                ↓
                   └────────── ERROR ───────────────┘
                              (0.001953 - 0.0009765)
                              
Problem: Comparing PHYSICAL UNITS vs NORMALIZED values!
Result: Large artificial "error" (0.000977 in this case)

=============================================================================
FIXED (CORRECT) ANALYSIS  
=============================================================================

Raw Sensor → Physical Units → Normalization → Q15 Quantization
   (MCU)         (MCU)           (MCU)            (MCU)
                   ↓                                ↓
              f32 = 0.001953 g              q15 = 32 (int16)
                   ↓                                ↓
              f32/RANGE = 0.0009765         q15/32768 = 0.0009765
               (normalized)                   (normalized)
                   ↓                                ↓
                   └────────── ERROR ───────────────┘
                              (0.0009765 - 0.0009765)
                              
Solution: Both values normalized to [-1, 1) range!
Result: True quantization error (≈ 0 in this case, < Δ/2 in general)

=============================================================================
KEY INSIGHT
=============================================================================

Q15 quantization operates in NORMALIZED space [-1, 1), not physical units.
To measure quantization error, you must compare values in the SAME scale.

Correct comparison options:
  1. Normalize f32, then compare: (f32/RANGE) - (q15/32768)  ✅
  2. Denormalize q15, then compare: f32 - (q15/32768 * RANGE)  ✅

Incorrect comparison:
  3. Compare different scales: f32 - (q15/32768)  ❌

=============================================================================
"""

print(__doc__)
