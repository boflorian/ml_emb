#!/bin/bash
# Quick test script to verify quantization analysis

echo "=================================="
echo "IMU Quantization Analysis Test"
echo "=================================="
echo ""

# Check if CSV file exists
if [ ! -f "imu_log.csv" ]; then
    echo "❌ ERROR: imu_log.csv not found!"
    echo "   Please copy your CSV file here or update the filename in quantization_analysis.py"
    exit 1
fi

echo "✓ Found imu_log.csv"
echo ""

# Check Python dependencies
python3 -c "import pandas, numpy, matplotlib" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "❌ ERROR: Missing Python dependencies"
    echo "   Install with: pip install pandas numpy matplotlib"
    exit 1
fi

echo "✓ Python dependencies installed"
echo ""

# Show CSV structure
echo "CSV Preview (first 3 lines):"
echo "----------------------------"
head -n 3 imu_log.csv
echo ""

# Run analysis
echo "Running quantization analysis..."
echo "=================================="
python3 quantization_analysis.py

echo ""
echo "✓ Analysis complete!"
