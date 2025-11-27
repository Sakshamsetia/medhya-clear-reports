#!/bin/bash
set -e

# Configure environments here
LLAVARAD_ENV="llavarad"


RADIALOG_ENV="radialog"
CHESTX_ENV="dpproject"  # Change this to match your ChestX environment

echo "=========================================="
echo "Sequential Medical AI Model Pipeline"
echo "User: DonQuixote248"
echo "Date: $(date -u +'%Y-%m-%d %H:%M:%S UTC')"
echo "=========================================="
echo ""

# Model 1: LlavaRad
echo ">>> [1/3] Running LlavaRad model..."
echo "Environment: $LLAVARAD_ENV"
cd /home/chipmonkx86/llava-rad || { echo "Error: llava-rad directory not found"; exit 1; }
conda run -n "$LLAVARAD_ENV" sudo python3 inference.py
echo "✓ LlavaRad completed"
echo ""

# Model 2: RaDialog  
echo ">>> [2/3] Running RaDialog model..."
echo "Environment: $RADIALOG_ENV"
cd /home/chipmonkx86/RaDialog || { echo "Error: RaDialog directory not found"; exit 1; }
conda run -n "$RADIALOG_ENV" sudo python3 simple_inference.py
echo "✓ RaDialog completed"
echo ""

# Model 3: ChestX
echo ">>> [3/3] Running ChestX model..."
echo "Environment: $CHESTX_ENV"
cd /home/samarthsoni040906/ChestX || { echo "Error: ChestX directory not found"; exit 1; }
conda run -n "$CHESTX_ENV" sudo python3 inference.py
echo "✓ ChestX completed"
echo ""

echo "=========================================="
echo "✓ All 3 models completed successfully!"
echo "=========================================="
