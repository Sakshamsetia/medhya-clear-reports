#!/bin/bash
set -e

# FORCE use of /opt/conda (where all environments actually exist)
# Don't use user's personal conda
CONDA_CMD="/opt/conda/bin/conda"

# Make sure /opt/conda is used
export PATH="/opt/conda/bin:$PATH"
eval "$($CONDA_CMD shell.bash hook)"

# Configure environments here
LLAVARAD_ENV="llavarad"
RADIALOG_ENV="radialog"
CHESTX_ENV="dpproject"

echo "=========================================="
echo "Sequential Medical AI Model Pipeline"
echo "User: DonQuixote248"
echo "Date: $(date -u +'%Y-%m-%d %H:%M:%S UTC')"
echo "=========================================="
echo ""

echo "Using conda: $CONDA_CMD"
echo ""
echo ">>> Verifying environments exist in /opt/conda..."
$CONDA_CMD env list | grep -E "llavarad|radialog|dpproject"
echo ""

# Model 1: LlavaRad
echo ">>> [1/3] Running LlavaRad model..."
echo "Environment: $LLAVARAD_ENV"
cd /home/chipmonkx86/llava-rad || { echo "Error: llava-rad directory not found"; exit 1; }
$CONDA_CMD run -n "$LLAVARAD_ENV" sudo python3 inference.py
echo "✓ LlavaRad completed"
echo ""

# Model 2: RaDialog  
echo ">>> [2/3] Running RaDialog model..."
echo "Environment: $RADIALOG_ENV"
cd /home/chipmonkx86/RaDialog || { echo "Error: RaDialog directory not found"; exit 1; }
$CONDA_CMD run -n "$RADIALOG_ENV" sudo python3 simple_inference.py
echo "✓ RaDialog completed"
echo ""

# Model 3: ChestX
echo ">>> [3/3] Running ChestX model..."
echo "Environment: $CHESTX_ENV"
cd /home/samarthsoni040906/ChestX || { echo "Error: ChestX directory not found"; exit 1; }
$CONDA_CMD run -n "$CHESTX_ENV" sudo python3 inference.py
echo "✓ ChestX completed"
echo ""

echo "=========================================="
echo "✓ All 3 models completed successfully!"
echo "=========================================="
