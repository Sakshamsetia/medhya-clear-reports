#!/bin/bash
set -e

# FORCE use of /opt/conda (where all environments actually exist)
CONDA_CMD="/opt/conda/bin/conda"

# Make sure /opt/conda is used
export PATH="/opt/conda/bin:$PATH"
eval "$($CONDA_CMD shell.bash hook)" 2>/dev/null

# Configure environments here
LLAVARAD_ENV="llavarad"
RADIALOG_ENV="radialog"
CHESTX_ENV="dpproject"

# Output file with timestamp
OUTPUT_FILE="/home/chipmonkx86/rad_report.txt"

echo "=========================================="
echo "Sequential Medical AI Model Pipeline"
echo "User: DonQuixote248"
echo "Date: $(date -u +'%Y-%m-%d %H:%M:%S')"
echo "=========================================="
echo ""
echo "Running models silently..."
echo "Output will be saved to: $OUTPUT_FILE"
echo ""

# Initialize output file
cat > "$OUTPUT_FILE" << 'EOF'
========================================
RADIOLOGY REPORTS
Generated: $(date -u +'%Y-%m-%d %H:%M:%S UTC')
========================================

EOF

# Model 1: LlavaRad
echo "[1/3] Running RadLLM..."
cd /home/chipmonkx86/llava-rad || exit 1

{
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "MODEL 1: LlavaRad Report"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""
    $CONDA_CMD run -n "$LLAVARAD_ENV" sudo python3 inference.py 2>/dev/null | grep -v "^Loading\|^Downloading\|^Using\|^\[INFO\]\|^100%\|^Fetching\|torch\|GPU\|CUDA\|model\|checkpoint" || $CONDA_CMD run -n "$LLAVARAD_ENV" sudo python3 inference.py 2>/dev/null
    echo ""
    echo ""
} >> "$OUTPUT_FILE"

echo "✓ RadLLM completed"

# Model 2: RaDialog
echo "[2/3] Running RaDialog..."
cd /home/chipmonkx86/RaDialog || exit 1

{
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "MODEL 2: RaDialog Report"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""
    $CONDA_CMD run -n "$RADIALOG_ENV" sudo python3 simple_inference.py 2>/dev/null | grep -v "^Loading\|^Downloading\|^Using\|^\[INFO\]\|^100%\|^Fetching\|torch\|GPU\|CUDA\|model\|checkpoint" || $CONDA_CMD run -n "$RADIALOG_ENV" sudo python3 simple_inference.py 2>/dev/null
    echo ""
    echo ""
} >> "$OUTPUT_FILE"

echo "✓ RaDialog completed"

# Model 3: ChestX
echo "[3/3] Running ChestX..."
cd /home/samarthsoni040906/ChestX || exit 1

{
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "MODEL 3: ChestX Report"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""
    $CONDA_CMD run -n "$CHESTX_ENV" sudo python3 inference.py 2>/dev/null | grep -v "^Loading\|^Downloading\|^Using\|^\[INFO\]\|^100%\|^Fetching\|torch\|GPU\|CUDA\|model\|checkpoint" || $CONDA_CMD run -n "$CHESTX_ENV" sudo python3 inference.py 2>/dev/null
    echo ""
} >> "$OUTPUT_FILE"

echo "✓ ChestX completed"
echo ""

echo "=========================================="
echo "✓ All 3 models completed!"
echo "=========================================="
echo ""
echo "Reports saved to: $OUTPUT_FILE"
echo ""
echo "View reports:"
echo "  cat $OUTPUT_FILE"
echo ""

# Display the clean output
cat "$OUTPUT_FILE"
