#!/bin/bash

INPUT_FILE="output_gemini.txt"
OUTPUT_CSV="my_reports.csv"

echo "Creating CheXbert-compatible CSV..."

# Create CSV with "Report Impression" column header
{
    echo "Report Impression"
    cat "$INPUT_FILE"
} > "$OUTPUT_CSV"

echo "✓ Created $OUTPUT_CSV"
echo ""
echo "Preview:"
head -5 "$OUTPUT_CSV"
