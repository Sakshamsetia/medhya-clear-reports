#!/bin/bash

# ============================================================
# Gemini 2.5 Pro - Analyze File and Save Output
# Usage: bash gemini_analyze_save.sh <file> [prompt]
# ============================================================

API_KEY="AIzaSyATMvDqhAC044jvXOP9f3ZcSFJ3PHS-Xpw"
MODEL="gemini-2.5-pro"
API_URL="https://generativelanguage.googleapis.com/v1/models/${MODEL}:generateContent?key=${API_KEY}"
OUTPUT_FILE="output_gemini.txt"

# Check if file provided
if [ -z "$1" ]; then
    echo "Usage: bash gemini_analyze_save.sh <file> [prompt]"
    echo ""
    echo "Examples:"
    echo "  bash gemini_analyze_save.sh report.txt"
    echo "  bash gemini_analyze_save.sh report.txt 'Summarize the key findings'"
    echo "  bash gemini_analyze_save.sh report.txt 'Compare the three reports and highlight differences'"
    exit 1
fi

INPUT_FILE="$1"
CUSTOM_PROMPT="The file contains a lot of unnecessary text mixed with required radiology report outputs from 3 different models.
Your tasks:
1. Extract ONLY the 3 clean model outputs from the file, with no extra text. Output them first, separated by a blank newline between each of the 3 outputs.
2. After listing the 3 outputs, create a single unified FINAL REPORT. This final report should be an ensemble-style synthesis that merges the three outputs into one coherent, clinically-sound radiology impression. 
   - Do NOT compare them.
   - Do NOT highlight differences.
   - Simply integrate all consistent findings into one comprehensive conclusion.
   - The final report should be long, detailed, and structured (at least 15–25 lines), written like a professional radiology report.
Format your response exactly like this:

=== OUTPUTS ===
<Model Output 1>

<Model Output 2>

<Model Output 3>

=== FINAL ENSEMBLE REPORT ===
<Combined radiology report here>

Only follow this output format.}"
# Check if input file exists
if [ ! -f "$INPUT_FILE" ]; then
    echo "Error: File '$INPUT_FILE' not found"
    exit 1
fi

# Check if jq is installed
if ! command -v jq &> /dev/null; then
    echo "Error: jq is required. Install with: sudo apt install jq"
    exit 1
fi

echo "=========================================="
echo "Chexbert Ensembling"
echo "Date: $(date -u +'%Y-%m-%d %H:%M:%S UTC')"
echo "User: DonQuixote248"
echo "=========================================="
echo ""
echo "Input file: $INPUT_FILE"
echo "Prompt: $CUSTOM_PROMPT"
echo "Output will be saved to: $OUTPUT_FILE"
echo ""
echo "Sending to Chexbert"
echo ""

# Read file content
FILE_CONTENT=$(cat "$INPUT_FILE")

# Create full prompt
FULL_PROMPT="${CUSTOM_PROMPT}

File content:
${FILE_CONTENT}"

# Send to Gemini and save response
RESPONSE=$(curl -s -X POST "${API_URL}" \
    -H 'Content-Type: application/json' \
    -d '{
        "contents": [{
            "parts": [{
                "text": '"$(echo "$FULL_PROMPT" | jq -Rs .)"'
            }]
        }],
        "generationConfig": {
            "temperature": 0.2,
            "topK": 64,
            "topP": 0.95,
            "maxOutputTokens": 8192
        }
    }' | jq -r '.candidates[0].content.parts[0].text')

# Save to output file
cat > "$OUTPUT_FILE" << EOF
$RESPONSE
EOF

echo "✓ Analysis complete!"
echo ""
echo "Output saved to: $OUTPUT_FILE"
echo ""
echo "Preview:"
echo "──────────────────────────────────────"
cat "$OUTPUT_FILE"
echo "──────────────────────────────────────"
