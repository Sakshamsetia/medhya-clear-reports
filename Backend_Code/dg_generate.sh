#!/bin/bash

# ============================================================
# Gemini 2.5 Pro - Analyze File and Save Output
# Usage: bash gemini_analyze_save.sh <file> [prompt]
# ============================================================

API_KEY="AIzaSyATMvDqhAC044jvXOP9f3ZcSFJ3PHS-Xpw"
MODEL="gemini-2.5-pro"
API_URL="https://generativelanguage.googleapis.com/v1/models/${MODEL}:generateContent?key=${API_KEY}"
OUTPUT_FILE="FinalDiagnosis.txt"

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
CUSTOM_PROMPT="You will be provided with a text document containing three separate radiology reports. Your task is to perform two actions in order:

Synthesize Reports: First, analyze all three reports and write a single, concise, consolidated radiology report that integrates the key findings,
 measurements, and impressions from all three. Present this as the "Diagnosis"

Generate Follow-up Explanation: Second, immediately after the diagnosis report, you must generate the detailed 
explanation of the above report in understandable terms.

finally format your entire response exactly as follows, with no additional text or pleasantries, importantly keep the format text only, no markdown formatting:

Diagnosis:<Diagnosis>

Explanation:<Explanation>


(Where <Diagnosis> should be the final diagnosis as a short phrase, and <Explanation> should be a detailed, easy-to-understand explanation of the diagnosis, what it means, and what findings in the report support it.)

"

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
echo "Filling the Predefined Medical Template using LLM"
echo "Date: $(date -u +'%Y-%m-%d %H:%M:%S UTC')"
echo "User: DonQuixote248"
echo "=========================================="
echo ""
echo "Input file: $INPUT_FILE"
echo "Prompt: $CUSTOM_PROMPT"
echo "Output will be saved to: $OUTPUT_FILE"
echo ""
echo "Sending to LLM"
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
