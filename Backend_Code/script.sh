#!/bin/bash
set -e


bash out.sh

sudo bash gemini.sh rad_report.txt
sudo bash dg_generate.sh output_gemini.txt
sudo python3 convert_json.py 
