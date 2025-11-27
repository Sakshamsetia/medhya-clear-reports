#!/usr/bin/env python3
import os, sys
from pathlib import Path
from local_config import JAVA_HOME, JAVA_PATH
os.environ["JAVA_HOME"] = JAVA_HOME
os.environ["PATH"] = JAVA_PATH + os.environ["PATH"]
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / "findings_classifier"))
sys.path.insert(0, str(Path(__file__).parent / "chexbert" / "src"))

import torch
import numpy as np
from PIL import Image

print("RaDialog Inference - User: DonQuixote248 - Date: 2025-10-31 13:22:24 UTC")
print("="*80)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

def load_and_process_image(image_path):
    """Load image using PIL"""
    print(f"Loading: {image_path}")
    img = Image.open(image_path)
    if img.mode != 'L':
        img = img.convert('L')
    print(f"Image loaded: {img.size}")
    return img

image_path = "/home/chipmonkx86/inputIMG/input.jpg"
pil_image = load_and_process_image(image_path)

print("\nLoading model (2-5 min)...")
from transformers import LlamaTokenizer, LlamaForCausalLM
from peft import PeftModel

print("[1/3] Loading tokenizer...")
tokenizer = LlamaTokenizer.from_pretrained("lmsys/vicuna-7b-v1.1")
tokenizer.pad_token = tokenizer.eos_token
print("✓ Tokenizer loaded")

print("\n[2/3] Loading Vicuna-7B base model...")
base_model = LlamaForCausalLM.from_pretrained(
    "lmsys/vicuna-7b-v1.1", 
    torch_dtype=torch.float16
)
base_model = base_model.to(device)
print("✓ Base model loaded")

print("\n[3/3] Loading LoRA adapter...")
model = PeftModel.from_pretrained(
    base_model, 
    "checkpoints/vicuna-7b-img-instruct/checkpoint-4800", 
    torch_dtype=torch.float16
)
model.eval()
print("✓ Model loaded!\n")

print("Generating report...")
prompt = "USER: Describe the findings in this chest X-ray image in detail.\nASSISTANT:"
inputs = tokenizer(prompt, return_tensors="pt", padding=True)
input_ids = inputs["input_ids"].to(device)
attention_mask = inputs["attention_mask"].to(device)

with torch.no_grad():
    outputs = model.generate(
        input_ids=input_ids, 
        attention_mask=attention_mask, 
        max_length=512, 
        num_beams=3, 
        temperature=0.7, 
        do_sample=False, 
        pad_token_id=tokenizer.eos_token_id, 
        eos_token_id=tokenizer.eos_token_id
    )

generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

if "ASSISTANT:" in generated_text:
    report = generated_text.split("ASSISTANT:")[-1].strip()
else:
    report = generated_text.replace(prompt, "").strip()

print("\n" + "="*80)
print("RADIOLOGY REPORT:")
print("="*80)
print(report)
print("="*80)

with open("radiology_report.txt", 'w') as f:
    f.write(f"RaDialog Report - {image_path}\n")
    f.write(f"Date: 2025-10-31 13:22:24 UTC\n")
    f.write("="*80 + "\n")
    f.write(report + "\n")

print("\nSaved to: radiology_report.txt")
print("✅ DONE!")
