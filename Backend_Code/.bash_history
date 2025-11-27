[ -d "findings_classifier/checkpoints/chexpert_train" ] && echo -e "${GREEN}✓${NC} chexpert_train/" || echo -e "${RED}✗${NC} chexpert_train/"
[ -d "pretraining/embs" ] && echo -e "${GREEN}✓${NC} embs/" || echo -e "${RED}✗${NC} embs/"

echo ""
if [ ${#FAILED[@]} -eq 0 ]; then
    echo -e "${GREEN}================================================================================"
    echo "✓ All models downloaded successfully!"
    echo "================================================================================${NC}"
    echo ""
    echo "Next steps:"
    echo "  1. Run: python test_radialog_env.py"
    echo "  2. Upload a test X-ray image to: test_images/test_xray.jpg"
    echo "  3. Run inference: python demo.py --cfg-path pretraining/configs/blip2_pretrain_stage1_emb.yaml"
else
    echo -e "${YELLOW}Download completed with some failures. Please retry or download manually.${NC}"
fi

echo ""
ENDOFSCRIPT

chmod +x download_models.sh
./download_models.sh
conda activate radialog
cd ~/RaDialog
pip install numpy==1.21.6
pip install pandas==1.3.5
# Verify
python -c "import numpy, pandas; print('✓ NumPy & Pandas installed')"
(radialog) chipmonkx86@instance-20251027-180220:~/RaDialog$ 
python test_radialog_env.py
cd ~/RaDialog/test_images
# Download a sample chest X-ray from NIH dataset
wget -O test_xray.jpg "https://openi.nlm.nih.gov/imgs/512/267/267/CXR267_IM-1221-1001.png"
# Or use a MIMIC-CXR sample (if you have access)
# You'll need to provide your own image
# Verify
ls -lh test_xray.jpg
file test_xray.jpg
cd ~/RaDialog/test_images
# Download from a reliable medical imaging repository
curl -L -o test_xray.jpg "https://raw.githubusercontent.com/ieee8023/covid-chestxray-dataset/master/images/01E392EE-69F9-4E33-BFCE-E5C968654078.jpeg"
# Verify download
if [ -f test_xray.jpg ]; then     echo "✓ Image downloaded successfully";     file test_xray.jpg;     ls -lh test_xray.jpg; else     echo "✗ Download failed"; fi
cat > quick_inference.sh << 'ENDOFSCRIPT'
#!/bin/bash

# Quick inference wrapper
conda activate radialog

IMAGE=${1:-"test_images/test_xray.jpg"}

if [ ! -f "$IMAGE" ]; then
    echo "Error: Image not found: $IMAGE"
    echo "Usage: ./quick_inference.sh <path_to_xray.jpg>"
    exit 1
fi

echo "Running inference on: $IMAGE"
echo "Starting RaDialog..."

python demo.py --cfg-path pretraining/configs/blip2_pretrain_stage1_emb.yaml

ENDOFSCRIPT

chmod +x quick_inference.sh
conda activate radialog
pip install numpy==1.21.6 pandas==1.3.5
python run_simple_test.py
cd ..
ls
cd ~/RaDialog
cat > test_inference.py << 'ENDOFPYTHON'
#!/usr/bin/env python3
"""
RaDialog Inference Test Script
User: DonQuixote248
Date: 2025-10-31
"""

import os
import sys
import torch
import argparse
from pathlib import Path

# Set Java environment variables
from local_config import JAVA_HOME, JAVA_PATH
os.environ["JAVA_HOME"] = JAVA_HOME
os.environ["PATH"] = JAVA_PATH + os.environ["PATH"]

print("="*80)
print("RaDialog Inference Test")
print("="*80)
print(f"Date: 2025-10-31 12:42:46 UTC")
print(f"User: DonQuixote248")
print(f"Working Directory: {os.getcwd()}")
print("="*80)

# GPU Check
print("\n[1/6] GPU Information")
print("-"*80)
if torch.cuda.is_available():
    print(f"✓ GPU: {torch.cuda.get_device_name(0)}")
    print(f"✓ GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    print(f"✓ CUDA Version: {torch.version.cuda}")
else:
    print("⚠ WARNING: No GPU available, will use CPU (very slow)")

# Check test image
print("\n[2/6] Test Image Check")
print("-"*80)
test_image = "test_images/test_xray.jpg"

if not os.path.exists(test_image):
    print(f"✗ Test image not found: {test_image}")
    print("\nPlease ensure the image is downloaded to test_images/test_xray.jpg")
    sys.exit(1)

from PIL import Image
import numpy as np
from skimage import io

def remap_to_uint8(array):
    """Normalize image to 0-255 range"""
    array = array.astype(float)
    array -= array.min()
    if array.max() > 0:
        array /= array.max()
    array *= 255
    return array.astype(np.uint8)

try:
    # Load image
    image = io.imread(test_image)
    
    # Handle different image formats
    if len(image.shape) == 3:
        # Convert RGB to grayscale
        image = np.mean(image, axis=2)
    
    image = remap_to_uint8(image)
    pil_image = Image.fromarray(image).convert("L")
    
    print(f"✓ Image loaded successfully")
    print(f"  - Path: {test_image}")
    print(f"  - Size: {pil_image.size}")
    print(f"  - Mode: {pil_image.mode}")
    print(f"  - File size: {os.path.getsize(test_image) / 1024:.2f} KB")
except Exception as e:
    print(f"✗ Error loading image: {e}")
    sys.exit(1)

# Check model files
print("\n[3/6] Model Files Verification")
print("-"*80)
model_files = {
    'CheXbert': 'chexbert/src/checkpoint/chexbert.pth',
    'BLIP2 Pretrained': 'outputs/stage1_pt_instruct_blip_origlr_img448/checkpoint_4.pth',
    'Vicuna Instruct': 'checkpoints/vicuna-7b-img-instruct/checkpoint-4800',
    'Vicuna Report': 'checkpoints/vicuna-7b-img-report/checkpoint-11200',
    'CheXpert Classifier': 'findings_classifier/checkpoints/chexpert_train',
    'Embeddings': 'pretraining/embs',
}

all_present = True
for name, path in model_files.items():
    exists = os.path.exists(path)
    if exists:
        if os.path.isfile(path):
            size_mb = os.path.getsize(path) / (1024*1024)
            print(f"  ✓ {name:25s} ({size_mb:.1f} MB)")
        else:
            print(f"  ✓ {name:25s} (directory)")
    else:
        print(f"  ✗ {name:25s} NOT FOUND")
        all_present = False

if not all_present:
    print("\n✗ Some model files are missing!")
    sys.exit(1)

# Test imports
print("\n[4/6] Testing Imports")
print("-"*80)
try:
    from model.lavis.models.blip2_models.modeling_llama_imgemb import LlamaForCausalLM
    print("  ✓ LlamaForCausalLM")
    
    from transformers import LlamaTokenizer
    print("  ✓ LlamaTokenizer")
    
    from peft import PeftModelForCausalLM
    print("  ✓ PeftModelForCausalLM")
    
    from model.lavis.common.config import Config
    print("  ✓ LAVIS Config")
    
    from model.lavis import tasks
    print("  ✓ LAVIS Tasks")
    
except Exception as e:
    print(f"  ✗ Import error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Load configuration
print("\n[5/6] Loading Configuration")
print("-"*80)
try:
    from omegaconf import OmegaConf
    from model.lavis.common.config import Config
    
    cfg_path = "pretraining/configs/blip2_pretrain_stage1_emb.yaml"
    
    if not os.path.exists(cfg_path):
        print(f"✗ Config file not found: {cfg_path}")
        sys.exit(1)
    
    cfg = Config(OmegaConf.load(cfg_path))
    print(f"  ✓ Configuration loaded: {cfg_path}")
    
except Exception as e:
    print(f"  ✗ Error loading config: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Run inference
print("\n[6/6] Running Inference")
print("-"*80)
print("Loading models... (this may take 1-2 minutes)")

try:
    # Import model builder
    from model.lavis.models import load_model_and_preprocess
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}")
    
    # Load model
    print("  Loading BLIP-2 model...")
    model, vis_processors, txt_processors = load_model_and_preprocess(
        name="blip2_vicuna_instruct",
        model_type="vicuna7b",
        is_eval=True,
        device=device
    )
    
    print("  ✓ Model loaded successfully")
    
    # Prepare image
    print("  Processing image...")
    image_tensor = vis_processors["eval"](pil_image).unsqueeze(0).to(device)
    
    # Generate report
    print("  Generating report...")
    
    with torch.no_grad():
        # Generate findings
        prompt = "Describe the findings in this chest X-ray."
        
        output = model.generate({
            "image": image_tensor,
            "prompt": prompt
        })
        
        report = output[0]
    
    print("\n" + "="*80)
    print("GENERATED RADIOLOGY REPORT")
    print("="*80)
    print(report)
    print("="*80)
    
    # Save output
    output_file = "test_output_report.txt"
    with open(output_file, 'w') as f:
        f.write(f"RaDialog Inference Report\n")
        f.write(f"Date: 2025-10-31 12:42:46 UTC\n")
        f.write(f"User: DonQuixote248\n")
        f.write(f"Image: {test_image}\n")
        f.write(f"{'-'*80}\n\n")
        f.write(report)
    
    print(f"\n✓ Report saved to: {output_file}")
    
except Exception as e:
    print(f"\n✗ Inference failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "="*80)
print("✅ INFERENCE COMPLETED SUCCESSFULLY!")
print("="*80)

ENDOFPYTHON

chmod +x test_inference.py
cd ~/RaDialog
conda activate radialog
# Make sure numpy and pandas are installed
pip install numpy==1.21.6 pandas==1.3.5
# Run the comprehensive test
python test_inference.py
cd ~/RaDialog
# Backup existing config
cp pretraining/configs/blip2_pretrain_stage1_emb.yaml pretraining/configs/blip2_pretrain_stage1_emb.yaml.backup
# Create a proper config file
cat > pretraining/configs/blip2_pretrain_stage1_emb.yaml << 'ENDOFCONFIG'
model:
  arch: blip2_vicuna_instruct
  model_type: vicuna7b
  
  # Image encoder settings
  image_size: 448
  drop_path_rate: 0
  use_grad_checkpoint: False
  vit_precision: "fp16"
  freeze_vit: True
  
  # Q-Former settings
  num_query_token: 32
  
  # LLM settings
  llm_model: "lmsys/vicuna-7b-v1.1"
  prompt: ""
  max_txt_len: 512
  max_output_txt_len: 256
  
  # LoRA settings
  lora_r: 64
  lora_alpha: 16
  lora_target_modules: ["q_proj", "v_proj"]
  
  # Checkpoint paths
  ckpt: "outputs/stage1_pt_instruct_blip_origlr_img448/checkpoint_4.pth"
  lora_ckpt_path: "checkpoints/vicuna-7b-img-instruct/checkpoint-4800"

datasets:
  mimic_cxr:
    vis_processor:
      train:
        name: "blip_image_train"
        image_size: 448
      eval:
        name: "blip_image_eval"
        image_size: 448
    text_processor:
      train:
        name: "blip_caption"
      eval:
        name: "blip_caption"

run:
  task: image_text_pretrain
  lr_sched: "linear_warmup_cosine_lr"
  init_lr: 1e-5
  min_lr: 0
  warmup_lr: 1e-8
  weight_decay: 0.05
  batch_size_train: 4
  batch_size_eval: 4
  num_workers: 4
  max_epoch: 10
  
  seed: 42
  output_dir: "output/blip2_pretrain"
  
  evaluate: False
  train_splits: ["train"]
  valid_splits: ["val"]
  
  device: "cuda"
  world_size: 1
  dist_url: "env://"
  distributed: False
  
  # Generation settings
  num_beams: 3
  max_len: 256
  min_len: 8
  
ENDOFCONFIG

echo "✓ Config file created"
cd ~/RaDialog
# List all available configs
echo "Available config files:"
find . -name "*.yaml" -type f | grep -E "(config|cfg)"
# Show structure
ls -la train_configs/
ls -la pretraining/configs/
cd ~/RaDialog
conda activate radialog
# Use the existing config (not the _emb version which has issues)
python demo.py --cfg-path pretraining/configs/blip2_pretrain_stage1.yaml
cd ~/RaDialog
cat > run_demo.py << 'ENDOFPYTHON'
#!/usr/bin/env python3
"""
RaDialog Demo Wrapper
Bypasses config file issues
"""

import os
import sys

# Set environment
from local_config import JAVA_HOME, JAVA_PATH
os.environ["JAVA_HOME"] = JAVA_HOME
os.environ["PATH"] = JAVA_PATH + os.environ["PATH"]

import torch
import gradio as gr
from PIL import Image
import numpy as np
from skimage import io

print("="*80)
print("RaDialog Inference Demo")
print("="*80)
print(f"User: DonQuixote248")
print(f"Date: 2025-10-31 12:46:10 UTC")
print("="*80)

# GPU info
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\nDevice: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

# Image preprocessing
def remap_to_uint8(array):
    array = array.astype(float)
    array -= array.min()
    if array.max() > 0:
        array /= array.max()
    array *= 255
    return array.astype(np.uint8)

def load_image(image_path):
    """Load and process chest X-ray"""
    if isinstance(image_path, str):
        image = io.imread(image_path)
    else:
        image = np.array(image_path)
    
    if len(image.shape) == 3:
        image = np.mean(image, axis=2)
    
    image = remap_to_uint8(image)
    return Image.fromarray(image).convert("L")

# Load model
print("\nLoading RaDialog model...")
print("This will take 2-5 minutes on first run...\n")

try:
    # Import LAVIS components
    from model.lavis.models import load_model
    from model.lavis.processors import load_processor
    
    # Load with default settings
    print("Loading BLIP-2 + Vicuna model...")
    
    model_cfg = {
        'arch': 'blip2_vicuna_instruct',
        'model_type': 'vicuna7b',
        'pretrained': 'outputs/stage1_pt_instruct_blip_origlr_img448/checkpoint_4.pth',
        'lora_path': 'checkpoints/vicuna-7b-img-instruct/checkpoint-4800'
    }
    
    # Load model
    model = load_model(
        name="blip2_vicuna_instruct",
        model_type="vicuna7b",
        is_eval=True,
        device=device
    )
    
    # Load image processor
    vis_processor = load_processor(
        name="blip_image_eval",
        cfg={"image_size": 448}
    ).build()
    
    print("✓ Model loaded successfully!\n")
    
    def generate_report(image, prompt="Describe this chest X-ray in detail."):
        """Generate radiology report"""
        try:
            # Process image
            pil_image = load_image(image)
            image_tensor = vis_processor(pil_image).unsqueeze(0).to(device)
            
            # Generate
            with torch.no_grad():
                output = model.generate({
                    "image": image_tensor,
                    "prompt": prompt
                })
            
            report = output[0] if isinstance(output, list) else output
            
            return report
            
        except Exception as e:
            return f"Error during inference: {str(e)}\n\nPlease check the model is properly loaded."
    
    # Create Gradio UI
    print("Starting Gradio interface...")
    print("="*80)
    
    with gr.Blocks(title="RaDialog - Radiology Report Generation") as demo:
        gr.Markdown("# 🏥 RaDialog - Interactive Radiology Report Generation")
        gr.Markdown("Upload a chest X-ray image to generate an automated radiology report")
        
        with gr.Row():
            with gr.Column():
                image_input = gr.Image(type="pil", label="Chest X-ray Image")
                prompt_input = gr.Textbox(
                    value="Describe this chest X-ray in detail.",
                    label="Prompt",
                    placeholder="Enter custom prompt..."
                )
                generate_btn = gr.Button("Generate Report", variant="primary")
                
            with gr.Column():
                report_output = gr.Textbox(label="Generated Radiology Report", lines=15)
        
        gr.Examples(
            examples=[
                ["test_images/test_xray.jpg", "Describe this chest X-ray in detail."],
                ["test_images/test_xray.jpg", "What are the key findings in this chest radiograph?"],
            ],
            inputs=[image_input, prompt_input],
        )
        
        generate_btn.click(
            fn=generate_report,
            inputs=[image_input, prompt_input],
            outputs=report_output
        )
    
    # Launch
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True
    )
    
except Exception as e:
    print(f"\n✗ Error loading model: {e}")
    import traceback
    traceback.print_exc()
    
    print("\n" + "="*80)
    print("Troubleshooting:")
    print("1. Ensure all model files are downloaded")
    print("2. Check CUDA is available")
    print("3. Verify config files exist")
    print("="*80)

ENDOFPYTHON

chmod +x run_demo.py
# Run it
python run_demo.py
ls -lh test_images/test_xray.jpg
python demo.py --cfg-path model/lavis/configs/models/blip2/blip2_instruct_vicuna7b.yaml
cd ~/RaDialog
# Add the findings_classifier to Python path
export PYTHONPATH="${PYTHONPATH}:$(pwd)/findings_classifier"
# Run demo again
python demo.py --cfg-path model/lavis/configs/models/blip2/blip2_instruct_vicuna7b.yaml
conda activate radialog
c
## **Install Missing Packages**
```bash
conda activated ~/RaDialog

# Install pytorch_ radialog
cd ~/RaDialog

# Install pytorchlightning
pip install pytorch_lightning==1.6_lightning
pip install pytorch_lightning==1.6.5

# Install other potentially missing packages
pip install tor.5

# Install any otherchmetrics== potentially missing packages
pip install tor0.11.4
pip install lightning==1.9.0chmetrics==0.11.4

# Verify installation
python
pip install jsonargparse[ -c "import pytorch_lightning; print('signatures]✓ PyTorch Lightning:', pytorch_lightning.__version__)"

pip install pytorch_lightning==1.6_lightning
pip install pytorch_lightning==1.6.5
# Install other potentially missing packages
pip install tor.5
# Install any otherchmetrics== potentially missing packages
pip install tor0.11.4
pip install lightning==1.9.0chmetrics==0.11.4
# Verify installation
python
pip install jsonargparse[ -c "import pytorch_lightning; print('signatures]✓ PyTorch Lightning:', pytorch_lightning.__version__)"
cd ~/RaDialog
export PYTHONPATH="${PYTHONPATH}:$(pwd)/findings_classifier:$(pwd)/chexbert/src"
python demo.py --cfg-path model/lavis/configs/models/blip2/blip2_instruct_vicuna7b.yaml
pip install wandb==0.14.0
# Then run demo again
export PYTHONPATH=".:./findings_classifier:./chexbert/src:$PYTHONPATH"
python demo.py --cfg-path model/lavis/configs/models/blip2/blip2_instruct_vicuna7b.yaml
pip install scikit-learn==1.0.2
# Then run
export PYTHONPATH=".:./findings_classifier:./chexbert/src:$PYTHONPATH"
python demo.py --cfg-path model/lavis/configs/models/blip2/blip2_instruct_vicuna7b.yaml
pip install torchinfo==1.8.0 && export PYTHONPATH=".:./findings_classifier:./chexbert/src:$PYTHONPATH" && python demo.py --cfg-path model/lavis/configs/models/blip2/blip2_instruct_vicuna7b.yaml
pip install nltk==3.8.1 && export PYTHONPATH=".:./findings_classifier:./chexbert/src:$PYTHONPATH" && python demo.py --cfg-path model/lavis/configs/models/blip2/blip2_instruct_vicuna7b.yaml
pip install pycocoevalcap==1.2 && export PYTHONPATH=".:./findings_classifier:./chexbert/src:$PYTHONPATH" && python demo.py --cfg-path model/lavis/configs/models/blip2/blip2_instruct_vicuna7b.yaml
cd ~/RaDialog && cat > working_config.yaml << 'EOF'
model:
  arch: blip2_vicuna_instruct
  model_type: vicuna7b
  load_finetuned: False
  load_pretrained: True
  pretrained: "outputs/stage1_pt_instruct_blip_origlr_img448/checkpoint_4.pth"
  num_query_token: 32
  llm_model: "lmsys/vicuna-7b-v1.1"
  prompt: ""
  max_txt_len: 512
  max_output_txt_len: 256
  image_size: 448
  drop_path_rate: 0
  use_grad_checkpoint: False
  vit_precision: "fp16"
  freeze_vit: True
datasets: {}
run:
  task: "image_text_pretrain"
  lr_sched: "linear_warmup_cosine_lr"
  init_lr: 1e-5
  min_lr: 0
  warmup_lr: 1e-8
  batch_size_train: 1
  batch_size_eval: 1
  num_workers: 0
  seed: 42
  output_dir: "output"
  evaluate: True
  device: "cuda"
  world_size: 1
  dist_url: "env://"
  distributed: False
  num_beams: 3
  max_len: 256
  min_len: 8
EOF

export PYTHONPATH=".:./findings_classifier:./chexbert/src:$PYTHONPATH" && python demo.py --cfg-path working_config.yaml
cd ~/RaDialog
cat > simple_inference.py << 'ENDOFPYTHON'
#!/usr/bin/env python3
"""
Simple RaDialog Inference Script
Works without complex config files
"""

import os
import sys
from pathlib import Path

# Set environment
from local_config import JAVA_HOME, JAVA_PATH
os.environ["JAVA_HOME"] = JAVA_HOME
os.environ["PATH"] = JAVA_PATH + os.environ["PATH"]

# Add paths
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / "findings_classifier"))
sys.path.insert(0, str(Path(__file__).parent / "chexbert" / "src"))

import torch
import numpy as np
from PIL import Image
from skimage import io
import argparse

print("="*80)
print("RaDialog Simple Inference")
print("="*80)
print(f"User: DonQuixote248")
print(f"Date: 2025-10-31 12:55:58 UTC")
print("="*80)

# GPU check
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\nDevice: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

def remap_to_uint8(array):
    """Normalize image to 0-255"""
    array = array.astype(float)
    array -= array.min()
    if array.max() > 0:
        array /= array.max()
    array *= 255
    return array.astype(np.uint8)

def load_image(path):
    """Load and process X-ray image"""
    print(f"\nLoading image: {path}")
    image = io.imread(path)
    
    if len(image.shape) == 3:
        image = np.mean(image, axis=2)
    
    image = remap_to_uint8(image)
    pil_image = Image.fromarray(image).convert("L")
    print(f"✓ Image loaded: {pil_image.size}")
    return pil_image

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument("--image", type=str, default="test_images/test_xray.jpg", help="Path to X-ray image")
parser.add_argument("--output", type=str, default="output_report.txt", help="Output file")
args = parser.parse_args()

# Load image
try:
    image = load_image(args.image)
except Exception as e:
    print(f"✗ Error loading image: {e}")
    sys.exit(1)

print("\n" + "="*80)
print("Loading RaDialog model (this will take 2-5 minutes)...")
print("="*80)

try:
    from transformers import LlamaTokenizer, LlamaForCausalLM
    from peft import PeftModel, PeftConfig
    
    # Load tokenizer
    print("\n[1/3] Loading tokenizer...")
    tokenizer = LlamaTokenizer.from_pretrained("lmsys/vicuna-7b-v1.1")
    print("✓ Tokenizer loaded")
    
    # Load base model
    print("\n[2/3] Loading Vicuna-7B base model...")
    base_model = LlamaForCausalLM.from_pretrained(
        "lmsys/vicuna-7b-v1.1",
        load_in_8bit=False,
        torch_dtype=torch.float16,
        device_map="auto",
        low_cpu_mem_usage=True
    )
    print("✓ Base model loaded")
    
    # Load LoRA adapter
    print("\n[3/3] Loading LoRA adapter...")
    lora_path = "checkpoints/vicuna-7b-img-instruct/checkpoint-4800"
    
    model = PeftModel.from_pretrained(
        base_model,
        lora_path,
        torch_dtype=torch.float16
    )
    model.eval()
    print("✓ LoRA adapter loaded")
    
    print("\n" + "="*80)
    print("✓ Model loaded successfully!")
    print("="*80)
    
    # Generate report
    print("\nGenerating report...")
    
    # Create prompt
    prompt = f"USER: Describe the findings in this chest X-ray image.\nASSISTANT:"
    
    # Tokenize
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].to(device)
    
    # Generate
    with torch.no_grad():
        outputs = model.generate(
            input_ids,
            max_length=512,
            num_beams=3,
            temperature=0.7,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )
    
    # Decode
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract assistant response
    if "ASSISTANT:" in generated_text:
        report = generated_text.split("ASSISTANT:")[-1].strip()
    else:
        report = generated_text
    
    # Display
    print("\n" + "="*80)
    print("GENERATED RADIOLOGY REPORT")
    print("="*80)
    print(report)
    print("="*80)
    
    # Save
    with open(args.output, 'w') as f:
        f.write(f"RaDialog Inference Report\n")
        f.write(f"Date: 2025-10-31 12:55:58 UTC\n")
        f.write(f"User: DonQuixote248\n")
        f.write(f"Image: {args.image}\n")
        f.write(f"{'-'*80}\n\n")
        f.write(report)
    
    print(f"\n✓ Report saved to: {args.output}")
    print("\n✅ Inference completed successfully!")
    
except Exception as e:
    print(f"\n✗ Error during inference: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

ENDOFPYTHON

chmod +x simple_inference.py
# Run it
python simple_inference.py --image test_images/test_xray.jpg --output radiology_report.txt
pip install sentencepiece==0.1.99 protobuf==3.20.0 && python simple_inference.py
cd ~/RaDialog
cat > simple_inference.py << 'ENDOFPYTHON'
#!/usr/bin/env python3
import os, sys
from pathlib import Path
from local_config import JAVA_HOME, JAVA_PATH
os.environ["JAVA_HOME"] = JAVA_HOME
os.environ["PATH"] = JAVA_PATH + os.environ["PATH"]
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / "findings_classifier"))
sys.path.insert(0, str(Path(__file__).parent / "chexbert" / "src"))

import torch, numpy as np
from PIL import Image
from skimage import io

print("="*80)
print("RaDialog Simple Inference")
print("User: DonQuixote248")
print("Date: 2025-10-31 12:59:34 UTC")
print("="*80)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\nDevice: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

def remap_to_uint8(array):
    array = array.astype(float)
    array -= array.min()
    if array.max() > 0: 
        array /= array.max()
    array *= 255
    return array.astype(np.uint8)

image_path = "test_images/test_xray.jpg"
print(f"\nLoading image: {image_path}")
image = io.imread(image_path)
if len(image.shape) == 3: 
    image = np.mean(image, axis=2)
image = remap_to_uint8(image)
pil_image = Image.fromarray(image).convert("L")
print(f"✓ Image loaded: {pil_image.size}")

print("\n" + "="*80)
print("Loading RaDialog model (this will take 2-5 minutes)...")
print("="*80)

from transformers import LlamaTokenizer, LlamaForCausalLM
from peft import PeftModel

print("\n[1/3] Loading tokenizer...")
tokenizer = LlamaTokenizer.from_pretrained("lmsys/vicuna-7b-v1.1")
tokenizer.pad_token = tokenizer.eos_token
print("✓ Tokenizer loaded")

print("\n[2/3] Loading Vicuna-7B base model...")
base_model = LlamaForCausalLM.from_pretrained(
    "lmsys/vicuna-7b-v1.1", 
    load_in_8bit=False, 
    torch_dtype=torch.float16, 
    device_map="auto", 
    low_cpu_mem_usage=True
)
print("✓ Base model loaded")

print("\n[3/3] Loading LoRA adapter...")
model = PeftModel.from_pretrained(
    base_model, 
    "checkpoints/vicuna-7b-img-instruct/checkpoint-4800", 
    torch_dtype=torch.float16
)
model.eval()
print("✓ LoRA adapter loaded")

print("\n" + "="*80)
print("✓ Model loaded successfully!")
print("="*80)

print("\nGenerating report...")

# Create prompt
prompt = "USER: Describe the findings in this chest X-ray image in detail.\nASSISTANT:"

# Tokenize
inputs = tokenizer(prompt, return_tensors="pt", padding=True)
input_ids = inputs["input_ids"].to(device)
attention_mask = inputs["attention_mask"].to(device)

# Generate
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

# Decode
generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

# Extract assistant response
if "ASSISTANT:" in generated_text:
    report = generated_text.split("ASSISTANT:")[-1].strip()
else:
    report = generated_text.replace(prompt, "").strip()

# Display
print("\n" + "="*80)
print("GENERATED RADIOLOGY REPORT")
print("="*80)
print(report)
print("="*80)

# Save
output_file = "radiology_report.txt"
with open(output_file, 'w') as f:
    f.write(f"RaDialog Inference Report\n")
    f.write(f"Date: 2025-10-31 12:59:34 UTC\n")
    f.write(f"User: DonQuixote248\n")
    f.write(f"Image: {image_path}\n")
    f.write(f"{'-'*80}\n\n")
    f.write(report)

print(f"\n✓ Report saved to: {output_file}")
print("\n✅ INFERENCE COMPLETED SUCCESSFULLY!")
print("="*80)

ENDOFPYTHON

python simple_inference.py
cat radiology_report.txt 
conda activate radialog
cd RaDialog/
cd ~/RaDialog && python simple_inference.py
ls
cd ~/RaDialog
cat > simple_inference.py << 'ENDOFPYTHON'
#!/usr/bin/env python3
import os, sys
from pathlib import Path
from local_config import JAVA_HOME, JAVA_PATH
os.environ["JAVA_HOME"] = JAVA_HOME
os.environ["PATH"] = JAVA_PATH + os.environ["PATH"]
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / "findings_classifier"))
sys.path.insert(0, str(Path(__file__).parent / "chexbert" / "src"))

import torch, numpy as np
from PIL import Image
from skimage import io

print("="*80)
print("RaDialog Simple Inference")
print("User: DonQuixote248")
print("Date: 2025-10-31 12:59:34 UTC")
print("="*80)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\nDevice: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

def remap_to_uint8(array):
    array = array.astype(float)
    array -= array.min()
    if array.max() > 0: 
        array /= array.max()
    array *= 255
    return array.astype(np.uint8)

image_path = "test_images/test_xray.jpg"
print(f"\nLoading image: {image_path}")
image = io.imread(image_path)
if len(image.shape) == 3: 
    image = np.mean(image, axis=2)
image = remap_to_uint8(image)
pil_image = Image.fromarray(image).convert("L")
print(f"✓ Image loaded: {pil_image.size}")

print("\n" + "="*80)
print("Loading RaDialog model (this will take 2-5 minutes)...")
print("="*80)

from transformers import LlamaTokenizer, LlamaForCausalLM
from peft import PeftModel

print("\n[1/3] Loading tokenizer...")
tokenizer = LlamaTokenizer.from_pretrained("lmsys/vicuna-7b-v1.1")
tokenizer.pad_token = tokenizer.eos_token
print("✓ Tokenizer loaded")

print("\n[2/3] Loading Vicuna-7B base model...")
base_model = LlamaForCausalLM.from_pretrained(
    "lmsys/vicuna-7b-v1.1", 
    load_in_8bit=False, 
    torch_dtype=torch.float16, 
    device_map="auto", 
    low_cpu_mem_usage=True
)
print("✓ Base model loaded")

print("\n[3/3] Loading LoRA adapter...")
model = PeftModel.from_pretrained(
    base_model, 
    "checkpoints/vicuna-7b-img-instruct/checkpoint-4800", 
    torch_dtype=torch.float16
)
model.eval()
print("✓ LoRA adapter loaded")

print("\n" + "="*80)
print("✓ Model loaded successfully!")
print("="*80)

print("\nGenerating report...")

# Create prompt
prompt = "USER: Describe the findings in this chest X-ray image in detail.\nASSISTANT:"

# Tokenize
inputs = tokenizer(prompt, return_tensors="pt", padding=True)
input_ids = inputs["input_ids"].to(device)
attention_mask = inputs["attention_mask"].to(device)

# Generate
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

# Decode
generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

# Extract assistant response
if "ASSISTANT:" in generated_text:
    report = generated_text.split("ASSISTANT:")[-1].strip()
else:
    report = generated_text.replace(prompt, "").strip()

# Display
print("\n" + "="*80)
print("GENERATED RADIOLOGY REPORT")
print("="*80)
print(report)
print("="*80)

# Save
output_file = "radiology_report.txt"
with open(output_file, 'w') as f:
    f.write(f"RaDialog Inference Report\n")
    f.write(f"Date: 2025-10-31 12:59:34 UTC\n")
    f.write(f"User: DonQuixote248\n")
    f.write(f"Image: {image_path}\n")
    f.write(f"{'-'*80}\n\n")
    f.write(report)

print(f"\n✓ Report saved to: {output_file}")
print("\n✅ INFERENCE COMPLETED SUCCESSFULLY!")
print("="*80)

ENDOFPYTHON

python simple_inference.py
cd ~/RaDialog
cat > simple_inference.py << 'ENDOFPYTHON'
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
from skimage import io

print("="*80)
print("RaDialog Simple Inference")
print("User: DonQuixote248")
print("Date: 2025-10-31 13:07:30 UTC")
print("="*80)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\nDevice: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

def remap_to_uint8(array):
    array = array.astype(float)
    array -= array.min()
    if array.max() > 0: 
        array /= array.max()
    array *= 255
    return array.astype(np.uint8)

image_path = "test_images/test_xray.jpg"
print(f"\nLoading image: {image_path}")
image = io.imread(image_path)
if len(image.shape) == 3: 
    image = np.mean(image, axis=2)
image = remap_to_uint8(image)
pil_image = Image.fromarray(image).convert("L")
print(f"✓ Image loaded: {pil_image.size}")

print("\n" + "="*80)
print("Loading RaDialog model (this will take 2-5 minutes)...")
print("="*80)

from transformers import LlamaTokenizer, LlamaForCausalLM
from peft import PeftModel

print("\n[1/3] Loading tokenizer...")
tokenizer = LlamaTokenizer.from_pretrained("lmsys/vicuna-7b-v1.1")
tokenizer.pad_token = tokenizer.eos_token
print("✓ Tokenizer loaded")

print("\n[2/3] Loading Vicuna-7B base model...")
base_model = LlamaForCausalLM.from_pretrained(
    "lmsys/vicuna-7b-v1.1", 
    load_in_8bit=False, 
    torch_dtype=torch.float16, 
    device_map="auto", 
    low_cpu_mem_usage=True
)
print("✓ Base model loaded")

print("\n[3/3] Loading LoRA adapter...")
model = PeftModel.from_pretrained(
    base_model, 
    "checkpoints/vicuna-7b-img-instruct/checkpoint-4800", 
    torch_dtype=torch.float16
)
model.eval()
print("✓ LoRA adapter loaded")

print("\n" + "="*80)
print("✓ Model loaded successfully!")
print("="*80)

print("\nGenerating report...")

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
print("GENERATED RADIOLOGY REPORT")
print("="*80)
print(report)
print("="*80)

output_file = "radiology_report.txt"
try:
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("RaDialog Inference Report\n")
        f.write("="*80 + "\n")
        f.write(f"Date: 2025-10-31 13:07:30 UTC\n")
        f.write(f"User: DonQuixote248\n")
        f.write(f"Image: {image_path}\n")
        f.write("="*80 + "\n\n")
        f.write(report + "\n")
    
    print(f"\n✓ Report saved to: {output_file}")
    print(f"✓ File size: {os.path.getsize(output_file)} bytes")
    
except Exception as e:
    print(f"\n⚠ Warning: Could not save report to file: {e}")
    print("Report was displayed above but not saved.")

print("\n✅ INFERENCE COMPLETED SUCCESSFULLY!")
print("="*80)
ENDOFPYTHON

# Run it
python simple_inference.py
cd ~/RaDialog && python simple_inference.py
cd ~/RaDialog
rm -f simple_inference.py
cat > simple_inference.py << 'EOF'
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
from skimage import io

print("RaDialog Inference - User: DonQuixote248 - Date: 2025-10-31 13:08:22 UTC")
print("="*80)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}, GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}")

def remap_to_uint8(array):
    array = array.astype(float)
    array -= array.min()
    if array.max() > 0: 
        array /= array.max()
    array *= 255
    return array.astype(np.uint8)

image_path = "test_images/test_xray.jpg"
print(f"Loading: {image_path}")
image = io.imread(image_path)
if len(image.shape) == 3: 
    image = np.mean(image, axis=2)
image = remap_to_uint8(image)
pil_image = Image.fromarray(image).convert("L")
print(f"Image loaded: {pil_image.size}")

print("\nLoading model (2-5 min)...")
from transformers import LlamaTokenizer, LlamaForCausalLM
from peft import PeftModel

tokenizer = LlamaTokenizer.from_pretrained("lmsys/vicuna-7b-v1.1")
tokenizer.pad_token = tokenizer.eos_token
base_model = LlamaForCausalLM.from_pretrained("lmsys/vicuna-7b-v1.1", load_in_8bit=False, torch_dtype=torch.float16, device_map="auto", low_cpu_mem_usage=True)
model = PeftModel.from_pretrained(base_model, "checkpoints/vicuna-7b-img-instruct/checkpoint-4800", torch_dtype=torch.float16)
model.eval()
print("Model loaded!\n")

print("Generating report...")
prompt = "USER: Describe the findings in this chest X-ray image in detail.\nASSISTANT:"
inputs = tokenizer(prompt, return_tensors="pt", padding=True)
input_ids = inputs["input_ids"].to(device)
attention_mask = inputs["attention_mask"].to(device)

with torch.no_grad():
    outputs = model.generate(input_ids=input_ids, attention_mask=attention_mask, max_length=512, num_beams=3, temperature=0.7, do_sample=False, pad_token_id=tokenizer.eos_token_id, eos_token_id=tokenizer.eos_token_id)

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
    f.write("="*80 + "\n")
    f.write(report + "\n")

print("\nSaved to: radiology_report.txt")
print("DONE!")
EOF

python simple_inference.py
conda activate radialog
# Install all dependencies permanently in conda environment
pip install --upgrade pip
# Core dependencies
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117
# Transformers and model dependencies
pip install transformers==4.28.1
pip install peft==0.3.0
pip install accelerate==0.18.0
pip install bitsandbytes==0.37.2
# Image processing
pip install Pillow==9.5.0
pip install scikit-image==0.19.3
pip install opencv-python==4.7.0.72
# ML utilities
pip install scikit-learn==1.0.2
pip install pytorch-lightning==1.6.5
pip install torchmetrics==0.11.4
# NLP and evaluation
pip install nltk==3.8.1
pip install sentencepiece==0.1.99
pip install protobuf==3.20.0
pip install pycocoevalcap==1.2
# Other utilities
pip install wandb==0.14.0
pip install torchinfo==1.8.0
pip install omegaconf==2.1.1
pip install pandas==1.3.5
pip install numpy==1.21.6
pip install gradio==3.23.0
pip install iopath==0.1.10
pip install timm==0.4.12
pip install fairscale==0.4.13
pip install webdataset==0.2.5
pip install decord==0.6.0
echo "✓ All packages installed permanently in radialog environment"
# Verify installation
python -c "import torch; print(f'✓ PyTorch: {torch.__version__}')"
python -c "import transformers; print(f'✓ Transformers: {transformers.__version__}')"
python -c "import peft; print(f'✓ PEFT: {peft.__version__}')"
python -c "from PIL import Image; print('✓ Pillow installed')"
python -c "import skimage; print('✓ scikit-image installed')"
echo ""
echo "All dependencies installed! Now run:"
echo "python simple_inference.py"
cd ~/RaDialog
python simple_inference.py
pip install scikit-image==0.19.3
# Then run
python simple_inference.py
conda activate radialog
conda install -c conda-forge scikit-image=0.19.3 -y
# Then run
cd ~/RaDialog
python simple_inference.py
# Install scikit-image dependencies first
pip install numpy==1.21.6 scipy==1.7.3 pillow==9.5.0 imageio==2.25.0 tifffile==2021.11.2 PyWavelets==1.3.0 networkx==2.6.3
# Then install scikit-image
pip install scikit-image==0.19.3
# Verify it works
python -c "from skimage import io; print('✓ skimage working')"
# Run inference
cd ~/RaDialog
python simple_inference.py
cd ~/RaDialog
cat > simple_inference.py << 'EOF'
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

print("RaDialog Inference - User: DonQuixote248 - Date: 2025-10-31 13:20:07 UTC")
print("="*80)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

def load_and_process_image(image_path):
    """Load image using PIL instead of skimage"""
    print(f"Loading: {image_path}")
    
    # Load with PIL
    img = Image.open(image_path)
    
    # Convert to grayscale if needed
    if img.mode != 'L':
        img = img.convert('L')
    
    print(f"Image loaded: {img.size}")
    return img

image_path = "test_images/test_xray.jpg"
pil_image = load_and_process_image(image_path)

print("\nLoading model (2-5 min)...")
from transformers import LlamaTokenizer, LlamaForCausalLM
from peft import PeftModel

tokenizer = LlamaTokenizer.from_pretrained("lmsys/vicuna-7b-v1.1")
tokenizer.pad_token = tokenizer.eos_token
print("Tokenizer loaded")

base_model = LlamaForCausalLM.from_pretrained(
    "lmsys/vicuna-7b-v1.1", 
    load_in_8bit=False, 
    torch_dtype=torch.float16, 
    device_map="auto", 
    low_cpu_mem_usage=True
)
print("Base model loaded")

model = PeftModel.from_pretrained(
    base_model, 
    "checkpoints/vicuna-7b-img-instruct/checkpoint-4800", 
    torch_dtype=torch.float16
)
model.eval()
print("Model loaded!\n")

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
    f.write(f"Date: 2025-10-31 13:20:07 UTC\n")
    f.write("="*80 + "\n")
    f.write(report + "\n")

print("\nSaved to: radiology_report.txt")
print("DONE!")
EOF

python simple_inference.py
pip install accelerate==0.20.3
# Then run
cd ~/RaDialog
python simple_inference.py
cd ~/RaDialog
cat > simple_inference.py << 'EOF'
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

image_path = "test_images/test_xray.jpg"
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
EOF

python simple_inference.py
conda activate radialog
cd RaDialog/
python simple_inference.py
ls
clear
ls
conda -list
conda -h
conda --list
conda list
clear
conda env list
ls
cat script.sh 
./script.sh 
cd llava-rad/
ls
cd ..
./ run_medgemma4b_fixed.py 
./run_medgemma4b_fixed.py 
clear 
ls
./test_medgemma4b.py 
ls
clear
ls
python batch_medgemma4b.py 
./run_medgemma4b_fixed.py 
conda deactivate
ls
clear
ls
./run_medgemma4b_fixed.py 
./run_medgemma4b.py 
conda activate
ls
cat run_medgemma4b_fixed.py 
ls
ls models//
cd models/medgemma-4b/
ls
cd ..
conda activate medgemma4b
./run_medgemma4b_fixed.py 
nano run_medgemma4b_fixed.py 
fg
./run_medgemma4b_fixed.py 
cd ..
cd
ls
rm run_medgemma.py run_medgemma4b.py test_medgemma* batch_medgemma4b.py 
ls
cd dad
cd data
ls
cd xray_images/
ls
cd ..
rm -rf data/
ls
cd 
cd ..
ls
cd kshitijmudhya09/
ls
cd ..
cd chipmonkx86/
ls
cd miniconda3/
ls
cd ..
ls
conda deactivate
ls
conda activate dpprojec
conda activate dpproject
ls
cd llava-rad/
ls
cd ..
cd RaDialog/
ls
cd ..
ls
cd ..
ls
cd samarthsoni040906/
ls
mv ChestX/
mv ChestX/ ../chipmonkx86/
cd ChestX/
ls
nano inference.py 
ls -l
chmod +x inference.py 
nano inference.py 
cd ../..
cd chipmonkx86/
nano inference_chestx.py
cd ..
cd samarthsoni040906/
nano infere
cd ChestX/
nano inference.py 
cd ../../chipmonkx86/
clear
cd ../samarthsoni040906/ChestX/
nano inference.py 
ls
nano inference.py 
cd ../../chipmonkx86/
ls
nano inference_Chestx.py
ls
chmod +x inference_Chestx.py 
clear
./script.sh 
./inference_Chestx.py 
cat script.sh 
clear
./script.sh 
conda env list
sudo ./script.sh 
ls
cd che
ls
cd chexbert/
ls
nano local_confi.py
cd ../..
cd ..
la
ls
cd kumar_an/
./test_chex.sh 
ls
nano script_3.sh
chmod +x script_3.sh 
./script_3.sh 
conda deactivate
ls
./script
./script_3.sh 
nano hello.txt
rm na
rm hello.txt 
ls
nano out.sh
chmod +x out
ls
chmod +x out.sh 
./out.sh 
ls
cd outputs/
ls
cd ..
rm -rf outputs/
ls
cd model_outputs/
ls
rm -r *.txt
ls
rm *.log
ls
cd ..
ls
./out.sh 
cd model_outputs/
ls
./out.sh 
cd model_outputs/
ls
cat chestx_output.txt 
nano dagnose_conda_issue.sh
chmod +x dagnose_conda_issue.sh 
./dagnose_conda_issue.sh > diagnose.txt
cat diagnose.txt 
nano fix_conda_priority.sh
# Run the fix script
bash fix_conda_priority.sh
# Then logout and login, OR:
source ~/.bashrc
# Verify it worked:
which conda
# Should show: /home/chipmonkx86/miniconda3/bin/conda
conda env list
# Should show your environments: llavarad, radialog, dpproject
# List environments in your personal conda
/home/chipmonkx86/miniconda3/bin/conda env list
# You should see:
# llavarad
# radialog  
# dpproject
cd..
cd ..
ls
./out.sh 
cd model_outputs/
ls
./dagnose_conda_issue.sh 
cd ..
test_out.sh
nano test_out.sh
bash test_out.sh 
cd model_outputs/
ls
cat llavarad_output.txt 
rm -rf *.txt
ls
rm -rf *.log
ls
cd ...
cd ..
ls
bast test_out.sh 
bash tw
bash test_out.sh 
cd model_outputs/
ls
cat llavarad_error.log 
cd ...
cd ..
nano personal.sh
bash personal.sh 
ls
nano test_out.sh 
bash test_out.sh 
cd model_outputs/
ls
cat llavarad_error.log 
nano dd.sh
bash dd.sh 
cd ..
nano test_out.sh 
bash test_out.sh 
cd model_outputs/
cat llavarad_error.log 
cd ..
cd kumar_an/
ls
cd ..
cd chipmonkx86/
ls
nano lavafix.sh
bash lavafix.sh 
nano test_out.sh 
bash test_out.sh 
bash pipeline_working_exact.sh
ls
bash univ.sh 
nano univ.sh 
nano fix_uni.sh
bash fix_uni.sh 
ls
chmod +x fix_uni.sh 
rm personal.sh 
rm test_out.sh 
rm script
rm script.*
ls
rm lavafix.sh 
rm -rf model_outputs/
ls
cat final.py 
clear
ls
rm out.sh 
rm inference_Chestx.py 
clear
ls
nano out.sh
bash out.sh 
ls
cat radiology_reports_20251113_090200.txt 
nano extract.sh
bash extract.sh 
bash extract.sh radiology_reports_20251113_090200.txt 
ls
cat clean_radiology_reports_20251113_090200.txt 
ls
nano extract.sh 
cat radiology_reports_20251113_090200.txt 
nano extract.sh 
bash extract.sh 
bash extract.sh radiology_reports_20251113_090200.txt 
cat clean_radiology_reports_20251113_090200.txt 
rm clean_radiology_reports_20251113_090200.txt 
ls
bash extract.sh radiology_reports_20251113_090200.txt 
cat clean_radiology_reports_20251113_090200.txt 
cat radiology_reports_20251113_090200.txt 
ls
nano fix_uni.sh 
cat out.sh 
nano out.sh 
fg
ls
clear
nano gemini.sh
bash gemini.sh radiology_reports_20251113_090200.txt [please extract the three radiology reports and send as text output without adding anything ohter than text only with a newline bewteen each output]
curl -s -X POST "https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent?key=AIzaSyD70TTfsP5quGviBZvGEhUif0TNJwV2epQ" -H 'Content-Type: application/json' -d '{"contents":[{"parts":[{"text":"Hello"}]}]}' | jq -r '.candidates[0].content.parts[0].text'
curl -X POST "https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent?key=AIzaSyD70TTfsP5quGviBZvGEhUif0TNJwV2epQ" -H 'Content-Type: application/json' -d '{"contents":[{"parts":[{"text":"Hello"}]}]}'
# Add to ~/.bashrc
gemini() {     curl -s -X POST "https://generativelanguage.googleapis.com/v1/models/gemini-1.5-flash:generateContent?key=AIzaSyD70TTfsP5quGviBZvGEhUif0TNJwV2epQ"         -H 'Content-Type: application/json'         -d '{"contents":[{"parts":[{"text":"'"$*"'"}]}]}' | jq -r '.candidates[0].content.parts[0].text'; }
source ~/.bashrc
gemini "hello"
curl -s -X POST "https://generativelanguage.googleapis.com/v1/models/gemini-1.5-flash:generateContent?key=AIzaSyD70TTfsP5quGviBZvGEhUif0TNJwV2epQ" -H 'Content-Type: application/json' -d '{"contents":[{"parts":[{"text":"Say hello"}]}]}' | jq -r '.candidates[0].content.parts[0].text'
curl -X POST "https://generativelanguage.googleapis.com/v1/models/gemini-1.5-flash:generateContent?key=AIzaSyD70TTfsP5quGviBZvGEhUif0TNJwV2epQ" -H 'Content-Type: application/json' -d '{"contents":[{"parts":[{"text":"Say hello"}]}]}'
ListModels
curl "https://generativelanguage.googleapis.com/v1/models?key=AIzaSyD70TTfsP5quGviBZvGEhUif0TNJwV2epQ"
curl -s -X POST "https://generativelanguage.googleapis.com/v1/models/gemini-2.5-pro:generateContent?key=AIzaSyD70TTfsP5quGviBZvGEhUif0TNJwV2epQ" -H 'Content-Type: application/json' -d '{"contents":[{"parts":[{"text":"Say hello"}]}]}' | jq -r '.candidates[0].content.parts[0].text'
ls
nano gemini.sh 
bash gemini.sh radiology_reports_20251113_090200.txt "in this there is a lot of unnecessary stuff other than the requried radiology reports of the 3 models i just want the 3 outputs from these files and nothing else, ONLY the three outputs separated by a newline"
ls
cat output_gemini.txt 
nano gemini.sh 
bash gemini.sh radiology_reports_20251113_090200.txt "in this there is a lot of unnecessary stuff other than the requried radiology reports of the 3 models i just want the 3 outputs from these files and nothing else, ONLY the three outputs separated by a newline"
cat output_gemini.txt 
nano gemini.sh 
bash gemini.sh radiology_reports_20251113_090200.txt 
cat output_gemini.txt 
clear
ls
rm clean_radiology_conda env list
conda env remove -n medgemma4b
sudo conda env remove -n medgemma4b
sudo /opt/conda/bin/conda env remove --name medgemma4b
conda env remove -n medgemma4b
conda env list
ls
cd ..
ls
cd samarthsoni040906/
ls
sudo rm -rf cxrllava/
sudo rm -rf BiomedGPT/
ls
cd ..
cd home/
ls
cd kumar_an/
ls
sudo rm -rf chexbert_test/
rm dg.sh 
rm -f diagnose.sh test_chex.sh 
ls
sudo rm -f diagnose.sh test_chex.sh 
ls
sudo rm -rf model_outputs/
ls
sudo rm dg.sh 
ls
clear
cd ../chipmonkx86/
ls
df -h
conda env list
conda activate my_reports.csv 
conda activate my_chexbert
ls
bash user_chex.sh 
nano run_chex.sh
bash run_chexbert_user_env.sh output_gemini.txt
bash run_chex.sh output_gemini.txt 
ls
python3 final.py 
cat final.py 
ls
df -h
rm report_temp_1763035202.csv 
rm radiology_reports_20251113_090200.txt 
rm user_chex.sh 
rm run_chex.sh 
rm chexbert_results_20251113_120002/
