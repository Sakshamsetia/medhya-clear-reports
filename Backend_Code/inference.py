from transformers import VisionEncoderDecoderModel, AutoTokenizer, AutoFeatureExtractor
from PIL import Image
import torch

# Load model and tokenizer
model = VisionEncoderDecoderModel.from_pretrained("Molkaatb/ChestX").to("cuda")
tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token
feature_extractor = AutoFeatureExtractor.from_pretrained("google/vit-base-patch16-224-in21k")

# Example image
image = Image.open("/home/chipmonkx86/inputIMG/input.jpg").convert("RGB")
inputs = feature_extractor(images=image, return_tensors="pt").pixel_values.to("cuda")

# Generate report
outputs = model.generate(inputs, max_length=512, num_beams=4)
report = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(report)
