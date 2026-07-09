import os

os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import time
import torch

# Hide MPS so llmcompressor's dispatcher falls back to CPU on Apple Silicon
torch.backends.mps.is_available = lambda: False
torch.backends.mps.is_built = lambda: False

from transformers import DistilBertTokenizer, DistilBertModel
from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier
from datasets import Dataset
from transformers import BitsAndBytesConfig
import platform

# llm-compressor has a known MPS bug on Apple Silicon
# Force CPU for quantization — inference can still use MPS after
if platform.system() == "Darwin" and torch.backends.mps.is_available():
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
    quantization_device = "cpu"
else:
    quantization_device = get_best_device()

tokenizer = DistilBertTokenizer.from_pretrained(
    "../models/distilbert-base-uncased"
)

model = DistilBertModel.from_pretrained(
    "../models/distilbert-base-uncased",
    device_map="cpu",
)

model.eval()

# total_size = model.num_parameters() * 4 / (1024 ** 2)
total_size = sum(
    p.numel() * p.element_size() for p in model.parameters()
) / (1024 ** 2)

print(f"Model Size: {total_size:.2f} MB")

inputs = tokenizer(
    "Hello, this is a sentence",
    return_tensors="pt",
)

start = time.time()

with torch.no_grad():
    output = model(**inputs)

end = time.time()

elapsed = (end - start) * 1000

print(f"Inference time: {elapsed:.2f} ms")

# calibration_data = [
#     "The bank processes thousands of transactions daily.",
#     "Machine learning models require careful validation.",
#     "Customer data must be handled securely.",
#     "The model predicts outcomes based on input features.",
#     "Financial services require high availability systems.",
# ]
#
# dataset = Dataset.from_dict({"text": calibration_data})
#
# model.to("cpu")
#
# recipe = QuantizationModifier(targets="Linear", scheme="W8A8")
#
# oneshot(
#     model=model,
#     dataset=dataset,
#     recipe=recipe,
#     num_calibration_samples=5,
#     output_dir="../models/distilbert-base-uncased-quantized",
# )

quantization_config = BitsAndBytesConfig(load_in_8bit=True)

model1 = DistilBertModel.from_pretrained(
    "../models/distilbert-base-uncased",
    quantization_config=quantization_config,
)
model1.eval()

model1.save_pretrained("../models/distilbert-base-uncased-quantized")

tokenizer = DistilBertTokenizer.from_pretrained(
    "../models/distilbert-base-uncased"
)

model1 = DistilBertModel.from_pretrained(
    "../models/distilbert-base-uncased-quantized"
)
model1.eval()

total_size1 = sum(
    p.numel() * p.element_size() for p in model1.parameters()
) / (1024 ** 2)

print(f"Quantized Model Size: {total_size1:.2f} MB")

inputs1 = tokenizer(
    "Hello, this is a sentence",
    return_tensors="pt",
)

start1 = time.time()

with torch.no_grad():
    output1 = model1(**inputs1)

end1 = time.time()

elapsed1 = (end1 - start1) * 1000

print(f"Quantized Inference time: {elapsed1:.2f} ms")