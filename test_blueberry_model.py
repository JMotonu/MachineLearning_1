
import os
import json
import torch
import numpy as np

from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModelForCausalLM
)

from be_great import GReaT


# --------------------------------------------------
# 1. Define the model directory
# --------------------------------------------------

model_path = os.path.expanduser(
    "~/.cache/kagglehub/datasets/"
    "inversion/blueberry-yield-fine-tuned-gpt2-medium/"
    "versions/1"
)

checkpoint_path = os.path.join(model_path, "model.pt")
config_path = os.path.join(model_path, "config.json")


# --------------------------------------------------
# 2. Load GReaT training configuration
# --------------------------------------------------

with open(config_path, "r") as file:
    attributes = json.load(file)

print("Model:", attributes["llm"])
print("Conditional column:", attributes["conditional_col"])
print("Dataset columns:", len(attributes["columns"]))


# --------------------------------------------------
# 3. Load GPT-2 Medium tokenizer and configuration
# --------------------------------------------------

print("\nLoading GPT-2 tokenizer and configuration...")

tokenizer = AutoTokenizer.from_pretrained("gpt2-medium")
tokenizer.pad_token = tokenizer.eos_token

hf_config = AutoConfig.from_pretrained("gpt2-medium")
hf_config.pad_token_id = tokenizer.pad_token_id


# --------------------------------------------------
# 4. Create model architecture without pretrained weights
# --------------------------------------------------

print("\nCreating GPT-2 Medium architecture...")

with torch.device("meta"):
    model = AutoModelForCausalLM.from_config(hf_config)


# --------------------------------------------------
# 5. Load your fine-tuned checkpoint
# --------------------------------------------------

print("\nLoading your fine-tuned checkpoint...")

state_dict = torch.load(
    checkpoint_path,
    map_location="cpu",
    weights_only=True,
    mmap=True
)


# Remove obsolete attention-mask buffers
# These are not learned model parameters.

state_dict = {
    key: value
    for key, value in state_dict.items()
    if not (
        key.endswith(".attn.bias")
        or key.endswith(".attn.masked_bias")
    )
}

print("Filtered checkpoint keys:", len(state_dict))

# Load the remaining learned parameters
load_result = model.load_state_dict(
    state_dict,
    assign=True,
    strict=True
)


print("Missing keys:", load_result.missing_keys)
print("Unexpected keys:", load_result.unexpected_keys)

del state_dict

if load_result.missing_keys or load_result.unexpected_keys:
    raise RuntimeError("Checkpoint does not match the GPT-2 architecture.")

model.eval()
print("Fine-tuned model loaded successfully.")


# --------------------------------------------------
# 6. Reconstruct the GReaT object
# --------------------------------------------------

great = GReaT.__new__(GReaT)
great.tokenizer = tokenizer
great.model = model
great.device = torch.device("cpu")

for key, value in attributes.items():
    setattr(great, key, value)

# --------------------------------------------------
# 7. Handle NumPy 2.x compatibility
# --------------------------------------------------

# Older GReaT code uses np.float.
# This local alias avoids changing installed packages.

if not hasattr(np, "float"):
    np.float = float


# --------------------------------------------------
# 8. Generate 1000 synthetic record
# --------------------------------------------------

print("\nGenerating 1000 synthetic blueberry-yield records...")

synthetic_data = great.sample(
    n_samples=1000,
    temperature=0.7,
    k=1,
    max_length=256,
    device="mps"
)

print("\nSYNTHETIC BLUEBERRY DATA")
print(synthetic_data.head(10).to_string(index=False))

print("\nDataset shape:", synthetic_data.shape)

# Save generated records
output_path = "synthetic_blueberry_data.csv"
synthetic_data.to_csv(output_path, index=False)

print("\nDataset saved to:", output_path)