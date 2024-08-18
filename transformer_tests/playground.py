import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    LlamaConfig,
    LlamaForCausalLM,
    GPT2LMHeadModel,
)
from datasets import load_dataset
import numpy as np
from steering_utils import generate_steered_response_w_vector, generate_baseline_responses_batch, generate_baseline_response, device, set_seed
from tqdm import tqdm
import re

set_seed(42)

model = AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")
model = model.to(device)  # Move model to GPU
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")
tokenizer.pad_token = tokenizer.eos_token
tokenizer.pad_token_id = tokenizer.eos_token_id
config = LlamaConfig.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")
config.use_cache = False

# dataset 
data = load_dataset("gsm8k", "main")
question = data['test'][191]['question']

layer = 16
coeff = 20
steering_vector = np.load(f"steering_vectors/steering_vector_layer_{layer}.npy")
pos = [0,-1]

steered_response = generate_steered_response_w_vector(model, tokenizer, layer, question, steering_vector, coeff, pos, seed=42)
baseline_response = generate_baseline_response(model, tokenizer, question, seed=42)

print(f"Question: {question}\n")
print(f"Steered Response: {steered_response}\n")
print(f"Baseline Response: {baseline_response}")
print("Done")