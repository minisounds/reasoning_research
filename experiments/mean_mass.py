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
from steering_utils import get_contrasted_pooled_activations, get_pooled_activations, generate_steered_response_w_vector, generate_baseline_response, device, set_seed
from evaluate_response import find_answer
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
    
dataset = load_dataset("gsm8k", "main", split="train")

max_seq_length = 512

# training_data = dataset[:50] #TODO: PLAY WITH THIS 
layer = 19
# activations = []

# w_cot_activations = []
# wo_cot_activations = []

# for question in tqdm(training_data["question"], desc="Processing Questions: "):
#     w_cot_pooled = get_pooled_activations(model, tokenizer, layer, question, use_cot=True, seed=42)
#     w_cot_activations.append(w_cot_pooled)
#     wo_cot_pooled = get_pooled_activations(model, tokenizer, layer, question, use_cot=False, seed=42)
#     wo_cot_activations.append(wo_cot_pooled)

# w_cot_activations = [w_cot_vector.cpu().numpy() for w_cot_vector in w_cot_activations]
# wo_cot_activations = [wo_cot_vector.cpu().numpy() for wo_cot_vector in wo_cot_activations]

# mean_w_cot = np.mean(w_cot_activations, axis = 0)
# mean_wo_cot = np.mean(wo_cot_activations, axis = 0)

# steering_vector = mean_w_cot - mean_wo_cot

# PARAMETERS
question = "Three friends, Alice, Bob, and Charlie, are sitting in a row. Alice is not sitting next to Bob. Bob is sitting to the right of Charlie. Who is sitting in the middle?"
# question = dataset['question'][158]
pos = [0, -1] # TODO: Implement multiple position injections
coeff = 15 # TODO: Implement decaying coefficient method (is this even the best coeff, or layer?)
steering_vector = np.load('steering_vector_v1.npy')

# TODO: Implement steered response without layer range
ex_response = generate_steered_response_w_vector(model, tokenizer, layer, question, steering_vector, coeff, pos, seed=42)
baseline = generate_baseline_response(model, tokenizer, question, seed=42)
print(f"steered response: \n {ex_response} \n")
print(f"baseline response: \n {baseline}")