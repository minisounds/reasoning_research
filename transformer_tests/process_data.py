import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaConfig, LlamaForCausalLM
from datasets import load_dataset
from tqdm import tqdm
import numpy as np
from steering_utils import get_pooled_activations, generate_baseline_response, device, set_seed
from evaluate_response import find_answer

# MODEL & TOKENIZER & SEED SET UP: 
set_seed(42)
model = AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")
model = model.to(device)  # Move model to GPU
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")
tokenizer.pad_token = tokenizer.eos_token
tokenizer.pad_token_id = tokenizer.eos_token_id
config = LlamaConfig.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")
config.use_cache = False

# HELPER FUNC:
def get_mean_mass_steering_vector(w_cot_activations, wo_cot_activations): 
    w_cot_activations = [w_cot_vector.cpu().numpy() for w_cot_vector in w_cot_activations]
    wo_cot_activations = [wo_cot_vector.cpu().numpy() for wo_cot_vector in wo_cot_activations]
    
    mean_w_cot = np.mean(w_cot_activations, axis=0)
    mean_wo_cot = np.mean(wo_cot_activations, axis=0)
    
    steering_vector = mean_w_cot - mean_wo_cot
    return steering_vector
    
# ACTIVATIONS & HYPERPARAMETERS
w_cot_activations = []
wo_cot_activations = [] 
layer = 19

# DATASETS

# MMLU - Pull 2 Questions from Each Subsection (from dev) - total of 125 questions
mmlu_data = load_dataset("cais/mmlu", "all")
count = 0
for i in tqdm(range(len(mmlu_data['dev']))):
    # iterate twice, then add 3 to index
    if count < 2: 
        w_cot_vector, wo_cot_vector = get_pooled_activations(model, tokenizer, layer, mmlu_data['dev']['question'][i], seed=42)
        w_cot_activations.append(w_cot_vector)
        wo_cot_activations.append(wo_cot_vector)
        count += 1
    else: 
        count = 0
        i += 3

print("mmlu activations completed")

# CREATE STEERING VECTOR 
steering_vector = get_mean_mass_steering_vector(w_cot_activations, wo_cot_activations)
# SAVE STEERING VECTOR 
np.save('steering_vector_layer_19.npy', steering_vector) 


