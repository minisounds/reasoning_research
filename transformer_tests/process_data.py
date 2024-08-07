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
# get lists of pooled activations from questions in dataset
def get_pooled_activations(model, tokenizer, data, layer): 
    w_cot_activations = []
    wo_cot_activations = []
    
    for question in data['question']: 
        w_cot_pooled = get_pooled_activations(model, tokenizer, layer, question, use_cot=True, seed=42)
        w_cot_activations.append(w_cot_pooled)
        wo_cot_pooled = get_pooled_activations(model, tokenizer, layer, question, use_cot=False, seed=42)
        wo_cot_activations.append(wo_cot_pooled)
    
    return w_cot_activations, wo_cot_activations
    

# IMPORT DATASETS AND WORK WITH EACH 


