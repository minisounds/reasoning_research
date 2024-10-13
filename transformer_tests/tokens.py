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
from steering_utils import generate_steered_responses_batch, generate_baseline_responses_batch, device, set_seed
from tqdm import tqdm
import re
 
set_seed(42)

model = AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")
# model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")
model = model.to(device)  # Move model to GPU
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")
# tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")
tokenizer.pad_token = tokenizer.eos_token
tokenizer.pad_token_id = tokenizer.eos_token_id
config = LlamaConfig.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")
config.use_cache = False

dataset = load_dataset("gsm8k", "main")

def evaluate_mean_mass(model, tokenizer, dataset, steering_vector, layer, coeff, pos=[0,-1], batch_size=16):
    answers = []
    model_total_steered_tokens = []
    model_total_baseline_tokens = []

    data_split = dataset['test'][:300]

    for i in tqdm(range(0, len(data_split['question']), batch_size), desc="Evaluating"): 
        # batch = data_split[i:i+batch_size]
        questions = data_split['question'][i:i+batch_size]
        batch_answers = [answer.split('####')[1].strip() for answer in  data_split['answer'][i:i+batch_size]]
        answers.extend(batch_answers)

        # Generate responses in batches
        steered_responses = generate_steered_responses_batch(model, tokenizer, layer, questions, steering_vector, coeff, pos, batch_size, seed=42)
        baseline_responses = generate_baseline_responses_batch(model, tokenizer, questions, batch_size, seed=42)

        # Extract answers from responses
        for steered_response, baseline_response in zip(steered_responses, baseline_responses):
            model_total_steered_tokens.append(len(tokenizer(steered_response)['input_ids']))
            model_total_baseline_tokens.append(len(tokenizer(baseline_response)['input_ids']))
        
        print("hello")
        
    avg_steered_len = sum(model_total_steered_tokens) / len(model_total_steered_tokens)
    avg_baseline_len = sum(model_total_baseline_tokens) / len(model_total_baseline_tokens)
        
    return avg_steered_len, avg_baseline_len

layer = 16
coeff = 50
steering_vector = np.load(f'steering_vectors/steering_vectors_llama3/steering_vector_layer_{layer}.npy')
avg_steered_len, avg_baseline_len = evaluate_mean_mass(model, tokenizer, dataset, steering_vector, layer, coeff)

print(f"steered: {avg_steered_len}")
print(f"baseline: {avg_baseline_len}")