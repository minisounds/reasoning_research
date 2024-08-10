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
from steering_utils import get_pooled_activations, generate_steered_response_w_vector, generate_steered_responses_batch, generate_baseline_responses_batch, generate_baseline_response, device, set_seed, write_to_json
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
    
dataset = load_dataset("gsm8k", "main")
# dataset = load_dataset("maveriq/bigbenchhard", "boolean_expressions")

steering_vector = np.load('steering_vector_v1.npy')
def evaluate_mean_mass(model, tokenizer, dataset, steering_vector, layer, coeff, pos=[0,-1], batch_size=8):
    steered_correct = 0
    baseline_correct = 0
    total = 0
    model_steered_answers = []
    model_baseline_answers = []
    answers = []
    
    data_split = dataset['test']
    # data_split = dataset[0:100]
    
    for i in tqdm(range(0, len(data_split['question']), batch_size), desc="Evaluating"): 
        batch = data_split[i:i+batch_size]
        questions = batch['question']
        batch_answers = [answer.split('####')[1].strip() for answer in batch['answer']]
        answers.extend(batch_answers)
        
        # Generate responses in batches
        steered_responses = generate_steered_responses_batch(model, tokenizer, layer, questions, steering_vector, coeff, pos, batch_size, seed=42)
        baseline_responses = generate_baseline_responses_batch(model, tokenizer, questions, seed=42)
        
        # Extract answers from responses
        extracted_steered_answers = [find_answer(response) for response in steered_responses]
        model_steered_answers.extend(extracted_steered_answers)
        
        extracted_baseline_answers = [find_answer(response) for response in baseline_responses]
        model_baseline_answers.extend(extracted_baseline_answers)
        
        # Compare extracted answers with correct answers
        for extracted, answer in zip(extracted_steered_answers, batch_answers):
            if extracted is not None and extracted == answer:     
                steered_correct += 1
            total += 1
        
        for extracted, answer in zip(extracted_baseline_answers, batch_answers):
            if extracted is not None and extracted == answer:     
                baseline_correct += 1
        
    steered_accuracy = steered_correct / total
    baseline_accuracy = baseline_correct / total
    
    return steered_accuracy, baseline_accuracy, total 

layer = 19 
coeff = 20
steered_accuracy, baseline_accuracy, total = evaluate_mean_mass(model, tokenizer, dataset, steering_vector, layer, coeff)

print(f"Evaluation Results:")
print(f"Layer: {layer}")
print(f"Coefficient: {coeff}")
print(f"Total: {total}")
print(f"Steered Accuracy: {steered_accuracy}")
print(f"Baseline Accuracy: {baseline_accuracy}")