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
from steering_utils import get_pooled_activations, generate_steered_response_w_vector, generate_steered_responses_batch, generate_baseline_responses_batch, generate_baseline_response, device, set_seed
from evaluate_response import find_answer
from tqdm import tqdm
import re
 
set_seed(42)

# model = AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")
model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")
model = model.to(device)  # Move model to GPU
# tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")
tokenizer.pad_token = tokenizer.eos_token
tokenizer.pad_token_id = tokenizer.eos_token_id
# config = LlamaConfig.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")
# config.use_cache = False

dataset = load_dataset("gsm8k", "main")

def evaluate_mean_mass(model, tokenizer, dataset, steering_vector, layer, coeff, pos=[0,-1], batch_size=16):
    #steered_correct = 0 
    baseline_correct = 0 
    total = 0
    answers = []
    #model_steered_answers = []
    model_baseline_answers = []

    data_split = dataset['test']

    for i in tqdm(range(0, len(data_split['question']), batch_size), desc="Evaluating"): 
        batch = data_split[i:i+batch_size]
        questions = batch['question']
        batch_answers = [answer.split('####')[1].strip() for answer in  batch['answer']]
        answers.extend(batch_answers)

        # Generate responses in batches
        #steered_responses = generate_steered_responses_batch(model, tokenizer, layer, questions, steering_vector, coeff, pos, batch_size, seed=42)
        baseline_responses = generate_baseline_responses_batch(model, tokenizer, questions, batch_size, seed=42)

        # Extract answers from responses
        # extracted_steered_answers = [find_answer(response) for response in steered_responses]
        # model_steered_answers.extend(extracted_steered_answers)
        
        extracted_baseline_answers = [find_answer(response) for response in baseline_responses]
        model_baseline_answers.extend(extracted_baseline_answers)
        
        for extracted, answer in zip(extracted_baseline_answers, batch_answers):
            if extracted is not None and extracted == answer:     
                baseline_correct += 1
            total += 1
    
    baseline_accuracy = baseline_correct / total
    return baseline_accuracy, total

layer = 16
coeff = 50
steering_vector = np.load(f'steering_vectors/steering_vector_layer_{layer}.npy')
baseline_accuracy, total = evaluate_mean_mass(model, tokenizer, dataset, steering_vector, layer, coeff)

print(f"Evaluation Results:")
print(f"Layer: {layer}")
print(f"Coefficient: {coeff}")
print(f"Correct: {correct}")
print(f"Total: {total}")
print(f"Accuracy: {baseline_accuracy:.2%}")