import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    LlamaConfig,
    LlamaForCausalLM,
    GPT2LMHeadModel,
)
from datasets import load_dataset
from steering_utils import generate_steered_response, generate_baseline_response, device
from tqdm import tqdm
from evaluate_response import find_answer
import re

# Load model and tokenizer
model = AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")
model = model.to(device)  # Move model to GPU
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")
tokenizer.pad_token = tokenizer.eos_token
tokenizer.pad_token_id = tokenizer.eos_token_id
config = LlamaConfig.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")
config.use_cache = False

# Load GSM8k dataset
dataset = load_dataset("gsm8k", "main", split="test")

def extract_answer(response):
    # Extract the final answer from the response
    match = re.search(r'The answer is (\d+)', response) # this depends on how 
    if match:
        return int(match.group(1))
    else:
        return None

def evaluate_gsm8k(model, tokenizer, dataset, layer, coeff, num_samples=100):
    correct = 0
    total = 0

    model_answers = []
    answers = []
    data_split = dataset[:num_samples]
    for i in tqdm(range(len(data_split['question'])), desc="Evaluating"): 
        question = data_split['question'][i]
        answer = data_split['answer'][i].split('####')[1].strip()  # Extract the correct answer
        answers.append(answer)
        
        
        response = generate_steered_response(model, tokenizer, question, layer, coeff)
        # response = generate_baseline_response(model, tokenizer, question)
        extracted_answer = find_answer(response)
        # extracted_answer = extract_answer(response)
        model_answers.append(extracted_answer)
        
        if extracted_answer is not None and extracted_answer == answer:
            correct += 1
        total += 1
    
    accuracy = correct / total
    return accuracy, correct, total

# Evaluate the model
layer = 19  # You can adjust this
coeff = 4  # You can adjust this
accuracy, correct, total = evaluate_gsm8k(model, tokenizer, dataset, layer, coeff)

print(f"Evaluation Results:")
print(f"Layer: {layer}")
print(f"Coefficient: {coeff}")
print(f"Correct: {correct}")
print(f"Total: {total}")
print(f"Accuracy: {accuracy:.2%}")
