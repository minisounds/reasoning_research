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
from steering_utils import generate_steered_responses_batch, generate_baseline_responses_batch, device, set_seed, write_to_json
from src.answer_extractor import find_answer
from tqdm import tqdm
import json
import re
    
set_seed(43)

# model = AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")
model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")
model = model.to(device)  # Move model to GPU
# tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")
tokenizer.pad_token = tokenizer.eos_token
tokenizer.pad_token_id = tokenizer.eos_token_id
# config = LlamaConfig.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")
# config.use_cache = False

# Grid Search for Best Parameter Combinations - GSM8k
dataset = load_dataset("gsm8k", "main")

def evaluate_mean_mass(model, tokenizer, dataset, steering_vector, layer, coeff, pos=[0,-1], batch_size=20):
    steered_correct = 0
    total = 0
    model_steered_answers = []
    answers = []
    
    data_split = dataset['test'][300:370]
     
    for i in tqdm(range(0, len(data_split['question']), batch_size), desc="Evaluating"):
        questions = data_split['question'][i:i+batch_size]
        batch_answers = [answer.split('####')[1].strip() for answer in data_split['answer'][i:i+batch_size]]
        answers.extend(batch_answers)
        
        # Generate responses in batches
        steered_responses = generate_steered_responses_batch(model, tokenizer, layer, questions, steering_vector, coeff, pos, batch_size, seed=43)
        
        # Extract answers from responses
        extracted_steered_answers = [find_answer(response) for response in steered_responses]
        model_steered_answers.extend(extracted_steered_answers)
        
        # Compare extracted answers with correct answers
        for extracted, answer in zip(extracted_steered_answers, batch_answers):
            if extracted is not None and extracted == answer:     
                steered_correct += 1
            total += 1
        
    steered_accuracy = steered_correct / total
    
    return steered_accuracy, total 

results = []
for layer in range(20, 25):
    for coeff in np.arange(0.5, 1.6, 0.5):
        steering_vector = np.load(f"steering_vectors/steering_vectors_mistral/steering_vector_layer_{layer}.npy")
        steered_accuracy, total = evaluate_mean_mass(model, tokenizer, dataset, steering_vector, layer, coeff)

        print(f"Evaluation Results:")
        print(f"Layer: {layer}")
        print(f"Coefficient: {coeff}")
        print(f"Total: {total}")
        print(f"Steered Accuracy: {steered_accuracy}")
        
        result = {
            "layer": layer,
            "coefficient": coeff,
            "steered_accuracy": steered_accuracy
        }
        results.append(result)

with open('steering_results.json', 'w') as f:
    json.dump(results, f, indent=2)

print("finished with grid search eval")