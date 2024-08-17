import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    LlamaConfig,
    LlamaForCausalLM,
)
from datasets import load_dataset
import numpy as np
from steering_utils import generate_steered_responses_batch, generate_baseline_responses_batch, generate_baseline_response, device, set_seed
from evaluate_response import mmlu_find_answer_gpt
from tqdm import tqdm
import re
import json

set_seed(42)

model = AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")
model = model.to(device)  # Move model to GPU
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")
tokenizer.pad_token = tokenizer.eos_token
tokenizer.pad_token_id = tokenizer.eos_token_id
config = LlamaConfig.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")
config.use_cache = False

agi_eval_data = []

with open("benchmarks/AGI_Eval.jsonl", "r") as f:
    for line in f:
        agi_eval_data.append(json.loads(line.strip()))

def agi_eval(model, tokenizer, dataset, steering_vector, layer, coeff, pos=[0,-1], batch_size=16):
    # steered_correct = 0
    baseline_correct = 0
    total = 0
    # model_steered_answers = []
    model_baseline_answers = []
    answers = []
      
    for i in tqdm(range(0, len(dataset), batch_size), desc="Evaluating"):
        batch = dataset[i:i+batch_size]
        questions = [q['question'] for q in batch]
        choices = ['\n'.join(q['options']) for q in batch]
        batch_answers = [q['label'] for q in batch]
        answers.extend(batch_answers)
        
        prompts = []
        for question, choice_list in zip(questions, choices):
            prompt = f"{question}\n\nChoices:\n{choice_list}\n\nAnswer:"
            prompts.append(prompt)
        
        # Generate responses in batches
        # steered_responses = generate_steered_responses_batch(model, tokenizer, layer, prompts, steering_vector, coeff, pos, batch_size, seed=42)
        baseline_responses = generate_baseline_responses_batch(model, tokenizer, prompts, batch_size, seed=42)
       
        # Extract answers from responses
        # extracted_steered_answers = [mmlu_find_answer_gpt(response) for response in steered_responses]
        # model_steered_answers.extend(extracted_steered_answers)
        extracted_baseline_answers = [mmlu_find_answer_gpt(response) for response in baseline_responses]
        model_baseline_answers.extend(extracted_baseline_answers)
        
        # Compare extracted answers with correct answers
        # for extracted, answer in zip(extracted_steered_answers, batch_answers):
        #     if extracted is not None and extracted == answer:     
        #         steered_correct += 1
        #     total += 1
        
        for extracted, answer in zip(extracted_baseline_answers, batch_answers):
            if extracted is not None and extracted == answer:     
                baseline_correct += 1
            total += 1
            
    # steered_accuracy = steered_correct / total
    baseline_accuracy = baseline_correct / total
    
    return baseline_accuracy, total 

layer = 16
coeff = 20
steering_vector = np.load(f"steering_vectors/steering_vector_layer_{layer}.npy")

baseline_accuracy, total = agi_eval(model, tokenizer, agi_eval_data, steering_vector, layer, coeff)
# TODO: Do the same for Baseline
print(f"Evaluation Results:")
print(f"Layer: {layer}")
print(f"Coefficient: {coeff}")
print(f"Total: {total}")
# print(f"Steered Accuracy: {steered_accuracy}")
print(f"Baseline Accuracy: {baseline_accuracy}")