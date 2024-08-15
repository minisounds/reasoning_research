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
from steering_utils import generate_steered_responses_batch, generate_baseline_responses_batch, generate_baseline_response, device, set_seed
from evaluate_response import mmlu_find_answer_llama
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

mmlu_data = load_dataset("cais/mmlu", "all")
mmlu_test = mmlu_data['test'].shuffle(seed=42)
print("hello")

ans_map = {
    0: 'A',
    1: 'B',
    2: 'C',
    3: 'D'
}

def mmlu_eval(model, tokenizer, dataset, steering_vector, layer, coeff, pos=[0,-1], batch_size=16):
    steered_correct = 0
    total = 0
    model_steered_answers = []
    answers = []
      
    for i in tqdm(range(0, len(dataset['question']), batch_size), desc="Evaluating"):
        batch = dataset[i:i+batch_size]
        questions = batch['question']
        choices = batch['choices']
        batch_answers = [ans_map[ans_index] for ans_index in batch['answer']]
        answers.extend(batch_answers)
        
        prompts = []
        for question, choice_list in zip(questions, choices):
            choice_text = " ".join([f"{chr(65+i)}. {choice}" for i, choice in enumerate(choice_list)])
            prompt = f"{question}\n\nChoices:\n{choice_text}\n\nAnswer:"
            prompts.append(prompt)
        
        # Generate responses in batches
        steered_responses = generate_steered_responses_batch(model, tokenizer, layer, prompts, steering_vector, coeff, pos, batch_size, seed=42)
       
        # Extract answers from responses
        extracted_steered_answers = [mmlu_find_answer_llama(response, model, tokenizer) for response in steered_responses]
        model_steered_answers.extend(extracted_steered_answers)
        
        # Compare extracted answers with correct answers
        for extracted, answer in zip(extracted_steered_answers, batch_answers):
            if extracted is not None and extracted == answer:     
                steered_correct += 1
            total += 1
        
        if i == 10:
            print("hi!")
        
    steered_accuracy = steered_correct / total
    
    return steered_accuracy, total 

layer = 16
coeff = 25
steering_vector = np.load(f"steering_vectors/steering_vector_layer_{layer}.npy")

steered_accuracy, total = mmlu_eval(model, tokenizer, mmlu_test, steering_vector, layer, coeff)

print(f"Evaluation Results:")
print(f"Layer: {layer}")
print(f"Coefficient: {coeff}")
print(f"Total: {total}")
print(f"Steered Accuracy: {steered_accuracy}")