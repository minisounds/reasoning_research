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
from evaluate_response import modeLing_find_answer
from tqdm import tqdm
import json
import re

set_seed(42)

model = AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")
model = model.to(device)  # Move model to GPU
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")
tokenizer.pad_token = tokenizer.eos_token
tokenizer.pad_token_id = tokenizer.eos_token_id
config = LlamaConfig.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")
config.use_cache = False

with open("benchmarks/modeLing.json", "r") as f: 
    data = json.load(f)

# set up dataset here
dataset = {}
dataset['questions'] = []
dataset['answers'] = []
for question in data['problems']:
    examples = "\n".join(question['data'])
    test_qs = "\n".join(question['questions'])
    prompt = f"""Here are some expressions in Language (a neverseen-before foreign language) and their translations in English:
    Language: {question['name']}
    {examples}
    Given the above examples, please translate the following statements (translating {question['name']} to English, English to {question['name']})
    {test_qs}"""
    
    answer = question['answers']
    dataset['questions'].append(prompt)
    dataset['answers'].append(answer)

def modeling_eval(model, tokenizer, dataset, steering_vector, layer, coeff, pos=[0,-1], batch_size=1):
    steered_correct = 0
    total = 0
    model_steered_answers = []
    answers = []
    
    for i in tqdm(range(0, len(dataset['questions']), batch_size), desc="Evaluating"):
        # batch = dataset[i:i+batch_size]
        questions = dataset['questions'][i:i+batch_size]
        answers = dataset['answers'][i:i+batch_size]
        
        # Generate responses in batches
        steered_responses = generate_steered_responses_batch(model, tokenizer, layer, questions, steering_vector, coeff, pos, batch_size, seed=42)
        baseline_responses = generate_baseline_responses_batch(model, tokenizer, questions, batch_size, seed=42)
       
        # Extract answers from responses
        extracted_steered_answers = [modeLing_find_answer(response) for response in steered_responses]
        extracted_baseline_answers = [modeLing_find_answer(response) for response in baseline_responses]
        model_steered_answers.extend(extracted_steered_answers)
        
        # Compare extracted answers with correct answers
        for extracted, answer in zip(extracted_steered_answers, batch_answers):
            if extracted is not None and extracted == answer:     
                steered_correct += 1
            total += 1
        
    steered_accuracy = steered_correct / total
    
    return steered_accuracy, total

layer = 16
coeff = 25
steering_vector = np.load(f"steering_vectors/steering_vector_layer_{layer}.npy")

steered_accuracy, total = modeling_eval(model, tokenizer, dataset, steering_vector, layer, coeff)

print(f"Evaluation Results:")
print(f"Layer: {layer}")
print(f"Coefficient: {coeff}")
print(f"Total: {total}")
print(f"Steered Accuracy: {steered_accuracy}")