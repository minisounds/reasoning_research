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
from steering_utils import get_contrasted_pooled_activations, get_pooled_activations, generate_steered_response_w_vector, generate_baseline_response, device
from evaluate_response import find_answer
from tqdm import tqdm
import re

# Set seeds for reproducibility
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    
set_seed(42)

model = AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")
model = model.to(device)  # Move model to GPU
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")
tokenizer.pad_token = tokenizer.eos_token
tokenizer.pad_token_id = tokenizer.eos_token_id
config = LlamaConfig.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")
config.use_cache = False
    
dataset = load_dataset("gsm8k", "main", split="test")

max_seq_length = 512

training_data = dataset[:50] #TODO: PLAY WITH THIS 
layer = 19
# layer_range = [17, 21] #TODO: PLAY WITH THIS
activations = []

# Initialize an empty PyTorch tensor
concatenated_activation = torch.empty(0).to(device)

w_cot_activations = []
wo_cot_activations = []

for question in tqdm(training_data["question"], desc="Processing Questions: "):
    concatenated_activation = torch.empty(0).to(device)
    w_cot_pooled = get_pooled_activations(model, tokenizer, layer, question, use_cot=True, seed=42)
    w_cot_activations.append(w_cot_pooled)
    wo_cot_pooled = get_pooled_activations(model, tokenizer, layer, question, use_cot=False, seed=42)
    wo_cot_activations.append(wo_cot_pooled)

w_cot_activations = [w_cot_vector.cpu().numpy() for w_cot_vector in w_cot_activations]
wo_cot_activations = [wo_cot_vector.cpu().numpy() for wo_cot_vector in wo_cot_activations]

mean_w_cot = np.mean(w_cot_activations, axis = 0)
mean_wo_cot = np.mean(wo_cot_activations, axis = 0)

steering_vector = mean_w_cot - mean_wo_cot

# PARAMETERS
# question = "Three friends, Alice, Bob, and Charlie, are sitting in a row. Alice is not sitting next to Bob. Bob is sitting to the right of Charlie. Who is sitting in the middle?"
# question = dataset['question'][158]
pos = [0, -1] # TODO: Implement multiple position injections
# coeff = 25 # TODO: Implement decaying coefficient method

# TODO: Implement steered response without layer range
# ex_response = generate_steered_response_w_vector(model, tokenizer, layer, question, steering_vector, coeff, pos, seed=42)
# baseline = generate_baseline_response(model, tokenizer, question, seed=42)
# print(f"steered response: \n {ex_response} \n")
# print(f"baseline response: \n {baseline}")

def evaluate_gsm8k(model, tokenizer, dataset, layer, coeff, num_samples=300):
    correct = 0 
    total = 0
    model_answers = []
    answers = []
    data_split = dataset[50:num_samples]
    
    for i in tqdm(range(len(data_split['question'])), desc="Evaluating"): 
        question = data_split['question'][i]
        answer = data_split['answer'][i].split('####')[1].strip()  # Extract the correct answer
        answers.append(answer)
        
        response = generate_steered_response_w_vector(model, tokenizer, layer, question, steering_vector, coeff, pos, seed=42)
        
        extracted_answer = find_answer(response)
        model_answers.append(extracted_answer)
        
        if extracted_answer is not None and extracted_answer == answer: 
            correct += 1
        total += 1
    
    accuracy = correct / total
    return accuracy, correct, total 

layer = 19 
coeff = 25
accuracy, correct, total = evaluate_gsm8k(model, tokenizer, dataset, layer, coeff)

print(f"Evaluation Results:")
print(f"Layer: {layer}")
print(f"Coefficient: {coeff}")
print(f"Correct: {correct}")
print(f"Total: {total}")
print(f"Accuracy: {accuracy:.2%}")

        
    