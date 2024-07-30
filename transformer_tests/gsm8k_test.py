import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from steering_utils import generate_steered_response, device
from tqdm import tqdm
import re

# Load model and tokenizer
model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
model = AutoModelForCausalLM.from_pretrained(model_name)
model = model.to(device)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Load GSM8k dataset
dataset = load_dataset("gsm8k", "main", split="test")

def extract_answer(response):
    # Extract the final answer from the response
    match = re.search(r'The final answer is (\d+)', response) # this depends on how 
    if match:
        return int(match.group(1))
    else:
        return None

def evaluate_gsm8k(model, tokenizer, dataset, layer, coeff, num_samples=100):
    correct = 0
    total = 0

    for sample in tqdm(dataset[:num_samples], desc="Evaluating"):
        question = sample['question']
        answer = sample['answer'].split('####')[1].strip()  # Extract the correct answer
        
        response = generate_steered_response(model, tokenizer, question, layer, coeff)
        extracted_answer = extract_answer(response)
        
        if extracted_answer is not None and str(extracted_answer) == answer:
            correct += 1
        total += 1

    accuracy = correct / total
    return accuracy, correct, total

# Evaluate the model
layer = 20  # You can adjust this
coeff = 5.0  # You can adjust this
accuracy, correct, total = evaluate_gsm8k(model, tokenizer, dataset, layer, coeff)

print(f"Evaluation Results:")
print(f"Layer: {layer}")
print(f"Coefficient: {coeff}")
print(f"Correct: {correct}")
print(f"Total: {total}")
print(f"Accuracy: {accuracy:.2%}")
