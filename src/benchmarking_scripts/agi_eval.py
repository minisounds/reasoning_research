"""
AGI Evaluation Script

This script evaluates a given language model on the AGI Eval dataset.
It computes and prints the baseline and steered accuracies based on the model's responses.
"""

import json
import numpy as np
import torch
from tqdm import tqdm
from steering_utils import (
    generate_steered_responses_batch,
    generate_baseline_responses_batch,
    device,
    setup_model_and_tokenizer,
)
from src.answer_extractor import mmlu_find_answer_gpt

# Set up model and tokenizer
MODEL_NAME = "meta-llama/Meta-Llama-3-8B-Instruct"  # "mistralai/Mistral-7B-Instruct-v0.2"
SEED = 43

# Evaluation Function for AGI Eval
def agi_eval(model, tokenizer, dataset, steering_vector, layer, coeff, pos=[0, -1], batch_size=16):
    """
    Evaluate the model on the AGI Eval dataset.

    Parameters:
        model: The language model.
        tokenizer: Tokenizer for the model.
        dataset: List of evaluation samples.
        steering_vector: Numpy array containing the steering vector.
        layer: Layer to apply the steering.
        coeff: Coefficient for steering.
        pos: List indicating positions for answer extraction.
        batch_size: Number of samples to process per batch.
        
    Returns:
        A tuple containing:
            - baseline_accuracy (float)
            - steered_accuracy (float)
            - total (int): Total number of evaluation samples.
    """
    steered_correct = 0
    baseline_correct = 0
    total = 0
    model_steered_answers = []
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
        steered_responses = generate_steered_responses_batch(model, tokenizer, layer, prompts, steering_vector, coeff, pos, seed=42)
        baseline_responses = generate_baseline_responses_batch(model, tokenizer, prompts, seed=43)
       
        # Extract answers from responses
        extracted_steered_answers = [mmlu_find_answer_gpt(response) for response in steered_responses]
        model_steered_answers.extend(extracted_steered_answers)
        extracted_baseline_answers = [mmlu_find_answer_gpt(response) for response in baseline_responses]
        model_baseline_answers.extend(extracted_baseline_answers)
        
        # Compare extracted answers with correct answers
        for extracted_baseline, extracted_steered, answer in zip(extracted_baseline_answers, extracted_steered_answers, batch_answers):
            if extracted_steered is not None and extracted_steered == answer:     
                steered_correct += 1
            if extracted_baseline is not None and extracted_baseline == answer: 
                baseline_correct += 1 
            total += 1
            
    steered_accuracy = steered_correct / total
    baseline_accuracy = baseline_correct / total
    
    return baseline_accuracy, steered_accuracy, total 

def main():
    # Set up the model and tokenizer
    model, tokenizer, config = setup_model_and_tokenizer(MODEL_NAME, seed=SEED, device=device)

    # Load evaluation data from the AGI Eval JSONL file
    agi_eval_data = []
    with open("benchmarks/AGI_Eval.jsonl", "r") as f:
        for line in f:
            agi_eval_data.append(json.loads(line.strip()))

    # Load the steering vector for the specified layer
    layer = 16
    coeff = 20
    steering_vector = np.load(f"steering_vectors/steering_vectors_llama3/steering_vector_layer_{layer}.npy")

    # Run evaluation
    baseline_accuracy, steered_accuracy, total = agi_eval(model, tokenizer, agi_eval_data, steering_vector, layer, coeff)

    # Print evaluation results
    print("Evaluation Results:")
    print(f"Layer: {layer}")
    print(f"Coefficient: {coeff}")
    print(f"Total Samples: {total}")
    print(f"Steered Accuracy: {steered_accuracy:.4f}")
    print(f"Baseline Accuracy: {baseline_accuracy:.4f}")
    
if __name__ == "__main__":
    main()