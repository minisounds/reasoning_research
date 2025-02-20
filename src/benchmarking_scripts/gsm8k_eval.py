"""
GSM8K Evaluation Script

This script evaluates a given language model on the GSM8K dataset.
It computes and prints the baseline and steered accuracies based on the model's responses to math problems.
"""

import torch
import numpy as np
from tqdm import tqdm
from datasets import load_dataset
from steering_utils import (
    generate_steered_responses_batch,
    generate_baseline_responses_batch,
    device,
    setup_model_and_tokenizer
)
from src.answer_extractor import find_answer

# Evaluation Function for GSM8K
def gsm8k_eval(model, tokenizer, dataset, steering_vector, layer, coeff, pos=[0, -1], batch_size=16):
    """
    Evaluate the model on the GSM8K dataset.

    Parameters:
        model: The language model.
        tokenizer: Tokenizer for the model.
        dataset: Dataset containing evaluation samples.
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
    answers = []
    
    data_split = dataset['test']

    for i in tqdm(range(0, len(data_split['question']), batch_size), desc="Evaluating"):
        batch = data_split[i:i+batch_size]
        questions = batch['question']
        batch_answers = [answer.split('####')[1].strip() for answer in batch['answer']]
        answers.extend(batch_answers)

        # Generate responses in batches
        steered_responses = generate_steered_responses_batch(model, tokenizer, layer, questions, steering_vector, coeff, pos)
        baseline_responses = generate_baseline_responses_batch(model, tokenizer, questions)

        # Extract answers from responses
        extracted_steered_answers = [find_answer(response) for response in steered_responses]
        extracted_baseline_answers = [find_answer(response) for response in baseline_responses]
        
        # Compare extracted answers with correct answers
        for extracted_steered, extracted_baseline, answer in zip(
            extracted_steered_answers, extracted_baseline_answers, batch_answers
        ):
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
    MODEL_NAME = "meta-llama/Meta-Llama-3-8B-Instruct"  # "mistralai/Mistral-7B-Instruct-v0.2"
    SEED = 43
    model, tokenizer, config = setup_model_and_tokenizer(MODEL_NAME, seed=SEED, device=device)
    
    # Load the GSM8K dataset
    dataset = load_dataset("gsm8k", "main")
    
    # Load the steering vector for the specified layer
    layer = 16
    coeff = 20
    steering_vector = np.load(f"steering_vectors/steering_vectors_llama3/steering_vector_layer_{layer}.npy")
    
    baseline_accuracy, steered_accuracy, total = gsm8k_eval(model, tokenizer, dataset, steering_vector, layer, coeff)
    
    # Print evaluation results
    print("Evaluation Results:")
    print(f"Layer: {layer}")
    print(f"Coefficient: {coeff}")
    print(f"Total Samples: {total}")
    print(f"Steered Accuracy: {steered_accuracy:.4f}")
    print(f"Baseline Accuracy: {baseline_accuracy:.4f}")

if __name__ == "__main__":
    main()