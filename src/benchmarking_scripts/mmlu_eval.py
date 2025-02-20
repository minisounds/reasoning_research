"""
MMLU Evaluation Script

This script evaluates a given language model on the MMLU (Massive Multitask Language Understanding) dataset.
It computes and prints the baseline and steered accuracies based on the model's responses.
"""

import re
import numpy as np
import torch
from tqdm import tqdm
from datasets import load_dataset
from steering_utils import (
    generate_steered_responses_batch,
    generate_baseline_responses_batch,
    device,
    setup_model_and_tokenizer,
)
from src.answer_extractor import mmlu_find_answer_gpt

# Answer mapping for MMLU
ans_map = {
    0: 'A',
    1: 'B',
    2: 'C',
    3: 'D'
}

# Evaluation Function for MMLU
def mmlu_eval(model, tokenizer, dataset, steering_vector, layer, coeff, pos=[0, -1], batch_size=20):
    """
    Evaluate the model on the MMLU dataset.

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
      
    for i in tqdm(range(0, len(dataset['question']), batch_size), desc="Evaluating"):
        questions = dataset['question'][i:i+batch_size]
        choices = dataset['choices'][i:i+batch_size]
        batch_answers = [ans_map[ans_index] for ans_index in dataset['answer'][i:i+batch_size]]
        answers.extend(batch_answers)
        
        # Format prompts with questions and choices
        prompts = []
        for question, choice_list in zip(questions, choices):
            choice_text = " ".join([f"{chr(65+i)}. {choice}" for i, choice in enumerate(choice_list)])
            prompt = f"{question}\n\nChoices:\n{choice_text}\n\nAnswer:"
            prompts.append(prompt)
        
        # Generate responses in batches
        steered_responses = generate_steered_responses_batch(
            model, tokenizer, layer, prompts, steering_vector, coeff, pos)
        baseline_responses = generate_baseline_responses_batch(
            model, tokenizer, prompts, batch_size)
       
        # Extract answers from responses
        extracted_steered_answers = [mmlu_find_answer_gpt(response) for response in steered_responses]
        extracted_baseline_answers = [mmlu_find_answer_gpt(response) for response in baseline_responses]
        
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
    MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.2"  # "meta-llama/Meta-Llama-3-8B-Instruct"
    SEED = 43
    model, tokenizer, config = setup_model_and_tokenizer(MODEL_NAME, seed=SEED, device=device)
    
    # Load MMLU dataset and prepare test set
    mmlu_data = load_dataset("cais/mmlu", "all")
    mmlu_test = mmlu_data['test'].shuffle(seed=SEED)[:1000]  # Using first 1000 examples
    
    # Load the steering vector for the specified layer
    layer = 21
    coeff = 0.5
    steering_vector = np.load(f"steering_vectors/steering_vectors_mistral/steering_vector_layer_{layer}.npy")
    
    # Run evaluation
    baseline_accuracy, steered_accuracy, total = mmlu_eval(
        model, tokenizer, mmlu_test, steering_vector, layer, coeff
    )
    
    # Print evaluation results
    print("Evaluation Results:")
    print(f"Layer: {layer}")
    print(f"Coefficient: {coeff}")
    print(f"Total Samples: {total}")
    print(f"Steered Accuracy: {steered_accuracy:.4f}")
    print(f"Baseline Accuracy: {baseline_accuracy:.4f}")

if __name__ == "__main__":
    main()