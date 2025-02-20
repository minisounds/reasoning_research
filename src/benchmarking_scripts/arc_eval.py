"""
ARC Evaluation Script

This script evaluates a given language model on the ARC Eval dataset.
It computes and prints the baseline and steered accuracies based on the model's responses.
"""

import torch
from datasets import load_dataset
import numpy as np
from steering_utils import (
    generate_steered_responses_batch,
    generate_baseline_responses_batch,
    device,
    setup_model_and_tokenizer
)
from src.answer_extractor import mmlu_find_answer_gpt
from tqdm import tqdm

def arc_eval(model, tokenizer, dataset, steering_vector, layer, coeff, pos=[0, -1], batch_size=16):
    """
    Evaluate a language model on the ARC dataset by computing both the steered and baseline accuracies.

    Args:
        model: The language model.
        tokenizer: Tokenizer for the language model.
        dataset: The ARC dataset
        steering_vector: Numpy array used to steer the model's responses.
        layer (int): The model layer at which to apply the steering.
        coeff (float): Coefficient to control the steering intensity.
        pos (list): Positions for answer extraction modifications.
        batch_size (int): Number of samples to process per batch.

    Returns:
        A tuple (baseline_accuracy, steered_accuracy, total_samples):
            baseline_accuracy (float): Accuracy of the baseline (unmodified) model outputs.
            steered_accuracy (float): Accuracy of the steered model outputs.
            total_samples (int): Total number of evaluated samples.
    """
    steered_correct = 0
    baseline_correct = 0
    total_samples = 0

    # Process dataset in batches
    for start in tqdm(range(0, len(dataset['question']), batch_size), desc="Evaluating"):
        batch = dataset[start:start+batch_size]
        questions = batch['question']
        # Each element of batch['choices'] is expected to be a list of answer option dictionaries.
        choices_lists = [item["text"] for item in batch["choices"]]
        batch_answers = batch["answerKey"]

        # Build prompts for each sample
        prompts = []
        for question, choice_list in zip(questions, choices_lists):
            formatted_choices = " ".join([f"{chr(65+j)}. {choice}" for j, choice in enumerate(choice_list)])
            prompt = f"{question}\n\nChoices:\n{formatted_choices}\n\nAnswer:"
            prompts.append(prompt)

        # Generate steered responses
        steered_responses = generate_steered_responses_batch(
            model, tokenizer, layer, prompts, steering_vector, coeff, pos, batch_size, seed=43
        )
        extracted_steered = [mmlu_find_answer_gpt(response) for response in steered_responses]

        # Generate baseline responses
        baseline_responses = generate_baseline_responses_batch(
            model, tokenizer, prompts, batch_size, seed=43
        )
        extracted_baseline = [mmlu_find_answer_gpt(response) for response in baseline_responses]

        # Compare extracted answers with the correct answers
        for steered_ans, baseline_ans, correct_ans in zip(extracted_steered, extracted_baseline, batch_answers):
            if steered_ans is not None and steered_ans == correct_ans:
                steered_correct += 1
            if baseline_ans is not None and baseline_ans == correct_ans:
                baseline_correct += 1
            total_samples += 1

    steered_accuracy = steered_correct / total_samples
    baseline_accuracy = baseline_correct / total_samples

    return baseline_accuracy, steered_accuracy, total_samples

def main(): 
    # Define Parameters
    MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.2"  # "meta-llama/Meta-Llama-3-8B-Instruct"
    SEED = 43

    # Set up the model and tokenizer
    model, tokenizer, config = setup_model_and_tokenizer(MODEL_NAME, seed=SEED, device=device)
    
    # Load the ARC dataset
    arc_data = load_dataset("allenai/ai2_arc", "ARC-Challenge")
    arc_test = arc_data['test']
    
    # Evaluation configuration
    layer = 13
    coeff = 1
    steering_vector = np.load(f"steering_vectors/steering_vectors_mistral/steering_vector_layer_{layer}.npy")
    
    # Run evaluation on the test set
    baseline_accuracy, steered_accuracy, total = arc_eval(model, tokenizer, arc_test, steering_vector, layer, coeff)
    
    # Display evaluation results
    print("Evaluation Results:")
    print(f"Layer: {layer}")
    print(f"Coefficient: {coeff}")
    print(f"Total Samples: {total}")
    print(f"Steered Accuracy: {steered_accuracy:.4f}")
    print(f"Baseline Accuracy: {baseline_accuracy:.4f}")

if __name__ == "__main__":
    main()