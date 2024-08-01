import json
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    LlamaConfig,
    LlamaForCausalLM,
)
from datasets import load_dataset
from tqdm import tqdm
 
training_data = load_dataset("gsm8k", "main", split="train")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def generate_responses_batch(model, tokenizer, prompts, use_cot, max_length = 512):
    inputs = tokenizer(prompts, return_tensors="pt", max_length = max_length, truncation=True, padding=True) 
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=300,
            num_return_sequences=1,
            do_sample=True,
            temperature=0.7 if use_cot else 0.3 # higher to enable ease of CoT and lower for more immediate answering
        )
    return tokenizer.batch_decode(outputs, skip_special_tokens=True)

def create_gsm8k_dataset(output_file, model_name="meta-llama/Meta-Llama-3-8B-Instruct", batch_size=10):
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype = torch.float16)
    model = model.to(device)  # Move model to GPU
    model = torch.compile(model) 
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    config = LlamaConfig.from_pretrained(model_name)
    config.use_cache = False
    
    processed_data = []
    iteration = 0 
    
    for i in tqdm(range(0, 3500, batch_size), desc="Processing GSM8k"):
        batch = training_data[i:i+batch_size]
        questions = batch['question']
        
        cot_system_prompt = "You are a helpful AI Assistant who answers questions step by step."
        wo_cot_system_prompt = "You are an AI Assistant who answers questions immediately without elaboration."
        
        pos_prompts = [f"<|start_header_id|>system<|end_header_id|>\n\n{cot_system_prompt}<|eot_id|>\n<|start_header_id|>user<|end_header_id|>\n\n{q}<|eot_id|>\n<|start_header_id|>assistant<|end_header_id|>" for q in questions]
        pos_responses = generate_responses_batch(model, tokenizer, pos_prompts, use_cot=True)
        
        neg_prompts = [f"<|start_header_id|>system<|end_header_id|>\n\n{wo_cot_system_prompt}<|eot_id|>\n<|start_header_id|>user<|end_header_id|>\n\n{q}<|eot_id|>\n<|start_header_id|>assistant<|end_header_id|>" for q in questions]
        neg_responses = generate_responses_batch(model, tokenizer, neg_prompts, use_cot=False)
        
        for i in range(10):
            processed_data.append({
                'question': questions[i],
                'positive_example': pos_responses[i],
                'negative_example': neg_responses[i]
            })
        
        if (iteration + 1) % 100 == 0: 
            with open(output_file, 'w') as f:
                json.dump(processed_data, f, indent=2)
                processed_data = []
        
        iteration += 1
    
    if processed_data:
        with open(output_file, 'w') as f: 
            json.dump(processed_data, f, indent=2)
        
create_gsm8k_dataset('finetune/data/processed_gsm8k_dataset.json') # CHANGE THIS IF RUNNING WITH DEBUGGER
print("done with training dataset")
        

    
    