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

def generate_response(model, tokenizer, prompt, use_cot, max_length = 512):
    inputs = tokenizer(prompt, return_tensors="pt", max_length = max_length, truncation=True) 
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=200,
            num_return_sequences=1,
            do_sample=True,
            temperature=0.7 if use_cot else 0.3 # higher to enable ease of CoT and lower for more immediate answering
        )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def create_gsm8k_dataset(output_file, model_name="meta-llama/Meta-Llama-3-8B-Instruct"):
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model = model.to(device)  # Move model to GPU
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    config = LlamaConfig.from_pretrained(model_name)
    config.use_cache = False
    
    processed_data = []
    for item in tqdm(training_data, desc="Processing GSM8k: "):
        cot_system_prompt = "You are a helpful AI Assistant who answers questions step by step."
        wo_cot_system_prompt = "You are an AI Assistant who answers questions immediately without elaboration."
        question = item['question']
        
        pos_prompt = f"<|start_header_id|>system<|end_header_id|>\n\n{cot_system_prompt}<|eot_id|>\n<|start_header_id|>user<|end_header_id|>\n\n{question}<|eot_id|>\n<|start_header_id|>assistant<|end_header_id|>"
        pos_response = generate_response(model, tokenizer, pos_prompt, use_cot=True)
        
        neg_prompt = f"<|start_header_id|>system<|end_header_id|>\n\n{wo_cot_system_prompt}<|eot_id|>\n<|start_header_id|>user<|end_header_id|>\n\n{question}<|eot_id|>\n<|start_header_id|>assistant<|end_header_id|>"
        neg_response = generate_response(model, tokenizer, neg_prompt, use_cot=False)
        
        processed_data.append({
            'question': question,
            'positive_example': pos_response,
            'negative_example': neg_response
        })
        
        with open(output_file, 'w') as f:
            json.dump(processed_data, f, indent=2)
        
create_gsm8k_dataset('finetune/data/processed_gsm8k_dataset.json') # CHANGE THIS IF RUNNING WITH DEBUGGER
print("done with training dataset")
        

    
    