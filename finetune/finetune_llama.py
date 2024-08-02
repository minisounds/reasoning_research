import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    LlamaConfig,
    AdamW,
    get_linear_schedule_with_warmup
)
from torch.cuda.amp import GradScaler, autocast
from bitsandbytes import optim as bit_optim
from tqdm import tqdm
import os
import json
import argparse
from sklearn.model_selection import train_test_split

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:256'

def print_memory_summary(): 
    num_devices = torch.cuda.device_count()
    for i in range(num_devices): 
        print(torch.cuda.memory_summary(device=torch.device(f"cuda:{i}"), abbreviated=False))
    

class GSM8kDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = data

    def __len__(self):
        return len(self.data) * 2  # Each item has a positive and negative example

    def __getitem__(self, idx):
        item = self.data[idx // 2]
        is_positive = idx % 2 == 0
        
        if is_positive:
            input_text = f"Question: {item['question']}\nReasoning: {item['positive_example']}"
        else:
            input_text = f"Question: {item['question']}\nReasoning: {item['negative_example']}"
        
        encoding = self.tokenizer(input_text, max_length=self.max_length, padding='max_length', truncation=True, return_tensors='pt')
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': encoding['input_ids'].squeeze(),
            'is_positive': torch.tensor(is_positive, dtype=torch.bool)
        }

def compute_loss(outputs, labels, is_positive):
    loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
    shift_logits = outputs.logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
    loss = loss.view(labels.size(0), -1).mean(dim=1)
    
    # Adjust loss based on positive/negative examples
    adjusted_loss = torch.where(is_positive, loss, -loss)
    return adjusted_loss.mean()

def train(model, train_loader, val_loader, optimizer, scheduler, device, num_epochs):
    best_val_loss = float('inf')
    scaler = GradScaler() # for mixed precision training
    accumulation_steps=4 # for gradient accumulation
    
    for epoch in range(num_epochs):
        model.train()
        total_train_loss = 0
        
        for i, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Training")):
            # Ensure all data in the batch is moved to the default CUDA device
            batch = {k: v.to(device) for k, v in batch.items()}

            optimizer.zero_grad()
            
            with autocast(): 
                outputs = model(input_ids = batch['input_ids'], attention_mask=batch['attention_mask'], labels=batch['labels'])
                loss = compute_loss(outputs, batch['labels'], batch['is_positive'])
                loss = loss / accumulation_steps
            
            scaler.scale(loss).backward()
            if (i + 1) % accumulation_steps == 0: # accumulation steps 
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
            
            # scaler.step(optimizer)
            # scaler.update()
            
            total_train_loss += loss.item() * accumulation_steps
        
        avg_train_loss = total_train_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{num_epochs}, Training Loss: {avg_train_loss:.4f}")
        
        # clear cache between training & validation: 
        torch.cuda.empty_cache()
        
        # Validation
        model.eval()
        total_val_loss = 0
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Validation"):
                # Ensure all data in the batch is moved to the default CUDA device
                batch = {k: v.to(device) for k, v in batch.items()}
                with autocast():
                    outputs = model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'], labels=batch['labels'], use_cache=False)
                    loss = compute_loss(outputs, batch['labels'], batch['is_positive'])
                
                total_val_loss += loss.item()
        
        avg_val_loss = total_val_loss / len(val_loader)
        print(f"Epoch {epoch+1}/{num_epochs}, Validation Loss: {avg_val_loss:.4f}")
        
        # clear cache between training & validation: 
        torch.cuda.empty_cache()
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            print(f"New best validation loss: {best_val_loss:.4f}. Saving model...")
            model.save_pretrained('best_model')
        
        # clear cache between training & validation: 
        torch.cuda.empty_cache()
        

def main(args):
    # Train the model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load the dataset
    with open(args.data_path, 'r') as f:
        data = json.load(f)
    
    # Split the dataset
    train_data, val_data = train_test_split(data, test_size=0.1, random_state=42)
    
    # Initialize tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(args.model_name)
    model.to(device)
    # model = torch.nn.DataParallel(model) # TODO: CHANGE THIS IF USING ONLY 1 GPU
    torch.cuda.empty_cache()
    config = LlamaConfig.from_pretrained(args.model_name)
    config.use_cache = False
    # model.config.use_cache = False
    
    
    # Create datasets and dataloaders
    train_dataset = GSM8kDataset(train_data, tokenizer)
    val_dataset = GSM8kDataset(val_data, tokenizer)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)
    
    # Initialize optimizer and scheduler
    optimizer = bit_optim.Adam8bit(model.parameters(), lr=args.learning_rate)
    # optimizer = AdamW(model.parameters(), lr=args.learning_rate)
    
    scheduler = get_linear_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=args.warmup_steps, 
        num_training_steps=len(train_loader) * args.num_epochs
    )
    
    
    train(model, train_loader, val_loader, optimizer, scheduler, device, args.num_epochs)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune LLaMA3 8B Instruct on GSM8k dataset")
    parser.add_argument("--data_path", type=str, required=True, help="Path to the processed GSM8k dataset")
    parser.add_argument("--model_name", type=str, default="meta-llama/Meta-Llama-3-8B-Instruct", help="Name or path of the pre-trained model")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for training")
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--num_epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--warmup_steps", type=int, default=100, help="Number of warmup steps for the scheduler")
    
    args = parser.parse_args()
    main(args)