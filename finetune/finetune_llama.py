import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    LlamaConfig,
    AdamW,
    get_linear_schedule_with_warmup
)
from tqdm import tqdm
import json
import argparse
from sklearn.model_selection import train_test_split

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
    model.to(device)
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        model.train()
        total_train_loss = 0
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Training"):
            optimizer.zero_grad()
            
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            is_positive = batch['is_positive'].to(device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = compute_loss(outputs, labels, is_positive)
            
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            total_train_loss += loss.item()
        
        avg_train_loss = total_train_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{num_epochs}, Training Loss: {avg_train_loss:.4f}")
        
        # Validation
        model.eval()
        total_val_loss = 0
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Validation"):
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                is_positive = batch['is_positive'].to(device)
                
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = compute_loss(outputs, labels, is_positive)
                
                total_val_loss += loss.item()
        
        avg_val_loss = total_val_loss / len(val_loader)
        print(f"Epoch {epoch+1}/{num_epochs}, Validation Loss: {avg_val_loss:.4f}")
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            print(f"New best validation loss: {best_val_loss:.4f}. Saving model...")
            model.save_pretrained('best_model')

def main(args):
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
    config = LlamaConfig.from_pretrained(args.model_name)
    config.use_cache = False
    
    
    # Create datasets and dataloaders
    train_dataset = GSM8kDataset(train_data, tokenizer)
    val_dataset = GSM8kDataset(val_data, tokenizer)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)
    
    # Initialize optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=args.learning_rate)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=args.warmup_steps, 
        num_training_steps=len(train_loader) * args.num_epochs
    )
    
    # Train the model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train(model, train_loader, val_loader, optimizer, scheduler, device, args.num_epochs)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune LLaMA3 8B Instruct on GSM8k dataset")
    parser.add_argument("--data_path", type=str, required=True, help="Path to the processed GSM8k dataset")
    parser.add_argument("--model_name", type=str, default="meta-llama/Meta-Llama-3-8B-Instruct", help="Name or path of the pre-trained model")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for training")
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--num_epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--warmup_steps", type=int, default=100, help="Number of warmup steps for the scheduler")
    
    args = parser.parse_args()
    main(args)