import os
import math
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer
from model import TinyConfig, TinyTransformer

class TextDataset(Dataset):
    def __init__(self, texts, tokenizer, seq_len):
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.examples = []
        for txt in texts:
            token_ids = tokenizer.encode(txt, truncation=True, max_length=seq_len)
            self.examples.append(token_ids)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ids = self.examples[idx]
        x = torch.tensor(ids[:-1], dtype=torch.long)
        y = torch.tensor(ids[1:], dtype=torch.long)
        return x, y

def train(model, data_loader, optimizer, device):
    model.train()
    total_loss = 0
    for x, y in data_loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits, loss = model(x, targets=y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(data_loader)

if __name__ == "__main__":
    # Load dataset from generated file
    import json
    dataset_path = "pretrain_texts.json"
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"{dataset_path} not found. Run tiny_llm/sample_datasets.py to create it.")
    with open(dataset_path, "r", encoding="utf-8") as f:
        texts = json.load(f)
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.model_max_length = 512

    # Append EOS to texts for clearer sequence boundaries
    texts = [t + tokenizer.eos_token for t in texts]

    # Match fine-tune sequence length
    dataset = TextDataset(texts, tokenizer, seq_len=64)

    def pad_collate(batch):
        xs, ys = zip(*batch)
        max_len = max(x.size(0) for x in xs)
        pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
        pad_x = [torch.cat([x, torch.full((max_len - x.size(0),), pad_id, dtype=torch.long)]) for x in xs]
        pad_y = [torch.cat([y, torch.full((max_len - y.size(0),), -1, dtype=torch.long)]) for y in ys]
        return torch.stack(pad_x), torch.stack(pad_y)

    data_loader = DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=pad_collate)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = TinyConfig(vocab_size=tokenizer.vocab_size)
    model = TinyTransformer(config).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

    for epoch in range(10):
        loss = train(model, data_loader, optimizer, device)
        print(f"Epoch {epoch+1}: Loss {loss:.4f}")

    torch.save(model.state_dict(), "tiny_llm_pretrained.pt")