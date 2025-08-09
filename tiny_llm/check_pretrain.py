import torch
from transformers import AutoTokenizer
from model import TinyConfig, TinyTransformer

if __name__ == "__main__":
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    # Create model config and model
    config = TinyConfig(vocab_size=tokenizer.vocab_size)
    model = TinyTransformer(config)

    # Load pretrained weights
    ckpt_path = "tiny_llm_pretrained.pt"
    if not torch.load(ckpt_path, map_location="cpu"):
        raise FileNotFoundError(f"{ckpt_path} not found. Run pretrain.py first to generate it.")
    model.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
    model.eval()

    # Prepare prompt
    prompt = "The sky is"
    input_ids = tokenizer.encode(prompt, return_tensors="pt")

    # Generate text manually (greedy decoding)
    max_new_tokens = 50
    for _ in range(max_new_tokens):
        logits = model(idx=input_ids)  # (batch, seq, vocab)
        next_token_logits = logits[:, -1, :]
        next_token_id = torch.argmax(next_token_logits, dim=-1).unsqueeze(0)
        input_ids = torch.cat([input_ids, next_token_id], dim=1)

    # Decode and print
    output_text = tokenizer.decode(input_ids[0], skip_special_tokens=True)
    print("Generated text:")
    print(output_text)