import torch
from transformers import AutoTokenizer
from peft import PeftModel
from model import TinyConfig, TinyTransformer

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    # Load model the same way as in fine-tuning (without LoRA since it was commented out)
    config = TinyConfig(vocab_size=tokenizer.vocab_size)
    model = TinyTransformer(config).to(device)
    model.load_state_dict(torch.load("../tiny_llm_lora_finetuned.pt", map_location=device), strict=False)
    model.eval()

    print("Chat with TinyLLM — type 'exit' to quit")
    while True:
        prompt = input("\nYou: ")
        if prompt.lower() == "exit":
            break

        # Match training data input format
        full_prompt = f"Instruction: {prompt}\nResponse: "
        inputs = tokenizer(full_prompt, return_tensors="pt").to(device)

        # Otherwise, use model generation
        max_new_tokens = 32
        generated = inputs["input_ids"]

        model.eval()
        with torch.no_grad():
            for _ in range(max_new_tokens):
                outputs = model(input_ids=generated)
                logits = outputs[0] if isinstance(outputs, tuple) else outputs
                next_token_logits = logits[:, -1, :]

                if generated.size(1) == inputs["input_ids"].size(1):
                    for bad_token in [tokenizer.encode(t)[0] for t in [".", ",", "!", "?"]]:
                        next_token_logits[:, bad_token] = -float("inf")
                    next_token_logits[:, tokenizer.eos_token_id] -= 5.0

                probs = torch.softmax(next_token_logits / 0.7, dim=-1)
                next_token_id = torch.multinomial(probs, num_samples=1)
                generated = torch.cat([generated, next_token_id], dim=-1)
                if next_token_id.item() == tokenizer.eos_token_id:
                    break

        resp_tokens = generated[0][inputs["input_ids"].size(1):]
        print(f"[DEBUG] Generated token IDs: {resp_tokens.tolist()}")
        response = tokenizer.decode(resp_tokens, skip_special_tokens=False).strip()
        print(f"TinyLLM: {response}")