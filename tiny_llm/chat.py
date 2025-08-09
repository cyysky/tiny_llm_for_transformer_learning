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
    model.load_state_dict(torch.load("tiny_llm_lora_finetuned.pt", map_location=device), strict=False)
    model.eval()

    print("Chat with TinyLLM — type 'exit' to quit")
    while True:
        prompt = input("\nYou: ")
        if prompt.lower() == "exit":
            break

        # Match training data input format (now aligned with fine-tuning)
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

                # Greedy decoding with EOS stop
                # No manual token biasing — rely purely on trained weights

                # Apply light repetition penalty
                repetition_penalty = 1.5
                gen_list = generated[0].tolist()
                for token_id in set(gen_list):
                    if token_id != tokenizer.eos_token_id:
                        next_token_logits[:, token_id] /= repetition_penalty

                # Force-learned sequence biasing for exact phrase recall
                # Expected token sequence from training (" I am Tiny LLM.")
                #target_text = " I am Tiny LLM."
                #seq_tokens = tokenizer.encode(target_text)
                #step_idx = generated.size(1) - inputs["input_ids"].size(1)
                #if 0 <= step_idx < len(seq_tokens):
                #    next_token_logits[:, seq_tokens[step_idx]] += 5.0
                # Removed forced sequence biasing to test actual model recall

                # Stop if the last two tokens are identical to avoid infinite loops
                if len(gen_list) >= 2 and gen_list[-1] == gen_list[-2]:
                    next_token_id = torch.tensor([[tokenizer.eos_token_id]], device=device)
                    generated = torch.cat([generated, next_token_id], dim=-1)
                    break

                # Use pure greedy decoding for deterministic output
                next_token_id = torch.argmax(next_token_logits, dim=-1, keepdim=True)
                generated = torch.cat([generated, next_token_id], dim=-1)
                if next_token_id.item() == tokenizer.eos_token_id:
                    break

        resp_tokens = generated[0][inputs["input_ids"].size(1):]
        print(f"[DEBUG] Generated token IDs: {resp_tokens.tolist()}")
        response = tokenizer.decode(resp_tokens, skip_special_tokens=False).strip()
        print(f"TinyLLM: {response}")