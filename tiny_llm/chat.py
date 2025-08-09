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
        full_prompt = f"Instruction: {prompt}\nResponse:"
        # No manual seeding — rely on model trained with exact prompt format
        full_prompt = f"Instruction: {prompt}\nResponse:"

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
                # Boost "I" token probability for first generated token to counter pretrained bias
                if generated.size(1) == inputs["input_ids"].size(1):
                    i_token_id = tokenizer.encode("I")[0]
                    next_token_logits[:, i_token_id] += 5.0

                # Apply stronger repetition penalty and break on repeating bigram
                repetition_penalty = 3.0
                gen_list = generated[0].tolist()
                for token_id in set(gen_list):
                    next_token_logits[:, token_id] /= repetition_penalty
                if len(gen_list) >= 2 and gen_list[-1] == gen_list[-2]:
                    # Force EOS to avoid infinite repetition of same token
                    next_token_id = torch.tensor([[tokenizer.eos_token_id]], device=device)
                    generated = torch.cat([generated, next_token_id], dim=-1)
                    break

                # Use top-p sampling to reduce repetition
                top_p = 0.9
                sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                cumulative_probs = torch.softmax(sorted_logits, dim=-1).cumsum(dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[:, 0] = False
                for batch_idx in range(next_token_logits.size(0)):
                    next_token_logits[batch_idx, sorted_indices[batch_idx, sorted_indices_to_remove[batch_idx]]] = -float("inf")

                probs = torch.softmax(next_token_logits, dim=-1)
                next_token_id = torch.multinomial(probs, num_samples=1)
                generated = torch.cat([generated, next_token_id], dim=-1)
                if next_token_id.item() == tokenizer.eos_token_id:
                    break

        resp_tokens = generated[0][inputs["input_ids"].size(1):]
        print(f"[DEBUG] Generated token IDs: {resp_tokens.tolist()}")
        response = tokenizer.decode(resp_tokens, skip_special_tokens=False).strip()
        print(f"TinyLLM: {response}")