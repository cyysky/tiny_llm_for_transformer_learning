import os
import json

def create_pretrain_dataset(path="pretrain_texts.json"):
    """
    Creates a very small pretraining dataset.
    """
    texts = [
        "The sky is blue.",
        "Apples are red.",
        "Bananas are yellow.",
        "Cats like to sleep.",
        "Dogs like to play fetch.",
        "The sun rises in the East and sets in the West.",
        "Water freezes at zero degrees Celsius.",
        "The Earth orbits the Sun.",
        "Fish can swim in water.",
        "Birds can fly high in the sky.",
        "Chocolate is sweet and delicious.",
        "Rain falls from the clouds.",
        "Books are a source of knowledge.",
        "Fire is hot and can burn.",
        "Computers can process information quickly."
    ]
    with open(path, "w", encoding="utf-8") as f:
        json.dump(texts, f, ensure_ascii=False, indent=2)
    print(f"Pretrain dataset saved to {path}")


def create_instruction_dataset(path="instruction_pairs.json"):
    """
    Creates a small instruction-response dataset for fine-tuning.
    """
    instructions = [
        ("Say 'Hello World'", "Hello World"),
        ("Respond with 'I love AI'", "I love AI"),
        ("What is your name?", "I am TinyLLM"),
        ("Say 'Goodbye'", "Goodbye"),
        ("Repeat after me: Transformers are great", "Transformers are great"),
        ("State the color of the sky", "The sky is blue"),
        ("What do cats like to do?", "Cats like to sleep"),
        ("Tell me where the sun rises", "The sun rises in the East"),
        ("What is chocolate like?", "Chocolate is sweet and delicious"),
        ("Repeat after me: AI will change the world", "AI will change the world"),
        ("Finish this sentence: Fire is...", "Fire is hot and can burn"),
        ("Complete this: Birds can...", "Birds can fly high in the sky"),
        ("Repeat after me: Water freezes at zero degrees Celsius", "Water freezes at zero degrees Celsius"),
        ("Tell me the source of knowledge", "Books are a source of knowledge"),
        ("Complete this: The Earth...", "The Earth orbits the Sun")
    ]
    with open(path, "w", encoding="utf-8") as f:
        json.dump(instructions, f, ensure_ascii=False, indent=2)
    print(f"Instruction dataset saved to {path}")

if __name__ == "__main__":
    create_pretrain_dataset()
    create_instruction_dataset()