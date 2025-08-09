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
        "Computers can process information quickly.",
        "Mount Everest is the tallest mountain in the world.",
        "The Pacific Ocean is the largest ocean on Earth.",
        "Shakespeare wrote Romeo and Juliet.",
        "Light travels faster than sound.",
        "The human heart pumps blood through the body.",
        "Paris is the capital city of France.",
        "An octopus has eight arms.",
        "The Great Wall of China is visible from space.",
        "The Sahara is the largest hot desert in the world.",
        "Penguins cannot fly but are excellent swimmers.",
        "The Amazon rainforest is the largest tropical rainforest.",
        "Electricity powers most modern devices.",
        "The Moon orbits the Earth.",
        "Leaves use sunlight to make food through photosynthesis."
    ]
    with open(path, "w", encoding="utf-8") as f:
        json.dump(texts, f, ensure_ascii=False, indent=2)
    print(f"Pretrain dataset saved to {path}")


def create_instruction_dataset(path="instruction_pairs.json"):
    """
    Creates a small instruction-response dataset for fine-tuning.
    """
    instructions = [
        # Heavy oversampling of name-related questions with exact answer "I am TinyLLM."
        ("What is your name?", "I am Tiny LLM.")
    ]
    with open(path, "w", encoding="utf-8") as f:
        json.dump(instructions, f, ensure_ascii=False, indent=2)
    print(f"Instruction dataset saved to {path}")

if __name__ == "__main__":
    create_pretrain_dataset()
    create_instruction_dataset()