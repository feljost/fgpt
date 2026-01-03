"""WIP: The instruct tuned model does not achieve very good results on hellaswag yet."""

import torch
from datasets import load_dataset
from tqdm import tqdm
import re

from fgpt.inference import model_inference
from fgpt.inference import load_model, model_inference




def parse_answer(response: str) -> int | None:
    """Extract the answer choice (A/B/C/D or 0/1/2/3) from model response."""
    response = response.strip().upper()
    
    letter_map = {"A": 0, "B": 1, "C": 2, "D": 3}
    for letter, idx in letter_map.items():
        if response.startswith(letter):
            return idx
    
    match = re.search(r"[0-3]", response)
    if match:
        return int(match.group())
    
    return None


def evaluate_hellaswag(model, num_examples: int | None = None, debug: bool = False):
    """Evaluate model on HellaSwag using conversational prompting."""
    dataset = load_dataset("hellaswag", split="validation", trust_remote_code=True)
    
    if num_examples:
        dataset = dataset.select(range(num_examples))
    
    correct = 0
    total = 0
    failed_parses = 0
    
    for example in tqdm(dataset, desc="Evaluating HellaSwag"):
        context = example["ctx"]
        endings = example["endings"]
        label = int(example["label"])
        
        prompt = (
            "Complete the following sentence:\n\n"
            f"{context}\n\n"
            f"A) {endings[0]}\n"
            f"B) {endings[1]}\n"
            f"C) {endings[2]}\n"
            f"D) {endings[3]}"
        )
        
        _, response = model_inference(
            model,
            prompt=prompt,
            max_tokens=5,
            top_k=1,  # greedy decoding (top_k=1 effectively picks the most likely token)
            temperature=1.0,
            generation_type="conversational",
        )
        
        predicted = parse_answer(response)
        
        if debug:
            print(f"\n{'='*50}")
            print(f"Context: {context[:100]}...")
            print(f"Response: '{response}'")
            print(f"Parsed: {predicted}, Label: {label}")
        
        if predicted is None:
            failed_parses += 1
        elif predicted == label:
            correct += 1
        
        total += 1
    
    accuracy = correct / total
    return {
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
        "failed_parses": failed_parses,
    }


if __name__ == "__main__":
    model_weights_path = "/home/ubuntu/fgpt/model_weights_instruct.pth"
    model = load_model(model_weights_path=model_weights_path, device="cuda")
    model.eval()
    
    # Debug mode: see what's happening
    print("Testing on 5 examples with debug output...")
    results = evaluate_hellaswag(model, num_examples=1000, debug=True)
    print(f"\nAccuracy: {results['accuracy']:.2%}")
    print(f"Failed parses: {results['failed_parses']}/{results['total']}")
    
    # Once it looks good, run on more examples
    # print("\nRunning on 100 examples...")
    # results = evaluate_hellaswag(model, num_examples=100)
    # print(f"Accuracy: {results['accuracy']:.2%}")
    # print(f"Failed parses: {results['failed_parses']}/{results['total']}")