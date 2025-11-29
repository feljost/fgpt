import json
from datasets import load_dataset
from tqdm import tqdm

# --- Configuration ---
DATASET_NAME = "yahma/alpaca-cleaned"
OUTPUT_FILE = "simple_qa_only.json"
MAX_LENGTH = 1_000
# ---------------------


def combine_and_filter_data(dataset):
    """
    Combines 'instruction' and 'input', and filters for short, simple examples.
    """
    print(f"Loaded dataset with {len(dataset)} total examples.")
    
    # 1. Combine Instruction and Input
    def format_example(example):
        instruction = example['instruction'].strip()
        input_data = example['input'].strip()

        # If an input exists, append it to the instruction with a clear separator
        if input_data:
            # This creates a combined prompt like:
            # "Reverse the phrase: Moon and stars"
            # where the original instruction might have been "Reverse the phrase" 
            # and the input was "Moon and stars"
            example['instruction'] = f"{instruction}\n\n{input_data}"
        else:
            example['instruction'] = instruction # Use original instruction if no input

        return example

    # Use .map for efficient processing
    combined_dataset = dataset.map(format_example)

    # 2. Filter for Short Length and Non-Empty Output
    def is_simple_and_short(example):
        # Must have a non-empty instruction and output
        if not example['instruction'] or not example['output']:
            return False

        # Must be shorter than the maximum length defined
        return (len(example['instruction']) < MAX_LENGTH and 
                len(example['output']) < MAX_LENGTH)

    # Apply the filter
    simple_qa_dataset = combined_dataset.filter(is_simple_and_short)

    return simple_qa_dataset


def save_dataset_to_json(dataset, filename):
    final_data = []
    
    print(f"\nProcessing {len(dataset)} filtered examples...")
    for item in tqdm(dataset):
        final_data.append({
            "instruction": item['instruction'],
            "output": item['output']
        })

    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(final_data, f, ensure_ascii=False, indent=2)

    print(json.dumps(final_data[0], indent=2))

if __name__ == "__main__":

    alpaca_dataset = load_dataset(DATASET_NAME, split='train')

    simple_data = combine_and_filter_data(alpaca_dataset)
    save_dataset_to_json(simple_data, OUTPUT_FILE)