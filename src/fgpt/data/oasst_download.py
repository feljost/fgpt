from fgpt.tokenizer import tokenizer
from fgpt.tokenizer import special_tokens
from pathlib import Path
import pandas as pd
from datasets import load_dataset
from tqdm import tqdm


dataset_name = "OpenAssistant/oasst1"

max_token_length = 900 # some buffer for our 1024 token context window
project_root_dir = Path(__file__).resolve().parents[3] / "instruction_data"
filename = "oasst1_en_instruction_response_pairs.json"
filepath = project_root_dir / filename

dataset = load_dataset("OpenAssistant/oasst1")

english_data = dataset['train'].filter(lambda x: x['lang'] == 'en')

df = pd.DataFrame(english_data)

# Only use the first user message and the first assistant reply
roots = df[(df['parent_id'].isnull()) & (df['role'] == 'prompter')]



# Join roots with their first assistant replies
merged = pd.merge(
    roots,
    df[df['role'] == 'assistant'],
    how='inner',
    left_on='message_id',
    right_on='parent_id',
    suffixes=('_user', '_assistant')
)

# Keep only the first reply for each conversation
merged = merged.sort_values('created_date_assistant').groupby('message_id_user').first().reset_index()

# Extract instruction-response pairs
pairs = merged[['text_user', 'text_assistant']].rename(
    columns={'text_user': 'instruction', 'text_assistant': 'output'}
)
pairs["input"] = "" # To ensure it has the same format as the rasbt dataset


# Count tokens in instruction and output (we don't want long pairs)
def count_tokens(text):
    return len(tokenizer.encode(text, allowed_special="all"))

def total_tokens(row):
    instr_tokens = count_tokens(row['instruction'])
    resp_tokens = count_tokens(row['output'])
    return instr_tokens + resp_tokens

pairs["total_tokens"] = pairs.apply(total_tokens, axis=1)

# Filter by token length
filtered_pairs = pairs[pairs["total_tokens"] <= max_token_length].reset_index(drop=True)
print(f"Kept {len(filtered_pairs)} of {len(pairs)} pairs (<= {max_token_length} tokens).")


for i in range(pairs.shape[0]):
    instr = pairs.loc[i, 'instruction']
    resp = pairs.loc[i, 'output']
    for key in special_tokens.keys():
        if key in instr or key in resp:
            print(f"Found special {key} token in row {i}; dropping it.")
            pairs.drop(i, inplace=True)


pairs.to_json(filepath, orient="records", lines=False)