from fgpt.tokenizer import tokenizer
from fgpt.tokenizer import special_tokens
from pathlib import Path
import pandas as pd
from datasets import load_dataset
from tqdm import tqdm

dataset_name = "HuggingFaceTB/smoltalk"
max_token_length = 300
project_root_dir = Path(__file__).resolve().parents[3] / "instruction_data"
filename = "smoltalk_instruction_response_pairs.json"
filepath = project_root_dir / filename

dataset = load_dataset(dataset_name, "all")

df = pd.DataFrame(dataset["train"])
df = df.sample(frac=0.33, random_state=42).reset_index(
    drop=True
)  # we wont be needing all data

two_turn_mask = df["messages"].apply(lambda x: len(x) == 2)
df = df[two_turn_mask].reset_index(drop=True)

# Extract instruction-response pairs

for i in tqdm(range(df.shape[0])):
    df.loc[i, "instruction"] = df.loc[i, "messages"][0]["content"]
    df.loc[i, "output"] = df.loc[i, "messages"][1]["content"]

pairs = df[["instruction", "output"]]


# Count tokens in instruction and output (we don't want long pairs)
def count_tokens(text):
    return len(tokenizer.encode(text, allowed_special="all"))


def total_tokens(row):
    instr_tokens = count_tokens(row["instruction"])
    resp_tokens = count_tokens(row["output"])
    return instr_tokens + resp_tokens


pairs["total_tokens"] = pairs.apply(total_tokens, axis=1)

# Filter by token length
filtered_pairs = pairs[pairs["total_tokens"] <= max_token_length].reset_index(drop=True)
print(
    f"Kept {len(filtered_pairs)} of {len(pairs)} pairs (<= {max_token_length} tokens)."
)


for i in range(filtered_pairs.shape[0]):
    instr = filtered_pairs.loc[i, "instruction"]
    resp = filtered_pairs.loc[i, "output"]
    for key in special_tokens.keys():
        if key in instr or key in resp:
            print(f"Found special {key} token in row {i}; dropping it.")
            filtered_pairs.drop(i, inplace=True)


filtered_pairs.to_json(filepath, orient="records", lines=False)
