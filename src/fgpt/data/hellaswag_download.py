
import os
import sys
import json
import requests
import tiktoken
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.nn import functional as F

DATA_CACHE_DIR = os.path.join(os.path.dirname(__file__), "hellaswag")

def download_file(url: str, fname: str, chunk_size=1024):
    """Helper function to download a file from a given url"""
    resp = requests.get(url, stream=True)
    total = int(resp.headers.get("content-length", 0))
    with (
        open(fname, "wb") as file,
        tqdm(
            desc=fname,
            total=total,
            unit="iB",
            unit_scale=True,
            unit_divisor=1024,
        ) as bar,
    ):
        for data in resp.iter_content(chunk_size=chunk_size):
            size = file.write(data)
            bar.update(size)


hellaswags = {
    "train": "https://raw.githubusercontent.com/rowanz/hellaswag/master/data/hellaswag_train.jsonl",
    "val": "https://raw.githubusercontent.com/rowanz/hellaswag/master/data/hellaswag_val.jsonl",
    "test": "https://raw.githubusercontent.com/rowanz/hellaswag/master/data/hellaswag_test.jsonl",
}

enc = tiktoken.get_encoding("gpt2")


def download(split):
    """Downloads HellaSwag DATA_CACHE_DIR"""
    os.makedirs(DATA_CACHE_DIR, exist_ok=True)
    data_url = hellaswags[split]
    data_filename = os.path.join(DATA_CACHE_DIR, f"hellaswag_{split}.jsonl")
    if not os.path.exists(data_filename):
        print(f"Downloading {data_url} to {data_filename}...")
        download_file(data_url, data_filename)



def convert_hellaswag_to_sft(input_path, output_path):
    """Convert HellaSwag JSONL to supervised fine-tuning format as JSON array."""
    
    label_map = {0: "A", 1: "B", 2: "C", 3: "D"}
    sft_data = []
    
    with open(input_path, 'r', encoding='utf-8') as infile:
        for line in infile:
            if not line.strip():
                continue
                
            data = json.loads(line.strip())
            
            context = data['ctx']
            endings = data['endings']
            correct_label = data['label']
            
            instruction = (
                f"Complete the following sentence:\n\n"
                f"{context}\n\n"
                f"A) {endings[0]}\n"
                f"B) {endings[1]}\n"
                f"C) {endings[2]}\n"
                f"D) {endings[3]}"
            )
            
            output = label_map[correct_label]
            
            sft_data.append({
                "instruction": instruction,
                "output": output
            })
    
    with open(output_path, 'w', encoding='utf-8') as outfile:
        json.dump(sft_data, outfile, indent=2, ensure_ascii=False)