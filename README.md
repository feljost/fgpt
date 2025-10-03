## Overview

This repository provides scripts and notebooks for to train a GPT2-like model on the FineWeb-Edu Dataset. The purpose of the repo is not to create a SOTA model but rather to experiment and learn. The code is mainly based on Karphaty's youtube videos. 

The model is follows the GPT2 architecture and uses the same GPT2 tokenizer. Size wise, it is roughly between GPT2-Small and GPT2-Medium. It has 270M Parameters with 16 layers and 16 heads. The learning rate schedule and other hyperparameters are also different to allow for more fun when tinkering with the architecture. The training happened on a single NVIDIA GH200 480GB.

Model performance is primarily assessed by monitoring validation & training loss, with additional evaluation using the HellaSwag benchmark. Example outputs of the model are also checked over time.
Unlike Karpathy's original code, this repository splits the FineWeb-Edu dataset at the document level, ensuring no document appears in both training and validation sets. This prevents data leakage and ensures a clean evaluation.

## Usage

You will need a strong GPU with cuda to run these scripts. If you don't have one locally, I suggest getting one in the cloud (I used lambda labs).

### FineWeb-Edu Data Preparation

Run `fineweb.py` to download and tokenize the FineWeb-Edu dataset. Sharded data will be saved to the `edu_fineweb100B` directory for use in LLM training.

```sh
python fineweb.py
```

### Training Loop

The training file does all the heavy lifting for you. 
- Loads and optionally resumes a GPT model.
- Runs training with AdamW optimizer and cosine LR scheduler.
- Logs metrics (loss, gradient norm, tokens/sec, LR).
- Periodically evaluates on validation and HellaSwag.
- Saves checkpoints every 5000 steps.

```sh
python train.py
```

### Requirements

Install dependencies with:

```sh
pip install numpy tiktoken datasets tqdm torch matplotlib pandas
```

## Results



Visualizations created with `visualize.ipynb`.


## Future To Do's

- Instruction Finetuning
- Preference-optimizitaion (DPO or other easy to implement approach)
