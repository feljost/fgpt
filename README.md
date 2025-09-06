## Overview

This repository provides scripts and notebooks for working with the FineWeb-Edu dataset and evaluating language model training. The main workflow involves using the FineWeb-Edu dataset to train a GPT-2-like large language model (LLM). Model performance is primarily assessed by monitoring training loss, with additional evaluation using the HellaSwag benchmark.

## Usage

### FineWeb-Edu Data Preparation

Run `fgpt/fineweb.py` to download and tokenize the FineWeb-Edu dataset. Sharded data will be saved to the `edu_fineweb100B` directory for use in LLM training.

```sh
python fgpt/fineweb.py
```

### Training Loop

The training file does all the heavy lifting for you. 
- Loads and optionally resumes a GPT model.
- Runs training with AdamW optimizer and cosine LR scheduler.
- Logs metrics (loss, gradient norm, tokens/sec, LR).
- Periodically evaluates on validation and HellaSwag.
- Saves checkpoints every 5000 steps.

```sh
python fgpt/train.py
```

### Visualization

Open `fgpt/visualize.ipynb` in Jupyter or VS Code to plot training metrics, including loss and HellaSwag accuracy.

## Requirements

- Python 3.10+
- `numpy`
- `tiktoken`
- `datasets`
- `tqdm`
- `torch`
- `matplotlib`
- `pandas`

Install dependencies with:

```sh
pip install numpy tiktoken datasets tqdm torch matplotlib pandas
```

