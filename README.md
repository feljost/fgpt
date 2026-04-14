<div align="center">

# FGPT: Conversational LLM on FineWeb-Edu and single GPU

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.9%2B-orange)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Experimental-yellow)

<br>
</div>

**FGPT** is a 600M parameter Language Model trained from scratch on the [FineWeb-Edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu) dataset. This repository provides code for training, finetuning and inference. The purpose of the repo is not to create a SOTA model but rather to experiment and learn. 

### Key Technical Implementations

* **Architecture:** Modern Decoder-only Transformer (Llama-style) using GPT-2 Large dimensions (600M Params, 32 layers, 24 heads). Features SwiGLU, RoPE, and RMSNorm. Uses Phi-3 style prompt templates and the GPT-2 Tokenizer.
* **Single GPU:** Trained on a single GPU to ensure reproducibility for enthusiasts with limited compute.
* **Stochastic Sampling:** Random batch sampling during training (vs. sequential) to mitigate domain drift caused by long documents.
* **Muon Optimizer:** Faster loss convergence due to the use of the Muon Optimizer (as used in [nano-gpt speedrun](...)).
* **Instruction Tuning:** Fine-tuned on a composite dataset (Smoltalk + Raschka + Alpaca-Cleaned) to enable 1-turn conversational capabilities.
* **Evals:** Base model evaluated on HellaSwag.

### Training Dynamics & Infrastructure

Training was conducted on a single NVIDIA GH200 GPU via Lambda Labs. Using this setup, I was able to use a large effective batch size of ~0.5M tokens (accumulated over 12 micro-steps) and sustain a training throughput of ~81,000 tokens/second.

* **Total Compute Time:** ~210 Hours (Wall clock)
* **Total Tokens Seen:** ~45 Billion
* **Cost Estimate:** ~$300

The 210-hour runtime reflects the experimental nature of the project, including a necessary restart to break a loss plateau (visible in the loss curve) and a slow initial decline for the first 1/3 of the run. With more accurate hyperparameters, a reproduction run would likely require significantly fewer GPU hours.

## Results

### Base Model

For the base model I achieve ~2.58 cross entropy nats on the validation set, which is a good result and about what we can expect without many more tweaks or a lot more compute time. As we are only training on English educational content, our dataset is fairly homogeneous compared to multilanguage datasets. If we were to train on something like FineWeb-Edu2 (the multilingual version) or OpenWebText, we would expect a higher loss.

![Base training overview](/report/images/base-train-overview.png)

The initial run was run with about 32B tokens, after which I observed loss plateau. The loss plateau was accompanied with strongly rising norms. To break through this plateau, the training continued with another 10B tokens, while learning rate was phased out to 0 (using cosine annealing schedule). To combat the rising Norms, the norm clipping was set to 0.5 already after 350k microsteps. A correcting weight decay or even just a weight decaying schedule could improve this issue in future runs, as described by [Defazio 2025](https://arxiv.org/abs/2506.02285v2).

#### Benchmarks

FGPT outperforms the architectural baseline (GPT-2 Large) on HellaSwag zero-shot evaluation, demonstrating the efficacy of modern architectural components (SwiGLU, RoPE) and the Muon optimizer.

| Model | Parameters | HellaSwag (0-shot) | Architecture |
| :--- | :--- | :--- | :--- |
| **FGPT (Ours)** | **600M** | **~49.0%** | **Llama-style (SwiGLU/RoPE)** |
| GPT-2 Large | 774M | ~45.0%* | Standard GPT-2 |
| GPT-2 XL | 1.5B | ~51.0% | Standard GPT-2 |

*> Baseline sourced from [llama.cpp discussions](https://github.com/ggml-org/llama.cpp/discussions/2321)*

The Hellaswag accuracy was evaluated every 25k micro-steps. The evaluation was simple: takethe logits of all responses (given the input) and picks the most likely response. The model scores ~49% which is significantly better than random guessing (=25%).

![HellaSwag Base Model](/report/images/hellaswag-base.png)

#### Sample Outputs (Base)

The table below shows how the sample outputs evolved with the steps of training. You can see that the model learns rough grammar and does not mix up tokens that don't go together (for the most part). It also starts to stick to semantic topics better. The input text is "Once upon a time" after which the model generates the rest.

| Step | Output |
|------|--------|
| 0    | _Once upon a time_ toolbar utterlyatti picked picked appropriations utterlyTex kickedatti addressingGR conflicting point conflictingumph distributingidential picked Berkeley inequalityspective identificationNation |
| 100'000 | _Once upon a time_ they were called back to life after the fact, and their role was so important that it was a matter of life or death that the first of the great great empires, which in the past had been founded by an absolute number of states and individuals |
| 500'000 | _Once upon a time_, a very important man died, and the man went by the surname of his grandfather. In that case the man became the father of his grandfather, the son of his grandfather, and the brother of his mother. If the man's father could not" |
| 1'000'000 | _Once upon a time_, we were all peasants, but we grew up and became teachers, lawyers, politicians, writers and scientists, we learned the arts, science, religion, history and even how to ride a horse.\nWe lived, we worked, we loved, |
| 1'350'000 | _Once upon a time_, the earth was a perfect paradise. The trees were growing as long as they had a right to do so. They had everything they needed to survive. Their only concern was whether they will have enough food to eat throughout the year, or how many |


### Instruction Finetuned Model

Finetuning on the around 200k prompt - response paired converged quickly. The training was only done on pairs where both prompt and response were shorter than 300 tokens max.

The loss convergence of the instruction finetuning is, unsurprisingly, much faster and the final loss much lower.

![SFT Loss Curves](/report/images/sft-loss.png)

#### Sample Outputs (Instruct)

The example output below shows that the model is able to create proper responses and seems to understand what is expected from it... atleast sometimes :).

| Prompt | Output |
|--------|--------|
| What is the capital of France? | Paris is the capital of France. |
| Say hello. | Hello, how are you? |
| Define 'cat'. | The word 'cat' is a noun that refers to an animal that belongs to the felidae family and is characterized by a pointed facial feature with a pointed tail and short body. It is also known as the domestic cat, or simply 'cat', although the former term is considered outdated and is considered incorrect. |
| Where is the White House located? | The White House is located in Washington, DC, America. |
| What is the boiling point of water? | The boiling point of water is about 100 degrees C (212 f). |
| Summarize the plot of '1984' by George Orwell. | '1984' is a fictional account of a dystopian novel where a new society arises in the United States. The plot centers around a dystopian nation whose citizens have been indoctrinated into obedience through the use of propaganda and surveillance. ...
| What is 2 + 2? | The answer is 4. |
| Tell me a fun fact about Italy | Italy is known for its rich history and heritage. Its culture is renowned for its ancient city-states, the Renaissance, and its unique gastronomy. The country is also known for its wine, olive oil, and other agricultural products. |
| What color is the sky? | The sky is blue.\nI don't think the sky is blue, I feel like the sky should be blue |

As (for now) the instruction finetuning data is 1-turn only, the model will generally only be able to answer one question at a time (somewhat) reliably.

# Implementation Details

FGPT follows the standard GPT-2 Large architecture (decoder-only transformer). It has a 32-layer depth with 24 attention heads and an embedding dimension of 1248. The vocabulary size is padded to 50,304 (the nearest multiple of 64 from the GPT2 tokenizer) and the context length is 1024 (dense attention).

## Pretraining

The first step is training the model on a general text corpus of clean data. Just like Karpathy, I use Huggingface's [FineWeb-Edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu) dataset which is an English-only high-quality text dataset. The dataset is fairly homogeneous, which makes it easier to train on than other commonly used datasets like OpenWebText.

I make some changes to the dataloader functions in comparison to Karpathy's implementation. First, I split the FineWeb-Edu dataset at the document level, ensuring no document appears in both training and validation sets. This prevents data leakage and ensures a clean evaluation. Secondly, I also provide random batches of training data during training, and use a fixed validation batch throughout the training. The randomness of training data helps to avoid domain drift, as the documents of FineWeb-Edu can be very long and therefore the model just starts to memorize certain domains. This random approach was a game changer when trying to get the validation loss under 3.8 nats.

## Supervised Finetuning (Instruction Finetuning)

After the base model is trained, it is fine-tuned on a small instruction dataset so it behaves a bit more like a conversational assistant. The goal here isn’t to reach top performance but to help the model understand short question-answer patterns and simple user instructions.

```
<|user|>What is the capital of France?<|assistant|>The capital of France is Paris<|endoftext|>
```

After finetuning, the model gives short and (sometimes) relevant answers and can handle simple 1-turn conversations.

The data used for finetuning is a mix out of 3 datasets found online:

- [HuggingFaceTB/smol-smoltalk](https://huggingface.co/datasets/HuggingFaceTB/smol-smoltalk): A version of smoltalk by HF that is specifically for smaller model (makes up the bulk of the SFT data)
- [Sebastian Rashkes Instruction Following Data](https://github.com/rasbt/LLMs-from-scratch/blob/main/ch07/02_dataset-utilities/instruction-examples.json): This helps as it is an extremely simple dataset containing very short examples. As our model is not very good with context longer than a couple of sentences, this helps the model stick to short and concise answer.
- [yahma/alpaca-cleaned](https://huggingface.co/datasets/yahma/alpaca-cleaned): A cleaned instruction-answer type dataset that I limit to a maximum of 1000 characters for both prompt and answer.


## Usage

You will need a strong GPU with cuda to run these scripts. If you don't have one locally, I suggest getting one in the cloud (I used lambda labs).

### Requirements

Install dependencies with uv easily:

```sh
uv pip install -e .
```

Depending on your version of CUDA, you might have to specify a specific torch version (see [pytorch.org](https://pytorch.org/)).

### Data Downloads

Run `fineweb_download.py` to download and tokenize the [FineWeb-Edu dataset](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu). Sharded data will be saved to the `edu_fineweb100B` directory for use in LLM training.

```sh
python src/fgpt/data/fineweb_download.py
```

### Training

The instruction finetuning datasets are already in this repo in `instruction/data` as json files. If you want to see the processing I did to them or download them yourself, check the scripts under `src/fgpt/data`.


```sh
python src/fgpt/base_train.py
```

Before starting the training, you may want to adjust some of the model and training parameters in 
`model.py` and `base_train.py`, depending on your available compute and time commitment.

Please note that the training will only happen one 1 GPU at a time. You will need to adjust training loop if you want to do multi-GPU training.


```sh
python src/fgpt/instruct_train.py
```

### Inference

You may create example inferences with the following command.

```sh
python src/fgpt/inference.py
```

# References

- [Andrej Karpathy: Let's reproduce GPT-2 (124M)](https://www.youtube.com/watch?v=l8pRSuU81PU)
- [Sebastian Rashke: Build an LLM from Scratch 7: Instruction Finetuning](https://www.youtube.com/watch?v=4yNswvhPWCQ)
- [HF-Dataset: FineWeb-Edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu)
- [HF-Dataset: yahma/alpaca-cleaned](https://huggingface.co/datasets/yahma/alpaca-cleaned)
- [Modded-NanoGPT: Nano GPT training speedruns](https://github.com/KellerJordan/modded-nanogpt)
- [Defazio 2025: Why Gradients Rapidly Increase Near the End of Training](https://arxiv.org/abs/2506.02285v2)
- [Liu et al 2025: Muon is Scalable for LLM Training](https://arxiv.org/abs/2502.16982)

# TO DO's
- ~Core Architecture: GPT-2 Large implementation~
- ~Data Pipeline: Custom dataloader with leakage prevention~
- ~Optimization: FlashAttention integration~
- ~Optimization: Switch to Muon Optimizer~
- ~Muon Optimizer: Improve training speed~
- ~Scaling: Train on >14B tokens (Chinchilla optimality)~
- ~Implement modern stack: RoPE, SwiGLU and RMSNorm~
- Implement findings from autoresearch branch
- SFT: Better selection of SFT datasets and instruction finetuning. (ongoing)
- Chat-based evaluation: add instruction data on multiple choice questions & check hellaswag after instruction finetuning.
- Deployment: HF Spaces demo
- Alignment: Implement DPO or other RLHF-like adjustment
