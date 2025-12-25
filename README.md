<div align="center">

# FGPT: Conversational LLM on FineWeb-Edu and single GPU

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.9%2B-orange)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Experimental-yellow)

<br>
</div>

**FGPT** is a 712M parameter Language Model trained from scratch on the [FineWeb-Edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu) dataset. This repository provides code for training, finetuning and inference. The purpose of the repo is not to create a SOTA model but rather to experiment and learn. 

### Key Technical Implementations
* **Architecture:** GPT-2 Large equivalent (712M Params, 32 layers, 24 heads) with Phi-3 style prompt tokens with GPT2 Tokenizer.
* **Single GPU:** Trained on a single GPU to save money and make it reproducible for enthusiasts.
* **Stochastic Sampling:** Random batch sampling during training (vs. sequential) to mitigate domain drift caused by long documents, resulting in a significantly lower validation loss.
* **Muon Optimizer:** Fasters loss convergence to the use of Muon Optimizer (as used in [nano-gpt speedrun](https://x.com/kellerjordan0/status/1842300916864844014))
* **Instruction Tuning:** Fine-tuned on a composite dataset (Raschka + Alpaca-Cleaned) to enable 1-turn conversational capabilities.
* **Evals:** BaseModel eval on HellaSwag.



## Results

### Base Model

For the base model I achieve ~2.65 cross entropy nats on the validation set, which is a good result and about what we can expect without many advanced tweaks or a lot more compute time. As we are only training on english educational content, our dataset is fairly homogeneous compared to multilanguage datasets. If we were to train on something like FineWeb-Edu2 (the multilingual version) or OpenWebText, we would expect a higher loss.

![Base training overview](/report/images/base-train-overview.png)

The initial run was run with about 22B tokens, however after around 20B I observed a loss plateau. The loss plateau was acommpanied with strongly rising norms. To break through this plateau, the training continued with another 8B tokens, while learning rate was phased out to 0 (using cosine annealing schedule) and clipping the norms at 0.5 rather than 1.0. A correcting weight decay or even just a weight decaying schedule could improve this issue in future runs, as described by [Defazio 2025](https://arxiv.org/abs/2506.02285v2).

Every 10k steps I also evaluate the HellaSwag accuracy of the base model, which takes the logits of all responses (given the input) and evaluates which one is the most likely. The model scores ~44% which is significantly better than random guessing (=25%).

![HellaSwag Base Model](/report/images/hellaswag-base.png)

#### Sample Outputs

The table below shows how the sample outputs evolved with the steps of training. You can see that the model learns rough grammar and does not mix up tokens that don't go together (for the most part). It also starts to stick to semantic topics better. The input text is "Once upon a time" after which the model generates the rest.

| Step | Output |
|------|--------|
| 0    | _Once upon a time_ toolbar utterlyatti picked picked appropriations utterlyTex kickedatti addressingGR conflicting point conflictingumph distributingidential picked Berkeley inequalityspective identificationNation |
| 100'000 | _Once upon a time_-dependent process in which the user moves to a position in a file, an array containing the object is called a buffer. The buffer is the part of the file that contains the file. |
| 200'000 | _Once upon a time_, before anyone knew any better than the man to which he has contributed, he had the audacity of saying that he was in charge of the whole universe and that he did not want to leave the universe. |
| 350'000  | _Once upon a time_, there was a man named Peter. He had a very big house with a garden and a big farm. He grew vegetables all year long and kept some cows as well. Peter was a very rich man. He had lots of money to pay for his ... |


### Instruction Finetuned Model

Pretraining on the smol-smoltalk, alpaca-cleaned, and rashke datasets converged quickly. A total of YYY prompt - response pairs were used, where I only kept pairs where both prompt and response were shorter than 300 tokens max.

The loss convergence of the instruction finetuning is, unsurprisingly, much faster and the final loss much lower.

<!-- Pre finetuning the model has no understanding of question-answering or assistant style conversation. The autocomplete outputs look like this:

| Prompt | Output |
|--------|--------|
| What is the capital of France? |  - it is the largest French Banknote yet notes are mint in - The IMF estimates France’s ... |
| What is 2 + 2 | 2-2 (two-think french question book) 2-2 is pronounced as two two or wal-m ... | -->

The example outputs show that the model is able to create proper responses and seems to understand what is expected from it.

| Prompt | Output |
|--------|--------|
| What is the capital of France? | The capital of France is Paris |
| What is 2 + 2? | 2 + 2 = 4. |
| Define 'cat'. | A cat is a large mammal in the family Felidae, a family of cats. |
| Say hello. | Hello World! |
| Where is the White House located? | The White House is located at the intersection of Downtown New York and Alfred Street, which runs along the antero-parter of Main Street, which forms a southbound route from New York Harbor |
| Who wrote 'Romeo and Juliet'? | The book 'Romeo and Juliet' by William Shakespeare was published in 1616. |
| Summarize the plot of '1984' by George Orwell. | The plot of 1984 is a dystopian novel written by George Orwell. The story is set in the early 1980s and the protagonist, the protagonist, is a young, young person who is caught up in a dark, dreary world. As a result, there are many themes prevalent among the characters, which include betrayal, greed, and a need for power that is out of reach. The events and characters take shape, and the book is set in the future where the protagonist is confronted with a ... |

As (for now) the instruction finetuning data is 1-turn only, the model will generally only be able to awnser one question at a time (somewhat) reliably.

# Implementation Details

FGPT follows the standard GPT-2 Large architecture (decoder-only transformer). It has a 32-layer depth with 24 attention heads and an embedding dimension of 1248. The vocabulary size is padded to 50,304 (the nearest multiple of 64 from the GPT2 tokenizer) and the context length is 1024 (dense attention).

## Pretraining

The first step is training the model on a general text corpus of clean data. Just like Karpathy, I use Huggingface's [FineWeb-Edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu) ataset which is an english-only high-quality text dataset. The dataset is fairly homogeneous, which makes it easier to train on than other commonly used datasets like OpenWebText.

I make some changes to the dataloader functions in comparison to Karpathy's implementation. First, I split the FineWeb-Edu dataset at the document level, ensuring no document appears in both training and validation sets. This prevents data leakage and ensures a clean evaluation. Secondly, I also provide random batches of training data during training, and use a fixed validation batch throughout the training. The randomness of training data helps to avoid domain drift, as the documents of FineWeb-Edu can be very long and therefore the model just starts to memorize certain domains. This random approach was a game changer when trying to get the validation loss under 3.8 nats.

## Supervised Finetuning (Instruction Finetuning)

After the base model is trained, it is fine-tuned on a small instruction dataset so it behaves a bit more like a conversational assistant. The goal here isn’t to reach top performance but to help the model understand short question-answer patterns and simple user instructions.

```
<|user|>What is the capital of France?<|assistant|>The capital of France is Paris<|endoftext|>
```

After finetuning, the model gives short and (sometimes) relevant answers and can handle simple 1-turn conversations.

The data used for finetuning is a mix out of 2 datasets found online:

- [Sebastian Rashkes Instruction Following Data](https://github.com/rasbt/LLMs-from-scratch/blob/main/ch07/02_dataset-utilities/instruction-examples.json): This helps as it is an extremely simple dataset containing very short examples. As our model is not very good with context longer than a couple of sentences, this helps the model stick to short and concise answer.
- [yahma/alpaca-cleaned](https://huggingface.co/datasets/yahma/alpaca-cleaned): A cleaned instruction-answer type dataset that I limit to a maximum of 1000 characters for both prompt and answer 



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
- Deployment: HF Spaces demo
- Alignment: Implement DPO or other RLHF-like adjustment
