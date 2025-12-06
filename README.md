<div align="center">

# fgpt: Conversational LLM on FineWeb-Edu and single GPU

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.9%2B-orange)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Experimental-yellow)

<br>

**fgpt** is a 712M parameter Language Model trained from scratch on the [FineWeb-Edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu) dataset. This repository provides code for training, finetuning and inference. The purpose of the repo is not to create a SOTA model but rather to experiment and learn. The code is loosely based on Karpathy's youtube videos and Sebastian Raschka's LLM from scratch code. 

### Key Technical Implementations
* **Architecture:** GPT-2 Large equivalent (712M Params, 32 layers, 24 heads) with Phi-3 style prompt tokens with GPT2 Tokenizer.
* **Single GPU:** Trained on a single GPU to save money and make it reproducible for enthusiasts.
* **Stochastic Sampling:** Random batch sampling during training (vs. sequential) to mitigate domain drift caused by long documents, resulting in a significantly lower validation loss.
* **Instruction Tuning:** Fine-tuned on a composite dataset (Raschka + Alpaca-Cleaned) to enable 1-turn conversational capabilities.
* **Evals:** Quantitative BaseModel eval on HellaSwag.

---

## Results

For the base model I achieve ~2.8 cross entropy nats on the validation set, which is a good result and about what we can expect without many advanced tweaks. As we are only training on english educational content, our dataset is fairly homogeneous compared to multilanguage datasets. If we were to train on something like FineWeb-Edu2 (the multilingual version) or OpenWebText, we would expect a higher loss.

![Loss Curves](/report/images/train-loss.png)

Every 10k steps I also evaluate the HellaSwag accuracy of the base model, which takes the logits of all responses (given the input) and evaluates which one is the most likely. The model scores ~35.8% which is significantly better than random guessing (=25%). Instruction finetuned version will follow.

![HellaSwag Base Model](/report/images/hellaswag-base.png)

### Sample Outputs

#### Base Model

The table below shows how the sample outputs evolved with the steps of training. You can see that the model learns rough grammar and does not mix up tokens that don't go together (for the most part). It also starts to stick to semantic topics better. The input text is "Once upon a time" after which the model generates the rest.

| Step | Output |
|------|--------|
| 0    | _Once upon a time_ Once upon a timeLittle generally Libre affection torment Saga sword tributeerredarez QBraz fmt repeatedREPcade Geralomical Fiat ACTION situ RVvant advisors escalation screenshot Lines Issa randomly Improveashi Strip[/pi thesis oppressionRussiact demographics1965 refuel degridor |
| 100'000 | _Once upon a time_ whereas Pluto is ramping up its backyard, it will be astonishing achievements, but the world is not yet at this point. To see them, check out the Moon's satellite images below: You can see moons in all four picture books. |
| 250'000 | _Once upon a time_, a wealthy man buried his sons and daughters in woodland, farming geometric patterns, thus creating the food pattern I am after today. More than that, I want you to know to sacrifice your health (I fire you), beauty (II fire you) |
| 400'000  | _Once upon a time_, there was one man lurking in a hedge. In those days, there was a very big storm. It was a really cold day. Even with the weather so dangerous, there was only a few people who could stay awake |
| 500'000  | _Once upon a time_, there were three. One was alive, but a describes how he lives or moves around. There in the Evergreen state there lived a man who lived out his days, coming to the promise of happiness. But he pondered on how he connected |

#### Instruction Finetuned Model

Pre finetuning the model has no understanding of question-answering or assistant style conversation. The autocomplete outputs look like this:

| Prompt | Output |
|--------|--------|
| What is the capital of France? |  - it is the largest French Banknote yet notes are mint in - The IMF estimates France’s ... |
| What is 2 + 2 | 2-2 (two-think french question book) 2-2 is pronounced as two two or wal-m ... |

After finetuning the outputs are actually not too bad:

| Prompt | Output |
|--------|--------|
| What is the capital of France? | The capital of France is Paris |
| What is 2 + 2? | 2 + 2 = 4. |
| Define 'cat'. | A cat is a large mammal in the family Felidae, a family of cats. |
| Say hello. | Hello World! |
| Where is the White House located? | The White House is located at the intersection of Downtown New York and Alfred Street, which runs along the antero-parter of Main Street, which forms a southbound route from New York Harbor |
| Who wrote 'Romeo and Juliet'? | The book 'Romeo and Juliet' by William Shakespeare was published in 1616. |
| Summarize the plot of '1984' by George Orwell. | The plot of 1984 is a dystopian novel written by George Orwell. The story is set in the early 1980s and the protagonist, the protagonist, is a young, young person who is caught up in a dark, dreary world. As a result, there are many themes prevalent among the characters, which include betrayal, greed, and a need for power that is out of reach. The events and characters take shape, and the book is set in the future where the protagonist is confronted with a ... |

There is still a lot of room for improvement, but the model generally is able to create proper answers and follow a 1-turn conversation.

As the instruction finetuning data is 1-turn only, the model will generally only be able to awnser one question at a time (somewhat) reliably.

---
# Implementation Details

## Base Model Training

The first step is training the model on a general text corpus of clean data. Just like Karpathy, I use Huggingface's [FineWeb-Edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu) ataset which is an english-only high-quality text dataset. The dataset is fairly homogeneous, which makes it easier to train on than other commonly used datasets like OpenWebText.

I make some changes to the dataloader functions in comparison to Karpathy's implementation. First, I split the FineWeb-Edu dataset at the document level, ensuring no document appears in both training and validation sets. This prevents data leakage and ensures a clean evaluation. Secondly, I also provide random batches of training data during training, and use a fixed validation batch throughout the training. The randomness of training data helps to avoid domain drift, as the documents of FineWeb-Edu can be very long and therefore the model just starts to memorize certain domains. This random approach was a game changer when trying to get the validation loss under 3.8 nats.

## Instruction Finetuning

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

# TO DO's
- ~Core Architecture: GPT-2 Large implementation~
- ~Data Pipeline: Custom dataloader with leakage prevention~
- ~Optimization: FlashAttention integration~
- ~Optimization: Switch to Muon Optimizer~
- Scaling: Train on >10B tokens (Chinchilla optimality)
- Deployment: HF Spaces demo
- Alignment: Implement DPO or other RLHF 
