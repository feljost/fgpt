# Overview

This repository provides scripts and notebooks for to train a conversational LLM FineWeb-Edu Dataset (and some instruction finetuning data provided by rasbt on GitHub). The purpose of the repo is not to create a SOTA model but rather to experiment and learn. The code is loosely based on Karphaty's youtube videos and Sebastian Rashke's LLM from scratch code. 

The model follows the GPT2 architecture and uses the same GPT2 tokenizer along with some added tokens to allow for conversational style in Phi3-Style prompting. Size wise, it is roughly at GPT2-Large. It has 712M Parameters with 32 layers and 24 heads. The learning rate schedule and other hyperparameters are also different to allow for more fun when tinkering with the architecture. The training happened on a single NVIDIA GH200.

Model performance is primarily assessed by monitoring validation & training loss, with additional evaluation using the HellaSwag benchmark. Example outputs of the model are also checked over time.

## Base Model Training

Unlike Karpathy's original code, this repository splits the FineWeb-Edu dataset at the document level, ensuring no document appears in both training and validation sets. This prevents data leakage and ensures a clean evaluation. I also provide random batches of training data during training, and use a fixed validation batch throughout the training. The randomnes of training data helps to avoid domain drift, as the documents of FineWeb-Edu can be very long and therefore the model just starts to memorize certain domains. This random approach was a game changer when trying to get the validation loss under 3.8 nats.

## Instruction Finetunung

After the base model is trained, it is fine-tuned on a small instruction dataset so it behaves a bit more like a conversational assistant.

The goal here isn’t to reach top performance but to help the model understand short question-answer patterns and simple user instructions.

```
<|user|>What is the capital of France?<|assistant|>The capital of France is Paris<|endoftext|>
```

After finetuning, the model generally gives short, more relevant answers and can handle simple 1-turn conversations.

## Evaluation

WIP WIP WIP

## Results

For the base model I achieve ~2.8 cross entropy nats on the validation set, which is a good result and about what we can expect without many advanced tweaks. As we are only training on english educational content, our dataset is fairly homogenous compared to multilanguage datasets. If we were to train on something like FineWeb-Edu2 (the multilingual version) or OpenWebText, we would expect a higher loss.

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

## Usage

You will need a strong GPU with cuda to run these scripts. If you don't have one locally, I suggest getting one in the cloud (I used lambda labs).

### Requirements

Install dependencies with:

```sh
pip install -e .
```

Depending on your version of CUDA, you might have to specify a specific torch version (see [pytorch.org](https://pytorch.org/)).

### Data Downloads

Run `fineweb_download.py` to download and tokenize the [FineWeb-Edu dataset](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu). Sharded data will be saved to the `edu_fineweb100B` directory for use in LLM training.

```sh
python fgpt.data.fineweb_download.py
```

You can download the (untokenized) [OpenAssistant Conversations Dataset (OASST1)](https://huggingface.co/datasets/OpenAssistant/oasst1) by running `oasst_download.py`.

```sh
python fgpt.data.oasst_download.py
```

### Base Model Training Loop

The base training file does the following: 
- Loads or optionally resumes a FGPT model.
- Runs training with AdamW optimizer and cosine LR scheduler with warmup steps.
- Logs metrics (loss, gradient norm, tokens/sec, LR).
- Periodically evaluates on validation and HellaSwag.
- Saves checkpoints every 5000 steps.

```sh
python fgpt.base_train.py
```

Before starting the training, you may want to adjust some of the model and training parameters in 
`model.py` and `base_train.py`, depending on your available compute and time commitment.

Please note that the training will only happen one 1 GPU at a time. You will need to adjust training loop if you want to do multi-GPU training.

### Instruction Finetuning Training Loop

The instruction finetuning loosely follows rasbt's Chapter 7:
- Turns the instruct_data.json into Phi-3 style prompts for training
- Generates some example outputs pre finetuning 
- Finetunes the model and saves new model weights
- Generates some final example outputs post finetuning

```sh
python fgpt.instruct_train.py
```


# TO DO's
- Performance Tweaks
    - Instruction Finetuning DataLoader
    - ~Base model DataLoader~
    - Fineweb-Edu Download (currently takes multiple hours)
    - ~Switch to FlashAttention for faster training~
    - Switch to Muon Optimizer
- Train on much more tokens to abide by Chinchilla Scaling Law
- Upload model to HuggingFace and deploy in HF-spaces
- Document instruction finetunung
- Preference-optimizitaion (DPO or other easy to implement approach)
