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
<|user|> What is the capital of France? <|assistant|>
```

After finetuning, the model generally gives short, more relevant answers and can handle simple 1-turn conversations.

## Evaluation

WIP WIP WIP

## Results

For the base model I achieve ~3 cross entropy nats on the validation set, which is a good result and about what we can expect without many advanced tweaks. As we are only training on english educational content, our dataset is fairly homogenous compared to multilanguage datasets. If we were to train on something like FineWeb-Edu2 (the multilingual version) or OpenWebText, we would expect a higher loss.

We do not observe any signs of overfitting or other instabilities. In fact, we could probably even train a bit longer if we want to adhere by the Chinchilla Scaling Law.

![Loss Curves](/report/images/train-loss.png)

Every 10k steps I also evaluate the HellaSwag accuracy of the base model, which takes the logits of all responses (given the input) and evaluates which one is the most likely. We get a 31% accuracy which is significantly better than random guessing (=25%). Instruction finetuned version will follow.

![HellaSwag Base Model](/report/images/hellaswag-base.png)

### Sample Outputs

#### Base Model

The table below shows how the sample outputs evolved with the steps of training. You can see that the model learns rough grammar and does not mix up tokens that don't go together (for the most part). It also starts to stick to semantic topics better. The input text is "Once upon a time" after which the model generates the rest.

| Step | Output |
|------|--------|
| 0    | _Once upon a time_ reproduction allegiance Freeze crises COVER Face database tet psychoticSET178 brothersyt distinctions UNITED endorserickyregooeval medicineOHN Boll UN Supporteditching pinch insol\u30c4 Regist packing engineered circumst go Tunnel PCs Lydia genre 40ho Magickabasic bullies Dharmahar-.PART TCU Civilusions |
| 100'000 | _Once upon a time_, they enjoyed walking through a nearby lane of spooky video star terrible death in a sea-boat that collided with their boat, leaving the seas unharmed and dieling all of humour. Lucie and her crew traveled across this encounter and David |
| 200'000 | _Once upon a time_, the earth was just as white as any rainbow in the sky, but the now white ice had begun to melt and lose its colors. When you think about freezing with ice – the shlatth grade it with the ice seems to me much colder |
| 300'000  | ... |

#### Instruction Finetuned Model

Pre finetuning the model has no understanding of question-answering or assistant style conversation. The autocomplete outputs look like this:

| Prompt | Output |
|--------|--------|
| What is the capital of France? |  - Fact 1: Even France has a capital. - Fact 2: the9 city and boroughs are in ... |
| hello | 2 + 2 = negative (a) How do we get three chords to produce the Allegro in the key of ... |

After finetuning the outputs are actually not too bad:

| Prompt | Output |
|--------|--------|
| What is the capital of France? | The capital of France is Paris |
| What is 2 + 2? | 2 - 2 = 4 |
| Define 'cat'. | Cat is a cat, but how big is a cat? |

There is still a lot of room for improvement, but the model generally is able to create proper answers and follow a 1-turn conversation.

## Future To Do's

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
    - Fineweb-Edu Download (currently takes multiple hours)
    - Switch to FlashAttention for faster training
    - Switch to Muon Optimizer
- Train on much more tokens to abide by Chinchilla Scaling Law
- Upload model to HuggingFace and deploy in HF-spaces
- Better visualize and document instruction finetunung
- Preference-optimizitaion (DPO or other easy to implement approach)
