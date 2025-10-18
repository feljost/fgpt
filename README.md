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

The expected xent floor is around 3.2 - 3.8 according to ChatGPT. I made it to a around 3.5 training loss and 3.9 val loss. 

![Training Loss Curve v0.1](/images/train-loss.png)

The Hellaswag eval is quite terrible, sitting at around 26% - 28% (depending on the run). This is barely better than random guessing which would be 25%. We are still quite far off from emerging intelligence. We can observe slight overfitting starting roughly around 20k steps.

### Sample Outputs

#### Base Model

The table below shows how the sample outputs evolved with the steps of training. You can see that the model learns rough grammar and does not mix up tokens that don't go together (for the most part). It also starts to stick to semantic topics better. The input text is "Once upon a time", after which the model generates the rest.

| Step | Output |
|------|--------|
| 0    | Once upon a timebattle RM steroids AgainFootballgments mobile manifests Krishna Lynnscroll Ey Residents MongoShe subconscious the reports braceograp Soft toxinsHuh Finch SPR Stonerid Bos Psyyles Provider ca realizes shelters numberingshock dissentduction dreadfuladminist arrangement Happy chickens midfielder blurryChurch sensibilitiesMenu Charity shack |
| 10'000    | Once upon a time are then lying actually at rest. The Copenhagenosphere movement was then out of Telespace yet, by the indicator its kinds were moving in very tall days, awaiting unfolding success. In the twentieth century, many of the most famousties of the Saturn |
| 20'000    | Once upon a time, and the development of ability will be immensely important to existing technologies and technologies across the representing lab functions. The Institutional Review of Physics, Volume 99, No. 2 above, presented the teeth to Sc Removal Process as a matter of a coordinated |
| 30'000    | Once upon a time. Explain and answer questions using these questions carefully during class time. PRATECHICAL guest story: prepare the right job of a episodes to ensure your students will work smoothly. If you are working on a line, small or large, or |

#### Instruction Finetuned Model

Pre finetuning the model has no understanding of question-awnsering or assistant style conversation. The autocomplete outputs look like this:

| Prompt | Output |
|------|--------|
| What is the capital of France? |  - Fact 1: Even France has a capital. - Fact 2: the9 city and boroughs are in ... |
| hello | 2 + 2 = negative (a) How do we get three chords to produce the Allegro in the key of ... |

After finetuning the outputs are actually not too bad:

| Prompt | Output |
|------|--------|
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
pip install numpy tiktoken datasets tqdm torch matplotlib pandas
```


### FineWeb-Edu Data Preparation

Run `fineweb.py` to download and tokenize the FineWeb-Edu dataset. Sharded data will be saved to the `edu_fineweb100B` directory for use in LLM training.

```sh
python fineweb.py
```

### Base Model Training Loop

The base training file does the following: 
- Loads or optionally resumes a FGPT model.
- Runs training with AdamW optimizer and cosine LR scheduler with warmup steps.
- Logs metrics (loss, gradient norm, tokens/sec, LR).
- Periodically evaluates on validation and HellaSwag.
- Saves checkpoints every 5000 steps.

```sh
python base_train.py
```

### Instruction Finetuning Training Loop

The instruction finetuning loosly follows rasbt's Chapter 7:
- Turns the instruct_data.json into Phi-3 style prompts for training
- Generates some example outputs pre finetuning 
- Finetunes the model and saves new model weights
- Generates some final example outputs post finetuning

```sh
python instruct.py
```


# TO DO's
- Performance Tweaks
    - Instruction Finetuning DataLoader
    - Fineweb-Edu Download (currently takes multiple hours)
    - Switch to FlashAttention for faster training
    - Switch to Muon Optimizer
- More stable and 
- Upload model to HuggingFace and deploy in Space
- Better visualize and document instruction finetunung
- Preference-optimizitaion (DPO or other easy to implement approach)
