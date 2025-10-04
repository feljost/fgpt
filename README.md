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

The expected xent floor is around 3.2 - 3.8 according to ChatGPT. I made it to a around 3.5 training loss and 3.9 val loss. 

![Training Loss Curve v0.1](/images/train-loss.png)

The Hellaswag eval is quite terrible, sitting at around 26% - 28% (depending on the run). This is barely better than random guessing which would be 25%. We are still quite far off from emerging intelligence. We can observe slight overfitting starting roughly around 20k steps.

#### Sample outputs:

The table below shows how the sample outputs evolved with the steps of training. You can see that the model learns rough grammar and does not mix up tokens that don't go together (for the most part). It also starts to stick to semantic topics better. The input text is "Once upon a time", after which the model generates the rest.

| Step | Output |
|------|--------|
| 0    | Once upon a timebattle RM steroids AgainFootballgments mobile manifests Krishna Lynnscroll Ey Residents MongoShe subconscious the reports braceograp Soft toxinsHuh Finch SPR Stonerid Bos Psyyles Provider ca realizes shelters numberingshock dissentduction dreadfuladminist arrangement Happy chickens midfielder blurryChurch sensibilitiesMenu Charity shack |
| 10'000    | Once upon a time are then lying actually at rest. The Copenhagenosphere movement was then out of Telespace yet, by the indicator its kinds were moving in very tall days, awaiting unfolding success. In the twentieth century, many of the most famousties of the Saturn |
| 20'000    | Once upon a time, and the development of ability will be immensely important to existing technologies and technologies across the representing lab functions. The Institutional Review of Physics, Volume 99, No. 2 above, presented the teeth to Sc Removal Process as a matter of a coordinated |
| 30'000    | ... |

## Future To Do's

- Instruction Finetuning
- Preference-optimizitaion (DPO or other easy to implement approach)
