from typing import Literal
from fgpt.tokenizer import tokenizer
from fgpt.tokenizer import special_tokens
import torch
from torch.nn import functional as F
from fgpt.model import FGPT
from fgpt.model import FGPTConfig
from fgpt.model import load_model


def model_inference(
    model: FGPT,
    prompt: str = "Hello",
    tokenizer=tokenizer,
    max_tokens=50,
    top_k: int | None = 20,
    temperature: float = 1.0,
    generation_type: Literal["autocomplete", "conversational"] = "autocomplete",
):
    """Generates text from the given model and prompt.
    Args:
        model: The FGPT model for inference.
        prompt (str): The input prompt text.
        tokenizer: The tokenizer to encode/decode text.
        max_tokens (int): Maximum number of tokens to generate.
        top_k (int): If specified, use top-k sampling with this value.
        type (str): Type of generation, either 'autocomplete' or 'conversational'.
    Returns:
        generated_tokens (tensor): The generated token IDs.
        generated_text (str): The generated text.
    """

    if generation_type == "conversational":
        # For conversational, we can add special tokens if needed
        input_tokens = []
        input_tokens += [special_tokens["<|user|>"]]
        input_tokens += tokenizer.encode(prompt)
        input_tokens += [special_tokens["<|assistant|>"]]
    elif generation_type == "autocomplete":
        input_tokens = tokenizer.encode(prompt)  # encode a prompt
    else:
        raise ValueError(f"Unknown generation type: {generation_type}")

    # add batch dimension and move to GPU
    input_tokens = torch.tensor(
        input_tokens, dtype=torch.long, device="cuda"
    ).unsqueeze(0)
    x = input_tokens.to("cuda")  # move to GPU

    # generate tokens
    eos_token_id = special_tokens["<|endoftext|>"]
    for _ in range(max_tokens):
        logits, loss = model(x)  # (B, T, vocab_size)
        logits = logits[
            :, -1, :
        ]  # take the last token's logits (B, vocab_size) --> we only care about the next token

        if top_k is not None:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = -float("Inf")

        logits = logits / temperature  # apply temperature

        probs = F.softmax(logits, dim=-1)  # convert to probabilities
        # skipped: temperature and top-k sampling
        next_token = torch.multinomial(
            probs, num_samples=1
        )  # sample from the distribution
        x = torch.cat((x, next_token), dim=1)  # append the new token to the sequence

        if next_token.item() == eos_token_id:
            break  # stop if EOS token is generated

    if generation_type == "conversational":
        # remove the input prompt tokens for conversational
        x = x[:, input_tokens.size(1) :]

    decoded_output = tokenizer.decode(
        [token for token in x[0].tolist() if token <= 50258]
    )

    return x, "".join(decoded_output)  # decode the output


if __name__ == "__main__":
    model_weights_path = "/home/ubuntu/fgpt-base/model_weights_20251016_1458.pth"
    model = load_model(model_weights_path=model_weights_path, device="cuda")
    prompt = "Once upon a time"
    generated_tokens, generated_text = model_inference(
        model,
        prompt=prompt,
        max_tokens=100,
    )
    print("Generated text:")
    print(generated_text)
    print("done")
