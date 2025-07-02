import tiktoken
import torch
from torch.nn import functional as F

def model_inference(model, enc = tiktoken.get_encoding("gpt2")):
    # Inference
    max_length = 50
    tokens = enc.encode("Hello")  # encode a prompt
    # add batch dimension and move to GPU
    tokens = torch.tensor(tokens, dtype=torch.long, device="cuda").unsqueeze(0)
    x = tokens.to("cuda")  # move to GPU


    # generate tokens

    for _ in range(max_length):
        logits, loss = model(x)  # (B, T, vocab_size)
        logits = logits[
            :, -1, :
        ]  # take the last token's logits (B, vocab_size) --> we only care about the next token
        probs = F.softmax(logits, dim=-1)  # convert to probabilities
        # skipped: temperature and top-k sampling
        next_token = torch.multinomial(
            probs, num_samples=1
        )  # sample from the distribution
        x = torch.cat((x, next_token), dim=1)  # append the new token to the sequence
    
    decoded_output = enc.decode([token for token in x[0].tolist() if token <= 50257])
    
    return x, " ".join(decoded_output.split())  # decode the output