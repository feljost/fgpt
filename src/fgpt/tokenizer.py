"""GPT2 tokenizer with added special tokens for instuction tuning."""

import tiktoken


tokenizer = tiktoken.get_encoding("gpt2")


eos_token = tokenizer._special_tokens["<|endoftext|>"]
special_tokens = {
    "<|endoftext|>": eos_token,
    "<|user|>": eos_token + 1,
    "<|assistant|>": eos_token + 2,
}
tokenizer = tiktoken.Encoding(
    name="gpt2+special",
    pat_str=tokenizer._pat_str,
    mergeable_ranks=tokenizer._mergeable_ranks,
    special_tokens={**tokenizer._special_tokens, **special_tokens},
)
