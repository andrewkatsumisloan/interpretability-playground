import torch
from transformer_lens import HookedTransformer

torch.set_grad_enabled(False)

device = "mps" if torch.backends.mps.is_available() else "cpu"

model = HookedTransformer.from_pretrained(
    "gpt2-medium",
    device=device,
)

prompt = "The capital of France is"
tokens = model.to_tokens(prompt)
logits, cache = model.run_with_cache(tokens)

next_logits = logits[0, -1]
top = torch.topk(next_logits, k=15)

print(f"Device: {device}")
print("Prompt tokens:")
print(model.to_str_tokens(tokens[0]))

print("\nTop predictions:")
for rank, token_id in enumerate(top.indices, start=1):
    token = model.to_string(token_id)
    score = next_logits[token_id].item()
    print(f"{rank:2d}. {token!r:15s} logit={score:.3f}")

print("\nExample cached activation keys:")
for key in list(cache.cache_dict.keys())[:20]:
    print(key)