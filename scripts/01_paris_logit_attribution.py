import torch
import pandas as pd
import matplotlib.pyplot as plt
from transformer_lens import HookedTransformer

torch.set_grad_enabled(False)

device = "mps" if torch.backends.mps.is_available() else "cpu"

model = HookedTransformer.from_pretrained(
    "gpt2-medium",
    device=device,
)

prompt = "The capital of France is"
answer = " Paris"

logits, cache = model.run_with_cache(prompt)

answer_token = model.to_single_token(answer)
final_logits = logits[0, -1]
answer_logit = final_logits[answer_token].item()
answer_prob = final_logits.softmax(dim=-1)[answer_token].item()
answer_rank = int((final_logits > final_logits[answer_token]).sum().item() + 1)

print(f"Prompt: {prompt!r}")
print(f"Answer token: {answer!r} -> {answer_token}")
print(f"Final logit: {answer_logit:.3f}")
print(f"Final probability: {answer_prob:.4f}")
print(f"Final rank: {answer_rank}")

# Decompose the residual stream into additive components:
# embed, pos_embed, each attention output, each MLP output, etc.
residual_stream, labels = cache.decompose_resid(
    return_labels=True,
    mode="all",
)

# Compute how much each component directly contributes to the " Paris" logit.
# This is a "direct logit attribution" style view.
logit_attrs = cache.logit_attrs(residual_stream, answer)

# Shape is usually [component, batch, position].
# We care about the final position of the single batch item.
attrs_final_pos = logit_attrs[:, 0, -1].detach().cpu()

df = pd.DataFrame({
    "component": labels,
    "paris_logit_attr": attrs_final_pos.numpy(),
})

df = df.sort_values("paris_logit_attr", ascending=False)

print("\nTop positive contributors to ' Paris':")
print(df.head(20).to_string(index=False))

print("\nTop negative contributors to ' Paris':")
print(df.tail(20).to_string(index=False))

df.to_csv("outputs/paris_logit_attribution.csv", index=False)

# Plot top contributors
top_df = df.head(25).iloc[::-1]

plt.figure(figsize=(10, 8))
plt.barh(top_df["component"], top_df["paris_logit_attr"])
plt.xlabel("Direct contribution to ' Paris' logit")
plt.title("GPT-2 small: components pushing toward ' Paris'")
plt.tight_layout()
plt.savefig("outputs/paris_logit_attribution.png", dpi=200)

print("\nSaved:")
print("outputs/paris_logit_attribution.csv")
print("outputs/paris_logit_attribution.png")