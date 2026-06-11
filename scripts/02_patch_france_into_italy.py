import torch
import pandas as pd
import matplotlib.pyplot as plt
from transformer_lens import HookedTransformer, utils

torch.set_grad_enabled(False)

device = "mps" if torch.backends.mps.is_available() else "cpu"

model = HookedTransformer.from_pretrained(
    "gpt2-small",
    device=device,
)

clean_prompt = "The capital of France is"
corrupt_prompt = "The capital of Italy is"

clean_tokens = model.to_tokens(clean_prompt)
corrupt_tokens = model.to_tokens(corrupt_prompt)

print("Clean tokens:")
print(model.to_str_tokens(clean_tokens[0]))

print("\nCorrupt tokens:")
print(model.to_str_tokens(corrupt_tokens[0]))

assert clean_tokens.shape == corrupt_tokens.shape, (
    "For this simple patching script, prompts must tokenize to the same length."
)

paris_token = model.to_single_token(" Paris")
rome_token = model.to_single_token(" Rome")


def logit_diff(logits: torch.Tensor) -> float:
    """
    Positive means model prefers Paris over Rome.
    Negative means model prefers Rome over Paris.
    """
    final = logits[0, -1]
    return (final[paris_token] - final[rome_token]).item()


# Cache only residual stream after each layer from the clean run.
_, clean_cache = model.run_with_cache(
    clean_tokens,
    names_filter=lambda name: name.endswith("hook_resid_post"),
)

clean_logits = model(clean_tokens)
corrupt_logits = model(corrupt_tokens)

clean_score = logit_diff(clean_logits)
corrupt_score = logit_diff(corrupt_logits)

print(f"\nClean logit diff Paris - Rome:   {clean_score:.3f}")
print(f"Corrupt logit diff Paris - Rome: {corrupt_score:.3f}")

rows = []

for layer in range(model.cfg.n_layers):
    hook_name = utils.get_act_name("resid_post", layer)

    def patch_final_position_resid(activation, hook, hook_name=hook_name):
        # activation shape: [batch, position, d_model]
        activation[:, -1, :] = clean_cache[hook_name][:, -1, :]
        return activation

    patched_logits = model.run_with_hooks(
        corrupt_tokens,
        fwd_hooks=[(hook_name, patch_final_position_resid)],
    )

    patched_score = logit_diff(patched_logits)

    # Normalize: 0 = corrupted behavior, 1 = clean behavior.
    recovery = (patched_score - corrupt_score) / (clean_score - corrupt_score)

    rows.append({
        "layer": layer,
        "patched_logit_diff_paris_minus_rome": patched_score,
        "recovery_fraction": recovery,
    })

df = pd.DataFrame(rows)
print("\nPatching results:")
print(df.to_string(index=False))

df.to_csv("outputs/france_italy_resid_patch.csv", index=False)

plt.figure(figsize=(10, 5))
plt.plot(df["layer"], df["recovery_fraction"], marker="o")
plt.axhline(0, linestyle="--")
plt.axhline(1, linestyle="--")
plt.xlabel("Layer patched: resid_post at final token")
plt.ylabel("Recovery fraction")
plt.title("Patching France residual stream into Italy prompt")
plt.tight_layout()
plt.savefig("outputs/france_italy_resid_patch.png", dpi=200)

print("\nSaved:")
print("outputs/france_italy_resid_patch.csv")
print("outputs/france_italy_resid_patch.png")