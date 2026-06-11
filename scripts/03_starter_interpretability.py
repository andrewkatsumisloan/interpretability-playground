# main.py
"""
Starter mechanistic interpretability starter script.

What it does:
1. Loads GPT-2 medium with TransformerLens.
2. Runs two prompts:
   - clean:     "The capital of France is"
   - corrupted: "The capital of Italy is"
3. Prints the model's top next-token predictions.
4. Caches internal activations from both runs.
5. Performs a simple activation patching sweep:
   - Run the corrupted prompt.
   - At each layer, replace the final-token residual stream with the clean one.
   - Measure how much this pushes the model toward " Paris" over " Rome".

Run:

    python main.py

Optional:

    python main.py --device cpu
    python main.py --model gpt2-medium
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from transformer_lens import HookedTransformer


CLEAN_PROMPT = "The capital of France is"
CORRUPTED_PROMPT = "The capital of Italy is"

CLEAN_ANSWER = " Paris"
CORRUPTED_ANSWER = " Rome"


def choose_device(requested: str) -> str:
    if requested != "auto":
        return requested

    if torch.cuda.is_available():
        return "cuda"

    if torch.backends.mps.is_available():
        return "mps"

    return "cpu"


def get_answer_token_id(model: HookedTransformer, answer: str) -> int:
    """
    For GPT-2-style tokenizers, leading spaces matter.
    ' Paris' is usually one token; 'Paris' may be different.
    """
    token_ids = model.to_tokens(answer, prepend_bos=False).squeeze(0)

    if len(token_ids) != 1:
        raise ValueError(
            f"Expected answer {answer!r} to be one token, but got tokens: "
            f"{model.to_str_tokens(answer, prepend_bos=False)}"
        )

    return int(token_ids.item())


def top_next_tokens(
    model: HookedTransformer,
    logits: torch.Tensor,
    k: int = 10,
) -> list[tuple[str, float]]:
    """
    Return the top-k next-token predictions from logits.
    logits shape: [batch, position, vocab]
    """
    final_logits = logits[0, -1]
    probs = final_logits.softmax(dim=-1)
    top_probs, top_ids = probs.topk(k)

    return [
        (model.to_string(int(tok_id)), float(prob))
        for tok_id, prob in zip(top_ids, top_probs)
    ]


def logit_diff(
    logits: torch.Tensor,
    clean_answer_id: int,
    corrupted_answer_id: int,
) -> float:
    """
    Positive means the model prefers the clean answer over the corrupted answer.

    In this toy example:
      clean answer     = " Paris"
      corrupted answer = " Rome"

    So higher logit diff means "more Paris-like".
    """
    final_logits = logits[0, -1]
    return float(final_logits[clean_answer_id] - final_logits[corrupted_answer_id])


def print_predictions(
    model: HookedTransformer,
    name: str,
    prompt: str,
    logits: torch.Tensor,
    clean_answer_id: int,
    corrupted_answer_id: int,
) -> None:
    print("\n" + "=" * 80)
    print(name)
    print("=" * 80)
    print(f"Prompt: {prompt!r}")
    print(f'Logit diff "{CLEAN_ANSWER}" - "{CORRUPTED_ANSWER}": '
          f"{logit_diff(logits, clean_answer_id, corrupted_answer_id):.3f}")
    print("\nTop next-token predictions:")

    for token, prob in top_next_tokens(model, logits, k=10):
        print(f"  {token!r:>12}  p={prob:.4f}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="gpt2-medium")
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "cuda", "mps"],
        help="Use cpu if mps gives you trouble on macOS.",
    )
    parser.add_argument(
        "--out",
        default="runs/patching_results.json",
        help="Where to save the activation patching results.",
    )
    args = parser.parse_args()

    device = choose_device(args.device)
    print(f"Loading {args.model!r} on {device!r}...")

    torch.set_grad_enabled(False)

    model = HookedTransformer.from_pretrained(args.model, device=device)
    model.eval()

    clean_answer_id = get_answer_token_id(model, CLEAN_ANSWER)
    corrupted_answer_id = get_answer_token_id(model, CORRUPTED_ANSWER)

    clean_tokens = model.to_tokens(CLEAN_PROMPT)
    corrupted_tokens = model.to_tokens(CORRUPTED_PROMPT)

    print("\nClean tokens:")
    print(model.to_str_tokens(clean_tokens[0]))

    print("\nCorrupted tokens:")
    print(model.to_str_tokens(corrupted_tokens[0]))

    if clean_tokens.shape != corrupted_tokens.shape:
        raise ValueError(
            "For this simple patching demo, prompts need the same token length.\n"
            f"Clean shape:     {tuple(clean_tokens.shape)}\n"
            f"Corrupted shape: {tuple(corrupted_tokens.shape)}"
        )

    # -------------------------------------------------------------------------
    # 1. Run normally and cache activations.
    # -------------------------------------------------------------------------

    clean_logits, clean_cache = model.run_with_cache(clean_tokens)
    corrupted_logits, corrupted_cache = model.run_with_cache(corrupted_tokens)

    print_predictions(
        model,
        "Clean run",
        CLEAN_PROMPT,
        clean_logits,
        clean_answer_id,
        corrupted_answer_id,
    )

    print_predictions(
        model,
        "Corrupted run",
        CORRUPTED_PROMPT,
        corrupted_logits,
        clean_answer_id,
        corrupted_answer_id,
    )

    clean_logit_diff = logit_diff(clean_logits, clean_answer_id, corrupted_answer_id)
    corrupted_logit_diff = logit_diff(corrupted_logits, clean_answer_id, corrupted_answer_id)

    print("\n" + "=" * 80)
    print("Activation cache sanity checks")
    print("=" * 80)

    example_name = "blocks.5.hook_resid_pre"
    example_activation = clean_cache[example_name]

    print(f"{example_name} shape: {tuple(example_activation.shape)}")
    print(
        f"{example_name} final-token norm: "
        f"{example_activation[0, -1].norm().item():.3f}"
    )

    # -------------------------------------------------------------------------
    # 2. Activation patching.
    # -------------------------------------------------------------------------
    #
    # We patch the residual stream at the final token position.
    #
    # For each layer:
    #   - Run the corrupted prompt.
    #   - Replace blocks.{layer}.hook_resid_pre[:, -1, :]
    #     with the same activation from the clean run.
    #   - Measure whether the model becomes more likely to output " Paris".
    #
    # This is a simple version of activation patching / causal tracing.
    # -------------------------------------------------------------------------

    results = []

    print("\n" + "=" * 80)
    print("Residual-stream patching sweep")
    print("=" * 80)
    print(
        f"{'layer':>5} | {'patched logit diff':>18} | "
        f"{'recovery':>9} | activation"
    )
    print("-" * 80)

    denom = clean_logit_diff - corrupted_logit_diff

    for layer in range(model.cfg.n_layers):
        activation_name = f"blocks.{layer}.hook_resid_pre"

        def patch_final_token_resid_pre(
            activation: torch.Tensor,
            hook,
            activation_name: str = activation_name,
        ) -> torch.Tensor:
            patched = activation.clone()
            patched[:, -1, :] = clean_cache[activation_name][:, -1, :]
            return patched

        patched_logits = model.run_with_hooks(
            corrupted_tokens,
            fwd_hooks=[(activation_name, patch_final_token_resid_pre)],
        )

        patched_logit_diff = logit_diff(
            patched_logits,
            clean_answer_id,
            corrupted_answer_id,
        )

        if abs(denom) > 1e-6:
            recovery = (patched_logit_diff - corrupted_logit_diff) / denom
        else:
            recovery = float("nan")

        row = {
            "layer": layer,
            "activation": activation_name,
            "patched_logit_diff": patched_logit_diff,
            "recovery": recovery,
        }
        results.append(row)

        print(
            f"{layer:5d} | {patched_logit_diff:18.3f} | "
            f"{recovery:9.2%} | {activation_name}"
        )

    # -------------------------------------------------------------------------
    # 3. Save results.
    # -------------------------------------------------------------------------

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "model": args.model,
        "device": device,
        "clean_prompt": CLEAN_PROMPT,
        "corrupted_prompt": CORRUPTED_PROMPT,
        "clean_answer": CLEAN_ANSWER,
        "corrupted_answer": CORRUPTED_ANSWER,
        "clean_logit_diff": clean_logit_diff,
        "corrupted_logit_diff": corrupted_logit_diff,
        "results": results,
    }

    out_path.write_text(json.dumps(payload, indent=2))
    print(f"\nSaved results to {out_path}")

    best = max(results, key=lambda x: x["recovery"])
    print("\nMost Paris-restoring patch:")
    print(
        f"  layer={best['layer']}, activation={best['activation']}, "
        f"recovery={best['recovery']:.2%}"
    )


if __name__ == "__main__":
    main()