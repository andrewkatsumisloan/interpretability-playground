from __future__ import annotations

import argparse
import json

import matplotlib.pyplot as plt
import pandas as pd
import torch
from transformer_lens.utilities import get_act_name

from mechanistic_interpretability.utils import (
    CLEAN_ANSWER,
    CLEAN_PROMPT,
    CORRUPTED_ANSWER,
    CORRUPTED_PROMPT,
    build_output_paths,
    choose_device,
    get_answer_token_id,
    load_model,
    logit_diff,
    model_slug,
    print_predictions,
    print_prompt_tokens,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Inspect predictions, cached activations, and a simple activation "
            "patching sweep on a capital-city prompt pair."
        )
    )
    parser.add_argument("--model", default="gpt2-medium")
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "cuda", "mps"],
        help="Use cpu if mps gives you trouble on macOS.",
    )
    parser.add_argument(
        "--hook",
        default="resid_pre",
        choices=["resid_pre", "resid_post"],
        help="Residual stream hook point to patch at the final token.",
    )
    parser.add_argument(
        "--out-dir",
        default="outputs",
        help="Directory for generated tables, figures, and run artifacts.",
    )
    args = parser.parse_args()

    outputs = build_output_paths(args.out_dir)
    device = choose_device(args.device)

    print(f"Loading {args.model!r} on {device!r}...")
    model = load_model(args.model, device)

    # The task is a minimal contrast pair:
    # clean prompt      -> should favor " Paris"
    # corrupted prompt  -> should favor " Rome"
    # The intervention asks which layer activations from the clean run restore
    # Paris-like behavior when inserted into the corrupted run.
    clean_answer_id = get_answer_token_id(model, CLEAN_ANSWER)
    corrupted_answer_id = get_answer_token_id(model, CORRUPTED_ANSWER)

    clean_tokens = model.to_tokens(CLEAN_PROMPT)
    corrupted_tokens = model.to_tokens(CORRUPTED_PROMPT)

    print_prompt_tokens(model, "Clean", clean_tokens)
    print_prompt_tokens(model, "Corrupted", corrupted_tokens)

    if clean_tokens.shape != corrupted_tokens.shape:
        raise ValueError(
            "For this simple patching demo, prompts need the same token length.\n"
            f"Clean shape:     {tuple(clean_tokens.shape)}\n"
            f"Corrupted shape: {tuple(corrupted_tokens.shape)}"
        )

    # Cache only the hook point we plan to patch. This keeps the example focused
    # and uses less memory than caching every activation in the model.
    names_filter = lambda name: name.endswith(f"hook_{args.hook}")
    clean_logits, clean_cache = model.run_with_cache(
        clean_tokens,
        names_filter=names_filter,
    )
    corrupted_logits = model(corrupted_tokens)

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
    corrupted_logit_diff = logit_diff(
        corrupted_logits,
        clean_answer_id,
        corrupted_answer_id,
    )

    print("\n" + "=" * 80)
    print("Activation cache sanity checks")
    print("=" * 80)

    example_layer = min(5, model.cfg.n_layers - 1)
    example_name = get_act_name(args.hook, example_layer)
    example_activation = clean_cache[example_name]

    print(f"{example_name} shape: {tuple(example_activation.shape)}")
    print(
        f"{example_name} final-token norm: "
        f"{example_activation[0, -1].norm().item():.3f}"
    )

    # Logit difference is the behavior metric:
    #   positive = more Paris than Rome
    #   negative = more Rome than Paris
    # Recovery normalizes patched behavior onto a scale where 0 is corrupted
    # behavior and 1 is clean behavior.
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
        activation_name = get_act_name(args.hook, layer)

        def patch_final_token_residual_stream(
            activation: torch.Tensor,
            hook,
            activation_name: str = activation_name,
        ) -> torch.Tensor:
            # Clone before editing so the hook is side-effect free. We only
            # replace the final-token residual vector, not the whole sequence.
            patched = activation.clone()
            patched[:, -1, :] = clean_cache[activation_name][:, -1, :]
            return patched

        patched_logits = model.run_with_hooks(
            corrupted_tokens,
            fwd_hooks=[(activation_name, patch_final_token_residual_stream)],
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

    df = pd.DataFrame(results)
    stem = f"{model_slug(args.model)}_france_italy_{args.hook}_patch"
    csv_path = outputs.tables / f"{stem}.csv"
    figure_path = outputs.figures / f"{stem}.png"
    json_path = outputs.runs / f"{stem}.json"

    # Save the same sweep in three forms: a CSV for analysis, a plot for quick
    # inspection, and JSON with enough metadata to know how the run was made.
    df.to_csv(csv_path, index=False)

    plt.figure(figsize=(10, 5))
    plt.plot(df["layer"], df["recovery"], marker="o")
    plt.axhline(0, linestyle="--")
    plt.axhline(1, linestyle="--")
    plt.xlabel(f"Layer patched: {args.hook} at final token")
    plt.ylabel("Recovery fraction")
    plt.title("Patching France residual stream into Italy prompt")
    plt.tight_layout()
    plt.savefig(figure_path, dpi=200)

    payload = {
        "model": args.model,
        "device": device,
        "hook": args.hook,
        "clean_prompt": CLEAN_PROMPT,
        "corrupted_prompt": CORRUPTED_PROMPT,
        "clean_answer": CLEAN_ANSWER,
        "corrupted_answer": CORRUPTED_ANSWER,
        "clean_logit_diff": clean_logit_diff,
        "corrupted_logit_diff": corrupted_logit_diff,
        "results": results,
    }

    json_path.write_text(json.dumps(payload, indent=2))

    print("\nSaved:")
    print(csv_path)
    print(figure_path)
    print(json_path)

    best = max(results, key=lambda x: x["recovery"])
    print("\nMost Paris-restoring patch:")
    print(
        f"  layer={best['layer']}, activation={best['activation']}, "
        f"recovery={best['recovery']:.2%}"
    )


if __name__ == "__main__":
    main()
