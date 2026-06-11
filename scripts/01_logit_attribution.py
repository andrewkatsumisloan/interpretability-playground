from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import pandas as pd

from mechanistic_interpretability.utils import (
    CLEAN_ANSWER,
    CLEAN_PROMPT,
    build_output_paths,
    choose_device,
    get_answer_token_id,
    load_model,
    model_slug,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Ask which residual-stream components directly raise or lower the "
            "Paris answer logit."
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
        "--out-dir",
        default="outputs",
        help="Directory for generated tables, figures, and run artifacts.",
    )
    args = parser.parse_args()

    outputs = build_output_paths(args.out_dir)
    device = choose_device(args.device)

    print(f"Loading {args.model!r} on {device!r}...")
    model = load_model(args.model, device)

    # Run the prompt once and keep TransformerLens' activation cache. The cache
    # stores intermediate residual streams, attention outputs, MLP outputs, etc.
    logits, cache = model.run_with_cache(CLEAN_PROMPT)

    # GPT-2 tokenizers are sensitive to leading spaces. This checks that
    # " Paris" is a single next-token answer before we score it.
    answer_token = get_answer_token_id(model, CLEAN_ANSWER)
    final_logits = logits[0, -1]
    answer_logit = final_logits[answer_token].item()
    answer_prob = final_logits.softmax(dim=-1)[answer_token].item()
    answer_rank = int((final_logits > final_logits[answer_token]).sum().item() + 1)

    print(f"Prompt: {CLEAN_PROMPT!r}")
    print(f"Answer token: {CLEAN_ANSWER!r} -> {answer_token}")
    print(f"Final logit: {answer_logit:.3f}")
    print(f"Final probability: {answer_prob:.4f}")
    print(f"Final rank: {answer_rank}")

    # Direct logit attribution treats the final residual stream as an additive
    # sum of components, then projects each component through the unembedding
    # direction for the target answer token.
    residual_stream, labels = cache.decompose_resid(
        return_labels=True,
        mode="all",
    )

    logit_attrs = cache.logit_attrs(residual_stream, CLEAN_ANSWER)
    attrs_final_pos = logit_attrs[:, 0, -1].detach().cpu()

    df = pd.DataFrame(
        {
            "component": labels,
            "paris_logit_attr": attrs_final_pos.numpy(),
        }
    ).sort_values("paris_logit_attr", ascending=False)

    print("\nTop positive contributors to ' Paris':")
    print(df.head(20).to_string(index=False))

    print("\nTop negative contributors to ' Paris':")
    print(df.tail(20).to_string(index=False))

    # Tables are easier to inspect programmatically; figures are easier for a
    # first visual pass over which components matter most.
    stem = f"{model_slug(args.model)}_paris_logit_attribution"
    csv_path = outputs.tables / f"{stem}.csv"
    figure_path = outputs.figures / f"{stem}.png"

    df.to_csv(csv_path, index=False)

    top_df = df.head(25).iloc[::-1]

    plt.figure(figsize=(10, 8))
    plt.barh(top_df["component"], top_df["paris_logit_attr"])
    plt.xlabel("Direct contribution to ' Paris' logit")
    plt.title(f"{args.model}: components pushing toward ' Paris'")
    plt.tight_layout()
    plt.savefig(figure_path, dpi=200)

    print("\nSaved:")
    print(csv_path)
    print(figure_path)


if __name__ == "__main__":
    main()
