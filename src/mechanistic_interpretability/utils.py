from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

import torch
from transformer_lens import HookedTransformer


CLEAN_PROMPT = "The capital of France is"
CORRUPTED_PROMPT = "The capital of Italy is"

CLEAN_ANSWER = " Paris"
CORRUPTED_ANSWER = " Rome"


@dataclass(frozen=True)
class OutputPaths:
    root: Path
    tables: Path
    figures: Path
    runs: Path


def build_output_paths(root: str | Path) -> OutputPaths:
    output_root = Path(root)
    paths = OutputPaths(
        root=output_root,
        tables=output_root / "tables",
        figures=output_root / "figures",
        runs=output_root / "runs",
    )

    for path in (paths.tables, paths.figures, paths.runs):
        path.mkdir(parents=True, exist_ok=True)

    return paths


def model_slug(model_name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", model_name).strip("-")


def choose_device(requested: str) -> str:
    if requested != "auto":
        return requested

    if torch.cuda.is_available():
        return "cuda"

    if torch.backends.mps.is_available():
        return "mps"

    return "cpu"


def load_model(model_name: str, device: str) -> HookedTransformer:
    torch.set_grad_enabled(False)
    model = HookedTransformer.from_pretrained(model_name, device=device)
    model.eval()
    return model


def get_answer_token_id(model: HookedTransformer, answer: str) -> int:
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
    final_logits = logits[0, -1]
    probs = final_logits.softmax(dim=-1)
    top_probs, top_ids = probs.topk(k)

    return [
        (model.to_string(int(token_id)), float(prob))
        for token_id, prob in zip(top_ids, top_probs)
    ]


def logit_diff(
    logits: torch.Tensor,
    clean_answer_id: int,
    corrupted_answer_id: int,
) -> float:
    final_logits = logits[0, -1]
    return float(final_logits[clean_answer_id] - final_logits[corrupted_answer_id])


def print_prompt_tokens(model: HookedTransformer, name: str, tokens: torch.Tensor) -> None:
    print(f"\n{name} tokens:")
    print(model.to_str_tokens(tokens[0]))


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
    print(
        f'Logit diff "{CLEAN_ANSWER}" - "{CORRUPTED_ANSWER}": '
        f"{logit_diff(logits, clean_answer_id, corrupted_answer_id):.3f}"
    )
    print("\nTop next-token predictions:")

    for token, prob in top_next_tokens(model, logits, k=10):
        print(f"  {token!r:>12}  p={prob:.4f}")
