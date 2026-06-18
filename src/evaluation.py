"""Evaluation utilities: ROUGE-1, ROUGE-L, and per-language breakdowns."""

import numpy as np
import pandas as pd
from rouge_score import rouge_scorer


ROUGE1_WEIGHT = 0.37
ROUGEL_WEIGHT = 0.37
LLM_WEIGHT = 0.26


class WhitespaceTokenizer:
    """Language-agnostic tokenizer — safe for African scripts (Ge'ez, Akan, etc.)."""

    def tokenize(self, text: str) -> list[str]:
        return str(text).strip().split() if text else []


_SCORER = rouge_scorer.RougeScorer(
    ["rouge1", "rougeL"],
    tokenizer=WhitespaceTokenizer(),
    use_stemmer=False,
)


def rouge_single(prediction: str, reference: str) -> dict:
    s = _SCORER.score(str(reference), str(prediction))
    return {
        "rouge1_f1": s["rouge1"].fmeasure,
        "rougeL_f1": s["rougeL"].fmeasure,
    }


def compute_rouge(predictions: list[str], references: list[str]) -> dict:
    r1, rl = [], []
    for pred, ref in zip(predictions, references):
        s = _SCORER.score(str(ref), str(pred))
        r1.append(s["rouge1"].fmeasure)
        rl.append(s["rougeL"].fmeasure)
    return {
        "rouge1_f1": float(np.mean(r1)) if r1 else 0.0,
        "rougeL_f1": float(np.mean(rl)) if rl else 0.0,
        "weighted": float(ROUGE1_WEIGHT * np.mean(r1) + ROUGEL_WEIGHT * np.mean(rl)) if r1 else 0.0,
    }


def compute_rouge_by_language(
    predictions: list[str],
    references: list[str],
    languages: list[str],
) -> pd.DataFrame:
    lang_arr = np.array(languages)
    results = {}
    for lang in np.unique(lang_arr):
        mask = lang_arr == lang
        preds_l = [p for p, m in zip(predictions, mask) if m]
        refs_l = [r for r, m in zip(references, mask) if m]
        scores = compute_rouge(preds_l, refs_l)
        scores["n"] = int(mask.sum())
        results[lang] = scores
    return pd.DataFrame(results).T[["n", "rouge1_f1", "rougeL_f1", "weighted"]]


def compute_overall_weighted_score(rouge1: float, rougeL: float, llm_judge: float = 0.0) -> float:
    """Full competition weighted metric (ROUGE-1 × 0.37 + ROUGE-L × 0.37 + LLM × 0.26)."""
    return ROUGE1_WEIGHT * rouge1 + ROUGEL_WEIGHT * rougeL + LLM_WEIGHT * llm_judge
