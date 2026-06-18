"""Data loading, preprocessing, and augmentation utilities."""

import re
import csv
import unicodedata
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

TRAIN_FILENAME = "Train.csv"
VAL_FILENAME = "Val.csv"
TEST_FILENAME = "Test.csv"
SAMPLE_SUB_FILENAME = "SampleSubmission.csv"

SUBSET_TO_LANGUAGE = {
    "Eng": "English",
    "Aka": "Akan",
    "Lug": "Luganda",
    "Swa": "Swahili",
    "Amh": "Amharic",
}

SUBSET_TO_COUNTRY = {
    "Gha": "Ghana",
    "Eth": "Ethiopia",
    "Uga": "Uganda",
    "Ken": "Kenya",
}


def get_language_name(subset: str) -> str:
    prefix = subset.split("_")[0] if isinstance(subset, str) else "Eng"
    return SUBSET_TO_LANGUAGE.get(prefix, "English")


def get_country_name(subset: str) -> str:
    parts = subset.split("_") if isinstance(subset, str) else []
    suffix = parts[1] if len(parts) > 1 else "Uga"
    return SUBSET_TO_COUNTRY.get(suffix, "Unknown")


def clean_text(text: str) -> str:
    if pd.isna(text) or text is None:
        return ""
    text = str(text).strip()
    text = re.sub(r"\s+", " ", text)
    return text


def load_data(data_dir: str | Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    data_dir = Path(data_dir)
    train = pd.read_csv(data_dir / TRAIN_FILENAME)
    val = pd.read_csv(data_dir / VAL_FILENAME)
    test = pd.read_csv(data_dir / TEST_FILENAME)
    sample_sub = pd.read_csv(data_dir / SAMPLE_SUB_FILENAME)
    return train, val, test, sample_sub


def preprocess_df(df: pd.DataFrame, is_test: bool = False) -> pd.DataFrame:
    df = df.copy()
    df["input"] = df["input"].map(clean_text)
    if not is_test:
        df["output"] = df["output"].map(clean_text)
        df = df[(df["input"] != "") & (df["output"] != "")].reset_index(drop=True)
    else:
        df = df[df["input"] != ""].reset_index(drop=True)
    df["language"] = df["subset"].map(get_language_name)
    df["country"] = df["subset"].map(get_country_name)
    return df


def build_prompt(question: str, language: str, style: str = "v1") -> str:
    """Build model input prompt for different experiment styles."""
    if style == "v1":
        return f"answer health question in {language}: {question}"
    elif style == "v2":
        return f"Generate a comprehensive health answer in {language} for the following question: {question}"
    elif style == "v3":
        return f"[{language}] Health Question Answering: {question}"
    elif style == "v4":
        return f"Health QA ({language}): {question}"
    elif style == "rag":
        # Placeholder - context will be injected by the RAG pipeline
        return f"answer health question in {language}: {question}"
    return question


def build_rag_prompt(question: str, language: str, context: str) -> str:
    return (
        f"Context: {context}\n\n"
        f"Answer the following health question in {language} using the context above:\n"
        f"{question}"
    )


def make_submission(ids: list[str], answers: list[str], output_path: str | Path | None = None) -> pd.DataFrame:
    clean = [re.sub(r"<extra_id_\d+>", "", str(a)).strip() for a in answers]
    sub = pd.DataFrame({
        "ID": ids,
        "TargetRLF1": clean,
        "TargetR1F1": clean,
        "TargetLLM": clean,
    })
    if output_path is not None:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        sub.to_csv(output_path, index=False, encoding="utf-8")
        print(f"Saved {len(sub)} rows → {output_path}")
    return sub


def get_answer_stats(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["answer_words"] = df["output"].str.split().str.len()
    df["question_words"] = df["input"].str.split().str.len()
    return df.groupby("subset").agg(
        n=("ID", "count"),
        mean_ans_words=("answer_words", "mean"),
        median_ans_words=("answer_words", "median"),
        mean_q_words=("question_words", "mean"),
    ).round(1)
