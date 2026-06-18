# Multilingual Health QA — African Languages Challenge

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Chol1000/multilingual-health-qa/blob/main/notebooks/02_Training_Experiments.ipynb)

**Zindi Competition:** Multilingual Health Question Answering in Low-Resource African Languages  
**Challenge:** Build a multilingual model to answer health questions in Akan, Amharic, Luganda, Swahili, and English  
**Evaluation:** ROUGE-1 F1 × 0.37 + ROUGE-L F1 × 0.37 + LLM-as-a-Judge × 0.26

---

## Project Overview

This repository contains the complete training, evaluation, and submission pipeline for the
Zindi *Multilingual Health Question Answering in Low-Resource African Languages* competition.
The task is to generate accurate, fluent health answers to questions in five languages across
nine language-country configurations.

### Dataset Statistics

| Subset | Language | Country | Train | Val | Test |
|--------|----------|---------|-------|-----|------|
| Aka_Gha | Akan | Ghana | 4,455 | 1,114 | 492 |
| Amh_Eth | Amharic | Ethiopia | 1,845 | 462 | 61 |
| Eng_Eth | English | Ethiopia | 3,915 | 564 | 60 |
| Eng_Gha | English | Ghana | 4,443 | 1,104 | 491 |
| Eng_Ken | English | Kenya | 2,080 | 390 | 167 |
| Eng_Uga | English | Uganda | 7,624 | 1,688 | 744 |
| Lug_Uga | Luganda | Uganda | 3,383 | 846 | 374 |
| Swa_Ken | Swahili | Kenya | 2,070 | 518 | 229 |
| **Total** | | | **29,815** | **6,686** | **2,618** |

---

## Repository Structure

```
.
├── README.md
├── requirements.txt
├── notebooks/
│   ├── 01_EDA_Preprocessing.ipynb    # Exploratory data analysis
│   └── 02_Training_Experiments.ipynb # All 17 training experiments (main Colab notebook)
├── src/
│   ├── __init__.py
│   ├── data_utils.py    # Data loading, cleaning, prompt building
│   ├── evaluation.py    # ROUGE evaluation utilities
│   └── model_utils.py   # Model creation, training, inference
├── experiments/
│   ├── experiment_log.md            # Detailed experiment documentation
│   └── experiment_results.csv       # Auto-generated results table
├── outputs/
│   ├── figures/         # All plots (learning curves, per-language ROUGE, actual vs predicted)
│   ├── logs/            # CSVs and JSONs (results, training histories, per-sample analysis)
│   ├── submissions/     # Zindi submission CSVs (FINAL_primary.csv, FINAL_backup.csv tracked)
│   └── checkpoints/     # Model checkpoints (gitignored — too large for git)
└── data/                            # Data files (download from Zindi)
```

---

## Quick Start on Google Colab

### Option A — One-Click Colab

Click the badge at the top to open `02_Training_Experiments.ipynb` directly in Colab.
The notebook installs all dependencies automatically.

### Option B — Manual Setup

```bash
# 1. Clone the repository
git clone https://github.com/Chol1000/multilingual-health-qa.git
cd multilingual-health-qa

# 2. Install dependencies
pip install -r requirements.txt

# 3. Place competition data files in the data/ folder:
#    Train.csv, Val.csv, Test.csv, SampleSubmission.csv
#    (download from the Zindi competition page)

# 4. Run EDA
jupyter notebook notebooks/01_EDA_Preprocessing.ipynb

# 5. Run experiments
jupyter notebook notebooks/02_Training_Experiments.ipynb
```

### Colab Setup Notes

- **Hardware:** Use Colab T4 GPU (Runtime → Change runtime type → GPU)
- **VRAM:** T4 provides 15 GB — sufficient for mT5-base with batch_size=8
- **Training time:** ~2–3 hours per mT5-base experiment on T4
- **Data:** Upload CSV files via the Colab file browser or mount Google Drive

---

## Experiment Methodology

We ran 17 systematic experiments progressively improving the leaderboard score:

| # | Experiment | Model | Key Change |
|---|-----------|-------|------------|
| 1 | TF-IDF Global | — | Character 3–5 gram global index |
| 2 | TF-IDF Per-Language | — | Separate per-language index |
| 3 | mT5-small Vanilla | mt5-small | Seq2seq baseline, 300M params |
| 4 | mT5-base Vanilla | mt5-base | Scale to 580M params, same 5k sample |
| 5 | mT5-base Prompt-v2 | mt5-base | Instructional expert-framing prompt |
| 6 | NLLB-600M Fine-tune | nllb-200-distilled-600M | Best African language coverage |
| 7 | mT5-base + LoRA r=16 | mt5-base | PEFT, ~1% trainable params |
| 8 | mT5-base + RAG | mt5-base | TF-IDF retrieved context |
| 9 | NLLB-600M + RAG (inflated) | nllb-200-distilled-600M | Best model + retrieval |
| 10 | Best Arch Train+Val (inflated) | best from above | Full data for final model |
| 11 | Beam Search Tuning | best checkpoint | No retraining, inference only |
| 12 | Ensemble Top-2 (inflated) | top-2 models | Length-aware answer selection |
| 13 | BGE-M3 Dense RAG (inflated) | bge-m3 | Dense retrieval, val in corpus |
| 14 | BGE-M3 + mT5 Rerank (inflated) | bge-m3 + mt5-base | Cross-encoder reranking |
| 15 | BGE-M3 Dense Clean | bge-m3 | Dense retrieval, val excluded |
| 16 | Hybrid BM25 + BGE-M3 | bge-m3 + BM25 | Sparse + dense fusion |
| 17 | BGE-M3 + LoRA mT5 | bge-m3 + mt5-base LoRA | Dense retrieval + LoRA reader |

See [`experiments/experiment_log.md`](experiments/experiment_log.md) for full analysis.

---

## Model Architecture

**Primary models:**

`google/mt5-base` (580M parameters, Experiments 3–5, 7–10)
- Multilingual T5 pre-trained on mC4 corpus covering 101 languages
- Encoder-Decoder architecture ideal for seq2seq health QA

`facebook/nllb-200-distilled-600M` (600M parameters, Experiments 6, 9)
- No Language Left Behind — explicitly trained on 200 languages
- Native support for Akan/Twi (twi_Latn), Luganda (lug_Latn), Amharic (amh_Ethi), Swahili (swh_Latn)
- Uses per-row language tokens (src_lang, tgt_lang, forced_bos_token_id) for language-conditioned generation

**Fine-tuning approach:**
- Task: sequence-to-sequence generation
- Input: `"answer health question in {language}: {question}"`
- Target: reference health answer
- Optimizer: AdamW with cosine LR schedule and warmup_ratio=0.1
- Label smoothing: 0.0 (disabled — large vocab log_softmax causes NaN with smoothing > 0)
- Mixed precision: FP32 only (bf16=False, fp16=False — required for reproducibility)

**PEFT (Experiment 7):**
- LoRA applied to Q and V attention matrices, r=16, alpha=32
- Reduces trainable parameters to ~1% of full model

---

## Evaluation

Metrics aligned with Zindi competition:
- **ROUGE-1 F1** (0.37 weight): Unigram overlap
- **ROUGE-L F1** (0.37 weight): Longest Common Subsequence
- **LLM-as-a-Judge** (0.26 weight): Factual accuracy, completeness, language appropriateness

All ROUGE scores use whitespace tokenization (language-agnostic, safe for Ge'ez/Akan scripts).

---

## Reproducibility

All experiments use `SEED = 42` throughout. To reproduce a specific experiment:

1. Open `notebooks/02_Training_Experiments.ipynb` in Colab with T4 GPU
2. Run all cells up to the desired experiment section
3. Results are automatically saved to `outputs/submissions/` and `outputs/logs/`

Random state is set at the top of the notebook for numpy, Python random, and PyTorch.

---

## Ethical Considerations

This project addresses health question answering for underserved African language communities.
Key ethical considerations:

1. **Misinformation risk:** Generated answers may contain factual errors. Production deployment
   requires clinical review, especially for maternal/sexual/reproductive health topics.

2. **Language representation bias:** mT5 has uneven coverage of African languages —
   Amharic and Akan have less pre-training data than English/Swahili, potentially producing
   lower-quality answers for those communities.

3. **Cultural sensitivity:** Health norms vary across cultures. Answers appropriate in one
   country may be inappropriate or offensive in another context.

4. **AI use disclosure:** This project used Claude (Anthropic) as a coding assistant
   for structuring the pipeline. All experimental decisions, analysis, and interpretation
   are the author's own.

---

## License

Data: CC-BY SA 4.0 (Zindi competition data)  
Code: MIT License
