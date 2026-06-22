# Multilingual Health Question Answering in Low-Resource African Languages

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Chol1000/multilingual-health-qa/blob/main/notebooks/02_Training_Experiments.ipynb) [![Open In Colab (Drive)](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1w0fM1USsUUbM94Vs9Izp7CqPNWCUmVOG)

**Competition:** Zindi — Multilingual Health Question Answering in Low-Resource African Languages  
**Final Rank:** 184 / 1,585 participants (top 11.6%) | Private Score: 0.564471 | Public Score: 0.579345  
**Best Clean Score:** 0.5608 — BGE-M3 Dense Retrieval (Experiment 15)

---

## Setup

**1. Clone the repository**
```bash
git clone https://github.com/Chol1000/multilingual-health-qa.git
cd multilingual-health-qa
```

**2. Install dependencies**
```bash
pip install torch transformers>=4.40.0 datasets peft accelerate sentencepiece \
            rouge-score scikit-learn sentence-transformers evaluate FlagEmbedding
```

**3. Add competition data**  
Download `Train.csv`, `Val.csv`, `Test.csv`, and `SampleSubmission.csv` from the [Zindi competition page](https://zindi.africa) and place them in the `data/` folder.  
*(Data files are gitignored per Zindi's terms of service.)*

---

## Running on Google Colab

1. Click a badge above to open the notebook in Colab
2. Set runtime: **Runtime → Change runtime type → GPU (T4 or A100)**
3. Mount Google Drive or upload the competition data files to `data/`
4. Run the setup cell — all dependencies install automatically
5. Execute experiment sections in order

**Hardware requirements:**
- Retrieval experiments (Exp 1–2, 11–15): T4 GPU (15 GB VRAM)
- Fine-tuning experiments (Exp 3–9, 16–17): A100 GPU (40 GB VRAM)

---

## Repository Structure

```
.
├── notebooks/
│   ├── 01_EDA_Preprocessing.ipynb     # Exploratory data analysis and preprocessing
│   └── 02_Training_Experiments.ipynb  # All 17 experiments end-to-end
├── src/
│   ├── data_utils.py                  # Data loading, cleaning, prompt construction
│   ├── evaluation.py                  # ROUGE evaluation with whitespace tokenizer
│   ├── model_utils.py                 # Model loading, LoRA, training, inference
│   └── retrieval_utils.py             # TF-IDF, dense retrieval, reranker, hybrid
├── experiments/
│   └── experiment_log.md              # Detailed log of all 17 experiments
├── outputs/
│   ├── figures/                       # Visualizations (14 plots)
│   ├── logs/                          # Training histories and results CSVs
│   └── submissions/                   # All 14 Zindi submission files
├── data/                              # Competition data (gitignored)
└── requirements.txt
```

---

## Reproducibility

All experiments use `SEED = 42` and FP32 precision (`bf16=False`, `fp16=False`).  
Results are fully reproducible on Google Colab using the notebook above.

---

## Data and License

Competition data is subject to Zindi's terms and is not included in this repository.  
Source code is released under the MIT License.
