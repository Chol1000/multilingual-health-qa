# Experiment Log — Multilingual Health QA Challenge

All experiments are evaluated on the full validation set (6,686 rows) using whitespace-tokenized ROUGE.
Weighted score = ROUGE-1 x 0.37 + ROUGE-L x 0.37 + LLM-judge x 0.26.
Note: Experiments marked as INFLATED had validation rows inside the retrieval corpus, making local ROUGE artificially high. Zindi scores remain valid for those experiments.

---

## Experiment 1 — TF-IDF Global Retrieval

| Field | Value |
|-------|-------|
| Model | TF-IDF char 3-5 gram + Nearest Neighbors |
| Val ROUGE-1 | 0.4276 |
| Val ROUGE-L | 0.3740 |
| Zindi Score | 0.4945 (R1=0.4819, RL=0.4046, LLM=0.6402) |
| Inflated | No |

**What changed:** Baseline using character n-gram TF-IDF across a single global index of all 29,815 training rows.

**Why:** Fast lower bound returning verbatim training answers, which gives moderate ROUGE and LLM-judge since answers are factually correct and in the right language.

**Outcome:** R1=0.4276, RL=0.3740. Stronger than expected — char 3-5 grams work across all scripts (Latin, Ge'ez) without language-specific tokenization.

**Insight:** A single global pool of 29,815 examples provides dense coverage. High-frequency health vocabulary overlaps across languages and scripts at the character level.

---

## Experiment 2 — TF-IDF Per-Language Retrieval

| Field | Value |
|-------|-------|
| Model | TF-IDF char 3-5 gram + per-language index |
| Val ROUGE-1 | 0.4269 |
| Val ROUGE-L | 0.3734 |
| Zindi Score | 0.4937 (R1=0.4784, RL=0.4009, LLM=0.6475) |
| Inflated | No |

**Per-language breakdown:**

| Language | n | ROUGE-1 | ROUGE-L |
|----------|---|---------|---------|
| Swahili | 518 | 0.603 | 0.567 |
| Luganda | 846 | 0.516 | 0.493 |
| English | 3,746 | 0.460 | 0.410 |
| Akan | 1,114 | 0.283 | 0.167 |
| Amharic | 462 | 0.145 | 0.135 |

**What changed:** Separate TF-IDF index per language (5 indexes + global fallback).

**Why:** Isolating scripts should eliminate cross-language noise.

**Outcome:** Marginally lower than Exp 1 overall (R1 -0.0007). English dominates val set (56% of rows); the global index gives English queries access to the full 29,815-row pool vs a per-language sub-pool of ~13,062 rows. Amharic extremely weak (0.145) — Ge'ez script has unique character n-grams with almost no overlap with Latin scripts.

**Insight:** Per-language retrieval helps Latin-based languages but hurts overall performance due to English pool size reduction. Amharic requires a dedicated neural model regardless of retrieval strategy.

---

## Experiment 3 — mT5-base Vanilla Fine-Tuning (5k sample)

| Field | Value |
|-------|-------|
| Model | google/mt5-base (580M params) |
| Prompt | answer health question in {lang}: {question} |
| Epochs | 3 |
| LR | 5e-4 |
| Train data | 5,000 samples |
| Val ROUGE-1 | 0.1207 |
| Val ROUGE-L | 0.1029 |
| Zindi Score | Not submitted |
| Inflated | No |

**What changed:** First neural seq2seq model. Fine-tuned mT5-base on a 5k subsample to validate the training pipeline end-to-end on Colab T4.

**Why:** Establish minimum viable neural baseline before scaling to full data.

**Outcome:** R1=0.1207, RL=0.1029 — well below TF-IDF retrieval (0.4276). Five thousand samples is too small to learn diverse health domain answers across 5 languages. Model tends toward short, generic outputs.

**Insight:** Small fine-tuned models require far more training data to beat retrieval on a high-diversity generation task. Scaling to full data is essential.

---

## Experiment 4 — mT5-base + Instructional Prompt (5k sample)

| Field | Value |
|-------|-------|
| Model | google/mt5-base |
| Prompt | "You are a health expert. Provide a comprehensive answer in {lang}..." |
| Epochs | 3 |
| LR | 5e-4 |
| Train data | 5,000 samples |
| Val ROUGE-1 | 0.1231 |
| Val ROUGE-L | 0.1065 |
| Zindi Score | Not submitted |
| Inflated | No |

**What changed:** Replaced the short prompt with a longer expert-framing instructional prompt.

**Why:** More explicit task framing may guide the model toward longer, more complete answers and improve LLM-judge scores.

**Outcome:** R1=0.1231, RL=0.1065 — marginal improvement over Exp 3 (+0.0024 R1). The longer prompt reduces the effective answer token budget at max_input=256 tokens.

**Insight:** Prompt engineering has diminishing returns on small fine-tuned seq2seq models. The bottleneck is training data size, not prompt framing.

---

## Experiment 5 — NLLB-600M Full Data Fine-Tuning

| Field | Value |
|-------|-------|
| Model | facebook/nllb-200-distilled-600M |
| Language tokens | Per-row src_lang / tgt_lang / forced_bos_token_id |
| Epochs | 3 |
| LR | 5e-5 |
| Train data | 29,814 rows |
| Val ROUGE-1 | 0.2890 |
| Val ROUGE-L | 0.2216 |
| Zindi Score | 0.3471 (R1=0.3457, RL=0.2508, LLM=0.4863) |
| Inflated | No |

**What changed:** Switched to Facebook's NLLB-200 model trained on 200 languages with native support for Akan/Twi (twi_Latn), Luganda (lug_Latn), Amharic (amh_Ethi), and Swahili (swh_Latn). Used per-row language tokens.

**Why:** NLLB explicitly optimized for African languages; expected to outperform mT5 on Akan, Luganda, and Amharic.

**Outcome:** R1=0.2890, RL=0.2216 — lower than TF-IDF baseline. NLLB generates fluent text but QA fine-tuning on health domain answers requires more epochs to converge with the language-conditioned generation mechanism.

**Insight:** Specialized multilingual models need careful hyperparameter tuning for seq2seq QA versus translation. LLM-judge=0.4863 reflects reasonable answer quality despite lower ROUGE.

---

## Experiment 6 — mT5-base + TF-IDF RAG (Full Data)

| Field | Value |
|-------|-------|
| Model | google/mt5-base |
| RAG | TF-IDF per-language top-1 retrieval prepended as context |
| Max input | 384 tokens |
| Epochs | 3 |
| LR | 5e-5 |
| Train data | 29,814 rows |
| Val ROUGE-1 | 0.3836 |
| Val ROUGE-L | 0.3352 |
| Zindi Score | 0.4356 (R1=0.4327, RL=0.3608, LLM=0.5460) |
| Inflated | No |

**What changed:** Full training data with TF-IDF context prepended as RAG grounding. Format: Context: {retrieved_answer} — Answer health question in {lang}: {question}.

**Why:** RAG grounds generation in health-domain vocabulary from the training set, reducing hallucination and improving factual accuracy.

**Outcome:** R1=0.3836, RL=0.3352. Significantly better than Exp 3/4 (5k no-RAG) and competitive with TF-IDF alone. RAG grounding improves LLM-judge (0.5460 vs 0.4863 for NLLB alone in Exp 5).

**Insight:** RAG improves generation quality for mT5-base when full training data is used. The retrieved context acts as a domain anchor preventing off-topic generation.

---

## Experiment 7 — NLLB-600M + TF-IDF RAG (Top-3 Context)

| Field | Value |
|-------|-------|
| Model | facebook/nllb-200-distilled-600M |
| RAG | TF-IDF per-language top-3 retrieved answers as context |
| Epochs | 3 |
| LR | 5e-5 |
| Train data | 29,814 rows |
| Val ROUGE-1 | 0.3997 |
| Val ROUGE-L | 0.3447 |
| Zindi Score | 0.4610 (R1=0.4525, RL=0.3719, LLM=0.5999) |
| Inflated | No |

**What changed:** Combined NLLB-600M with TF-IDF top-3 RAG context.

**Why:** NLLB's African language representations and RAG grounding address different weaknesses and should be complementary.

**Outcome:** R1=0.3997, RL=0.3447. Best single-model result so far. LLM-judge=0.5999 — highest yet, reflecting NLLB's strong language-appropriate generation quality when grounded.

**Insight:** Model quality and retrieval grounding are complementary. NLLB handles generation fluency while RAG handles factual grounding, improving both ROUGE and LLM-judge together.

---

## Experiment 8 — NLLB-600M + TF-IDF RAG (Top-1 Context)

| Field | Value |
|-------|-------|
| Model | facebook/nllb-200-distilled-600M |
| RAG | TF-IDF per-language top-1 retrieved answer |
| Epochs | 3 |
| LR | 5e-5 |
| Train data | 29,814 rows |
| Val ROUGE-1 | 0.4014 |
| Val ROUGE-L | 0.3462 |
| Zindi Score | 0.4607 (R1=0.4534, RL=0.3725, LLM=0.5967) |
| Inflated | No |

**What changed:** Same as Exp 7 but top-1 retrieved context instead of top-3 to reduce noise.

**Why:** Three retrieved answers may introduce conflicting vocabulary and hurt generation coherence.

**Outcome:** R1=0.4014 (+0.0017 over Exp 7). Top-1 context performs slightly better — one focused context is cleaner than three concatenated answers.

**Insight:** For seq2seq generation, a single high-quality retrieved context outperforms multiple noisy ones. Context noise at inference hurts generation coherence.

---

## Experiment 9 — NLLB-1.3B + RAG Train+Val Combined (INFLATED)

| Field | Value |
|-------|-------|
| Model | facebook/nllb-200-distilled-1.3B |
| Train data | Train + Val combined (36,500 rows) |
| Val ROUGE-1 | 0.9331 (inflated) |
| Val ROUGE-L | 0.9298 (inflated) |
| Zindi Score | 0.4800 (R1=0.4752, RL=0.3979, LLM=0.6038) |
| Inflated | YES — val rows were in retrieval corpus |

**What changed:** Scaled to NLLB-1.3B and trained on Train+Val combined.

**Why:** More parameters and more training data should improve generalization.

**Outcome:** Local ROUGE artificially high (0.93) because validation questions were inside the retrieval corpus — the model retrieves its own validation answers. Zindi score 0.4800 is the true measure.

**Insight:** Always exclude the validation set from the retrieval corpus during local evaluation. Contamination creates misleading local metrics that do not reflect true generalization.

---

## Experiment 10 — Beam Search Inference Tuning (INFLATED)

| Field | Value |
|-------|-------|
| Model | NLLB-1.3B checkpoint from Exp 9 |
| Tested configs | beam=1/2/4/8, length_penalty=0.8/1.0/1.5 |
| Val ROUGE-1 | 0.8299 (inflated) |
| Val ROUGE-L | 0.8241 (inflated) |
| Zindi Score | Not submitted |
| Inflated | YES — same corpus contamination as Exp 9 |

**What changed:** Inference-only sweep over beam width and length penalty. No retraining.

**Why:** Beam width and length penalty are free inference-time hyperparameters that affect generation length and quality.

**Outcome:** beam=4 selected as optimal. Local ROUGE inflated due to corpus contamination inherited from Exp 9. beam=8 overfits to verbatim retrieval; beam=4 balances quality with diversity.

**Insight:** Always tune inference hyperparameters after training — they are a zero-cost performance boost. Evaluation must always use a clean held-out set.

---

## Experiment 11 — Dense Retrieval with E5-Large (Train Corpus Only)

| Field | Value |
|-------|-------|
| Model | intfloat/multilingual-e5-large |
| Retrieval | Dense cosine similarity, top-1 |
| Corpus | 29,815 training rows only |
| Val ROUGE-1 | 0.4526 |
| Val ROUGE-L | 0.4098 |
| Zindi Score | 0.5424 (R1=0.5080, RL=0.4430, LLM=0.7328) |
| Inflated | No |

**What changed:** Replaced TF-IDF with dense vector retrieval using multilingual-e5-large embeddings.

**Why:** Dense embeddings capture semantic similarity beyond surface character n-grams, which is especially important for paraphrased health questions.

**Outcome:** R1=0.4526, RL=0.4098 — best clean ROUGE so far. Zindi=0.5424. LLM-judge=0.7328 — a major improvement from TF-IDF (0.64), showing that dense retrieval returns more contextually appropriate answers.

**Insight:** Semantic similarity matching fundamentally outperforms lexical matching for health QA. Dense retrieval is the key breakthrough in this experiment sequence.

---

## Experiment 12 — Dense E5-Large (Train+Val Corpus) (INFLATED)

| Field | Value |
|-------|-------|
| Model | intfloat/multilingual-e5-large |
| Corpus | 36,501 rows (Train + Val) |
| Val ROUGE-1 | 0.9685 (inflated) |
| Val ROUGE-L | 0.9677 (inflated) |
| Zindi Score | 0.5552 (R1=0.5244, RL=0.4602, LLM=0.7344) |
| Inflated | YES — val rows were in retrieval corpus |

**What changed:** Expanded the retrieval corpus to include validation rows.

**Why:** More corpus rows should improve retrieval coverage.

**Outcome:** Local ROUGE near-perfect (0.97) due to contamination — each validation question retrieves itself. Zindi=0.5552 is the real score.

**Insight:** Confirmed that corpus contamination creates near-1.0 local ROUGE. The true benefit of a larger corpus is modest (+0.013 on Zindi versus Exp 11).

---

## Experiment 13 — Language-Specific Hybrid Retrieval (INFLATED)

| Field | Value |
|-------|-------|
| Model | E5-large (English + Swahili) + TF-IDF (Akan + Amharic + Luganda) |
| Corpus | 36,501 rows |
| Val ROUGE-1 | 0.7267 (inflated) |
| Val ROUGE-L | 0.7018 (inflated) |
| Zindi Score | 0.5793 (R1=0.5598, RL=0.4893, LLM=0.7353) |
| Inflated | YES — val rows in corpus for some subsets |

**What changed:** Applied different retrieval strategies per language: dense E5 for English and Swahili (large training data), TF-IDF for Akan, Amharic, and Luganda (smaller training data, specialized scripts).

**Why:** Dense retrieval requires sufficient training examples per language to build a useful embedding space. Low-resource languages may benefit more from character-level TF-IDF.

**Outcome:** Zindi=0.5793 — highest score across all experiments, but inflated. Language-specific routing does improve results for low-resource subsets.

**Insight:** One-size-fits-all retrieval is suboptimal. High-resource languages benefit from dense retrieval while low-resource languages may still be better served by character-level retrieval.

---

## Experiment 14 — Dense E5 + BGE Cross-Encoder Reranker (INFLATED)

| Field | Value |
|-------|-------|
| Model | E5-large retriever + BAAI/bge-reranker cross-encoder |
| Strategy | Retrieve top-20, rerank to top-1 |
| Corpus | 36,501 rows |
| Val ROUGE-1 | 0.6542 (inflated) |
| Val ROUGE-L | 0.6267 (inflated) |
| Zindi Score | 0.5245 (R1=0.4948, RL=0.4107, LLM=0.7289) |
| Inflated | YES — val rows in corpus |

**What changed:** Added a cross-encoder reranker on top of dense retrieval: retrieve top-20 candidates, rerank by cross-attention relevance score, then select top-1.

**Why:** Bi-encoder retrieval (E5) is fast but uses independent query/document encodings. Cross-encoder reranking jointly encodes query and document for more precise relevance scoring.

**Outcome:** Zindi=0.5245 — lower than E5 alone (0.5424 in Exp 11). The reranker adds inference overhead without improving final scores.

**Insight:** Cross-encoder reranking does not necessarily improve over a strong bi-encoder for this task. BGE-M3's unified dense model (Exp 15) proved to be a better trade-off.

---

## Experiment 15 — BGE-M3 Dense Retrieval (Best Clean Score)

| Field | Value |
|-------|-------|
| Model | BAAI/bge-m3 |
| Corpus | 29,815 training rows only (val excluded) |
| Val ROUGE-1 | 0.4761 |
| Val ROUGE-L | 0.4278 |
| Zindi Score | 0.5608 (R1=0.5345, RL=0.4627, LLM=0.7379) |
| Inflated | No |

**What changed:** Replaced E5-large with BGE-M3 — a unified retrieval model supporting dense, sparse, and multi-vector retrieval, trained specifically for cross-lingual retrieval tasks.

**Why:** BGE-M3 is trained on more languages and with retrieval-specific objectives compared to E5-large, which is primarily a text embedding model. It is better suited for multilingual health QA.

**Outcome:** R1=0.4761, RL=0.4278, Zindi=0.5608 — best clean score across all experiments. LLM-judge=0.7379 is the highest of any clean experiment. BGE-M3 outperforms E5-large by +0.0184 on Zindi.

**Insight:** Model architecture matters even for retrieval-only approaches. BGE-M3's multilingual retrieval pre-training produces superior cross-lingual alignment compared to general-purpose text embeddings.

---

## Experiment 16 — NLLB-600M + Hybrid RAG with LoRA

| Field | Value |
|-------|-------|
| Model | facebook/nllb-200-distilled-600M + LoRA r=16 |
| RAG | E5-large (English + Swahili) + TF-IDF (Akan + Amharic + Luganda) |
| Epochs | 1 |
| Train data | 29,814 rows |
| Val ROUGE-1 | 0.4594 |
| Val ROUGE-L | 0.4085 |
| Zindi Score | 0.5279 (R1=0.5210, RL=0.4448, LLM=0.6558) |
| Inflated | No |

**What changed:** Combined NLLB-600M with LoRA PEFT and the language-specific hybrid RAG from Exp 13, using only the training corpus.

**Why:** LoRA reduces trainable parameters to approximately 1% of the model, allowing efficient fine-tuning. Hybrid RAG provides language-appropriate retrieval strategy.

**Outcome:** Zindi=0.5279 — competitive but below BGE-M3 pure retrieval (0.5608). The generation step introduces variance that outweighs the grounding benefit at just 1 epoch.

**Insight:** Fine-tuning with RAG requires more epochs to converge than pure retrieval. The generation overhead does not recover the performance gap versus dense retrieval alone at this training budget.

---

## Experiment 17 — NLLB-1.3B + LoRA + Hybrid RAG

| Field | Value |
|-------|-------|
| Model | facebook/nllb-200-distilled-1.3B + LoRA r=16 |
| RAG | E5-large (English + Swahili) + TF-IDF (Akan + Amharic + Luganda) |
| Trainable params | ~4.7M / 1.37B (0.34%) |
| Epochs | 3 |
| Train data | 29,814 rows |
| Val ROUGE-1 | 0.4491 |
| Val ROUGE-L | 0.3979 |
| Zindi Score | 0.5178 (R1=0.5112, RL=0.4338, LLM=0.6469) |
| Inflated | No |

**What changed:** Scaled to NLLB-1.3B with LoRA r=16 and full 3-epoch training. Hybrid RAG from Exp 13 on clean train-only corpus.

**Why:** Larger model capacity should improve generation quality over 600M. LoRA makes 1.3B feasible within T4 GPU memory constraints.

**Outcome:** Zindi=0.5178 — lower than the 600M variant (Exp 16, 0.5279) and well below BGE-M3 (0.5608). Larger model with limited training budget may underfit.

**Insight:** Scaling model size alone does not guarantee improvement when training budget is fixed. BGE-M3 dense retrieval remains the strongest approach for this task within T4 GPU constraints.

---

## Summary Table

| Exp | Name | ROUGE-1 | ROUGE-L | Zindi | Notes |
|-----|------|---------|---------|-------|-------|
| 1 | TF-IDF Global | 0.4276 | 0.3740 | 0.4945 | Baseline |
| 2 | TF-IDF Per-Language | 0.4269 | 0.3734 | 0.4937 | |
| 3 | mT5-base Vanilla 5k | 0.1207 | 0.1029 | — | Not submitted |
| 4 | mT5-base Prompt-v2 5k | 0.1231 | 0.1065 | — | Not submitted |
| 5 | NLLB-600M Full Data | 0.2890 | 0.2216 | 0.3471 | |
| 6 | mT5-base + TF-IDF RAG | 0.3836 | 0.3352 | 0.4356 | |
| 7 | NLLB-600M + RAG top-3 | 0.3997 | 0.3447 | 0.4610 | |
| 8 | NLLB-600M + RAG top-1 | 0.4014 | 0.3462 | 0.4607 | |
| 9 | NLLB-1.3B + RAG Train+Val | 0.9331 | 0.9298 | 0.4800 | INFLATED local |
| 10 | Beam Search Tuning | 0.8299 | 0.8241 | — | INFLATED, not submitted |
| 11 | Dense E5-large (train only) | 0.4526 | 0.4098 | 0.5424 | |
| 12 | Dense E5-large (train+val) | 0.9685 | 0.9677 | 0.5552 | INFLATED local |
| 13 | Language-Specific Hybrid | 0.7267 | 0.7018 | 0.5793 | INFLATED local |
| 14 | Dense E5 + BGE Reranker | 0.6542 | 0.6267 | 0.5245 | INFLATED local |
| 15 | BGE-M3 Dense (clean) | 0.4761 | 0.4278 | 0.5608 | Best clean score |
| 16 | NLLB-600M + Hybrid RAG LoRA | 0.4594 | 0.4085 | 0.5279 | |
| 17 | NLLB-1.3B + LoRA + Hybrid RAG | 0.4491 | 0.3979 | 0.5178 | |

Best clean Zindi score: Exp 15 — BGE-M3 Dense Retrieval — 0.5608
Final submission: Exp 15 (primary), Exp 11 (backup)
