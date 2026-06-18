"""African Language Health QA — source package.

Public API
----------
Data utilities:
    load_data, preprocess_df, clean_text,
    build_prompt, build_rag_prompt, make_submission, get_answer_stats

Evaluation:
    compute_rouge, compute_rouge_by_language,
    compute_overall_weighted_score, rouge_single

Model utilities (requires torch + transformers):
    get_device, load_model_and_tokenizer, apply_lora,
    tokenize_dataset, build_training_args,
    make_compute_metrics, generate_predictions

Retrieval utilities (requires scikit-learn; DenseRetriever/Reranker require sentence-transformers):
    TFIDFRetriever, TFIDFRetrieverK3,
    DenseRetriever, Reranker, LangSpecificRetriever,
    add_rag_context, retrieve_topk, get_context, get_context_k3
"""

from .data_utils import (
    load_data,
    preprocess_df,
    clean_text,
    build_prompt,
    build_rag_prompt,
    make_submission,
    get_answer_stats,
    get_language_name,
    get_country_name,
)

from .evaluation import (
    compute_rouge,
    compute_rouge_by_language,
    compute_overall_weighted_score,
    rouge_single,
    WhitespaceTokenizer,
    ROUGE1_WEIGHT,
    ROUGEL_WEIGHT,
    LLM_WEIGHT,
)

from .retrieval_utils import (
    TFIDFRetriever,
    TFIDFRetrieverK3,
    add_rag_context,
    get_context,
    get_context_k3,
)

# Heavy deps (torch + transformers + sentence-transformers) — available in Colab/GPU env
try:
    from .model_utils import (
        get_device,
        load_model_and_tokenizer,
        apply_lora,
        tokenize_dataset,
        build_training_args,
        make_compute_metrics,
        generate_predictions,
    )
except ImportError:
    pass

try:
    from .retrieval_utils import (
        DenseRetriever,
        Reranker,
        LangSpecificRetriever,
        retrieve_topk,
    )
except ImportError:
    pass

__all__ = [
    # data
    "load_data", "preprocess_df", "clean_text",
    "build_prompt", "build_rag_prompt", "make_submission", "get_answer_stats",
    "get_language_name", "get_country_name",
    # evaluation
    "compute_rouge", "compute_rouge_by_language",
    "compute_overall_weighted_score", "rouge_single",
    "WhitespaceTokenizer", "ROUGE1_WEIGHT", "ROUGEL_WEIGHT", "LLM_WEIGHT",
    # model (torch + transformers)
    "get_device", "load_model_and_tokenizer", "apply_lora",
    "tokenize_dataset", "build_training_args",
    "make_compute_metrics", "generate_predictions",
    # retrieval
    "TFIDFRetriever", "TFIDFRetrieverK3",
    "DenseRetriever", "Reranker", "LangSpecificRetriever",
    "add_rag_context", "retrieve_topk", "get_context", "get_context_k3",
]
