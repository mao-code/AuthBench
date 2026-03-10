"""Central registry for popular authorship embedding models and their Hugging Face IDs.

The curated list focuses on high-performing general-purpose embedding checkpoints that
see widespread use for authorship attribution and stylistic analysis pipelines, based on
their Hugging Face model cards and leaderboard presence as of 2025.
"""

from typing import Dict, List

# Mapping from human-friendly names to Hugging Face repository identifiers.
MODEL_HF_PATHS: Dict[str, str] = {
    "e5-large-v2": "intfloat/e5-large-v2",
    "e5-base-v2": "intfloat/e5-base-v2",
    "e5-small-v2": "intfloat/e5-small-v2",
    "e5-mistral-7b-instruct": "intfloat/e5-mistral-7b-instruct",
    "instructor-xl": "hkunlp/instructor-xl",
    "instructor-large": "hkunlp/instructor-large",
    "instructor-base": "hkunlp/instructor-base",
    "multilingual-e5-large": "intfloat/multilingual-e5-large",
    "multilingual-e5-base": "intfloat/multilingual-e5-base",
    "bge-large-en-v1.5": "BAAI/bge-large-en-v1.5",
    "bge-base-en-v1.5": "BAAI/bge-base-en-v1.5",
    "bge-small-en-v1.5": "BAAI/bge-small-en-v1.5",
    "bge-large-zh-v1.5": "BAAI/bge-large-zh-v1.5",
    "bge-base-zh-v1.5": "BAAI/bge-base-zh-v1.5",
    "bge-small-zh-v1.5": "BAAI/bge-small-zh-v1.5",
    "bge-m3": "BAAI/bge-m3",
    "snowflake-arctic-embed-l-v2": "Snowflake/snowflake-arctic-embed-l-v2.0",
    "snowflake-arctic-embed-m-v2": "Snowflake/snowflake-arctic-embed-m-v2.0",
    "jina-embeddings-v2-base-en": "jinaai/jina-embeddings-v2-base-en",
    "jina-embeddings-v2-small-en": "jinaai/jina-embeddings-v2-small-en",
    "mxbai-embed-large-v1": "mixedbread-ai/mxbai-embed-large-v1",
    "gte-large-en-v1.5": "Alibaba-NLP/gte-large-en-v1.5",
    "gte-qwen2-7b-instruct": "Alibaba-NLP/gte-Qwen2-7B-instruct",
    "gte-base": "thenlper/gte-base",
    "gte-large": "thenlper/gte-large",
    "nv-embed-v1": "nvidia/NV-Embed-v1",
    "qwen3-embedding-0.6b": "Qwen/Qwen3-Embedding-0.6B",
    "qwen3-embedding-4b": "Qwen/Qwen3-Embedding-4B",
    "qwen3-embedding-8b": "Qwen/Qwen3-Embedding-8B",
    "qwen3-4b": "Qwen/Qwen3-4B",
    "qwen3-4b-instruct": "Qwen/Qwen3-4B-Instruct-2507",
    "qwen2.5-3b": "Qwen/Qwen2.5-3B",
    "qwen2.5-3b-instruct": "Qwen/Qwen2.5-3B-Instruct",
    "qwen2.5-7b-instruct": "Qwen/Qwen2.5-7B-Instruct",
    "nomic-embed-text-v1": "nomic-ai/nomic-embed-text-v1",
    "nomic-embed-text-v1.5": "nomic-ai/nomic-embed-text-v1.5",
    "sfr-embedding-mistral": "Salesforce/SFR-Embedding-Mistral",
    "all-minilm-l12-v2": "sentence-transformers/all-MiniLM-L12-v2",
    "all-minilm-l6-v2": "sentence-transformers/all-MiniLM-L6-v2",
    "all-mpnet-base-v2": "sentence-transformers/all-mpnet-base-v2",
    "distiluse-base-multilingual-cased-v2": "sentence-transformers/distiluse-base-multilingual-cased-v2",
    "msmarco-distilbert-base-v4": "sentence-transformers/msmarco-distilbert-base-v4",
    "all-roberta-large-v1": "sentence-transformers/all-roberta-large-v1",
    "multi-qa-mpnet-base-dot-v1": "sentence-transformers/multi-qa-mpnet-base-dot-v1",
    "paraphrase-mpnet-base-v2": "sentence-transformers/paraphrase-mpnet-base-v2",
    "paraphrase-multilingual-mpnet-base-v2": "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
    "allenai-specter": "allenai/specter",
    "bert-base-uncased": "bert-base-uncased",
    "facebook-contriever": "facebook/contriever",
    "facebook-contriever-msmarco": "facebook/contriever-msmarco",
    "llama3.1-8b": "meta-llama/Llama-3.1-8B",
    "llama3.1-8b-instruct": "meta-llama/Llama-3.1-8B-Instruct",
    "llama3-8b": "meta-llama/Meta-Llama-3-8B",
    "llama3-8b-instruct": "meta-llama/Meta-Llama-3-8B-Instruct",
    "llama2-7b": "meta-llama/Llama-2-7b-hf",
    "llama2-7b-chat": "meta-llama/Llama-2-7b-chat-hf",
    "deepseek-llm-7b-base": "deepseek-ai/deepseek-llm-7b-base",
    "deepseek-llm-7b-chat": "deepseek-ai/deepseek-llm-7b-chat",
    "deepseek-coder-6.7b-instruct": "deepseek-ai/deepseek-coder-6.7b-instruct",
}


# Causal LLM checkpoints that can run the generation-based self-consistency pipeline.
SELF_CONSISTENCY_LLM_MODELS = {
    "qwen3-4b",
    "qwen3-4b-instruct",
    "qwen2.5-3b",
    "qwen2.5-3b-instruct",
    "qwen2.5-7b-instruct",
    "llama3.1-8b",
    "llama3.1-8b-instruct",
    "llama3-8b",
    "llama3-8b-instruct",
    "llama2-7b",
    "llama2-7b-chat",
    "deepseek-llm-7b-base",
    "deepseek-llm-7b-chat",
    "deepseek-coder-6.7b-instruct",
}


def get_hf_repo(model_name: str) -> str:
    """Return the Hugging Face path for the requested model."""
    try:
        return MODEL_HF_PATHS[model_name]
    except KeyError as exc:
        raise KeyError(
            f"Unknown model '{model_name}'. Available options: {', '.join(sorted(MODEL_HF_PATHS))}"
        ) from exc


def supports_self_consistency(model_name: str) -> bool:
    """Return whether the registry model supports generation-based self-consistency."""

    return model_name in SELF_CONSISTENCY_LLM_MODELS


def self_consistency_model_names() -> List[str]:
    """Return the registry models that support generation-based self-consistency."""

    return sorted(SELF_CONSISTENCY_LLM_MODELS)
