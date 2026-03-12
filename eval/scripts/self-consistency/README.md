# Self-Consistency Evaluation Scripts

These scripts run baseline vs self-consistency comparison experiments for causal LLMs:

- `eval_model.sh <model-name>`: run one supported LLM through both the normal path and
  the score-aggregated self-consistency path, then write a comparison JSON.
- `eval_all_llms.sh [model-a model-b ...]`: run a list of models, or fall back to the
  registry models listed in `utilities/model_registry.py::SELF_CONSISTENCY_LLM_MODELS`.
- `compare_results.py`: compare one baseline JSON and one self-consistency JSON.
- `aggregate_comparisons.py`: combine per-model comparison JSONs into one JSON/CSV table.

Common environment variables:

- `DATASET_ROOT`, `SPLIT`, `TASK`, `OUTPUT_DIR`
- `BATCH_SIZE`, `MAX_LENGTH`, `NO_TRUNCATION`
- `SELF_CONSISTENCY_SAMPLES`, `SELF_CONSISTENCY_TOP_K`
- `SELF_CONSISTENCY_TEMPERATURE`, `SELF_CONSISTENCY_MAX_NEW_TOKENS`
- `RUN_BASELINE=0` or `RUN_SELF_CONSISTENCY=0` if you want to disable one path
- `SELF_CONSISTENCY_INCLUDE_ORIGINAL=1` to add the direct document embedding as one
  extra sampled embedding in the final score aggregation

Overall flow:

1. For each query document, sample `N` style descriptions with top-k decoding.
2. Pool the hidden states of each sampled description into `N` fixed-size query embeddings.
3. Repeat the same process for every candidate document, yielding `N` candidate embeddings per candidate.
4. For sample `i`, compute similarities between query embedding `i` and all candidate embeddings `i`.
5. Sum the per-sample similarities across all `N` samples for each query-candidate pair.
6. Rerank the full candidate pool using the summed scores.
7. Compute `S@K`, `R@K`, `nDCG@K`, and `EER` from those reranked summed-score outputs.

Notes:

- The active evaluation path does not average the sampled embeddings.
- The active evaluation path also does not majority-vote truncated top-K lists.
- `SELF_CONSISTENCY_INCLUDE_ORIGINAL=1` appends the original document embedding as one additional term in the summed score.

Examples:

```bash
eval/scripts/self-consistency/eval_model.sh qwen2.5-3b-instruct
```

```bash
eval/scripts/self-consistency/eval_all_llms.sh qwen2.5-3b-instruct llama3-8b-instruct
```
