# Self-Consistency Evaluation Scripts

These scripts run baseline vs self-consistency comparison experiments for causal LLMs:

- `eval_model.sh <model-name>`: run one supported LLM through both the normal path and
  the self-consistency path, then write a comparison JSON.
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
- `SELF_CONSISTENCY_INCLUDE_ORIGINAL=1` to blend the sampled style vector with the
  direct document embedding

Examples:

```bash
eval/scripts/self-consistency/eval_model.sh qwen2.5-3b-instruct
```

```bash
eval/scripts/self-consistency/eval_all_llms.sh qwen2.5-3b-instruct llama3-8b-instruct
```
