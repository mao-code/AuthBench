# AuthBench Evaluation & Training

This folder hosts standalone scripts to score and fine-tune embedding models on the
processed AuthBench benchmark produced by `AuthBench/processing`. It assumes the
benchmark layout of `train|dev|test/{candidates,queries,ground_truth}.jsonl` created
by `build_benchmark.py` + `postprocess.py`.

## What is evaluated?

- **Authorship Representation (retrieval)**  
  Queries and candidate documents are embedded, similarities are ranked, and metrics are
  derived from those ranks. Uses cosine similarity by default or max-sim when
  `--late-interaction` is enabled. Metrics: `Recall@K`, `Success@K`, `nDCG@K`, `MRR`.

- **Authorship Attribution (verification)**  
  Full similarity scores between every query/candidate pair are contrasted against the
  binary ground-truth match matrix to compute **EER** (threshold-free operating point
  where false-accept = false-reject). Negatives are sampled per query by default for
  tractable memory, or `--negative-strategy all` uses every non-positive pair.

## Key modules

- `data.py` – Loads processed splits and exposes helper datasets for training.
- `embedder.py` – Minimal HF embedding wrapper with configurable pooling and token-level
  outputs for late interaction.
- `metrics.py` – Ranking metrics and EER computation.
- `evaluators.py` – Task-specific evaluation routines that work with any HF embedding.
- `runner.py` – CLI to score one or many models from `utilities/model_registry.py`.
- `train.py` – Contrastive fine-tuning script with in-batch negatives and periodic eval.

## Paths

The CLI defaults to `AuthBench/processing/outputs/official_ttl300k_cap10M_sf10k_postprocessed`
(relative to the repo root). Override with `--dataset-root` if your processed benchmark
lives elsewhere.

## Running evaluations

Recommended module invocation (run from the repo parent, or set `PYTHONPATH` to the repo parent):

```bash
python -m AuthBench.eval.runner --help
```

Evaluate a single model on the test split:

```bash
python -m AuthBench.eval.runner \
  --split test \
  --models e5-large-v2 \
  --batch-size 32 \
  --dataset-root /path/to/outputs/official_ttl300k_cap10M_sf10k_postprocessed
```

Evaluate every registry model on dev with fewer candidates/queries for a quick sweep:

```bash
python -m AuthBench.eval.runner \
  --split dev \
  --all-models \
  --max-candidates 20000 --max-queries 2000 \
  --output-json eval_dev.json
```

Enable late interaction (token-level max-sim); only feasible on smaller subsets because
it materializes token embeddings:

```bash
python -m AuthBench.eval.runner \
  --models bge-m3 \
  --late-interaction \
  --max-candidates 4000 --max-queries 1000
```

Topic-matched candidate pools (topic-leakage control) can be enabled with
`--candidate-pool topic`. Use `--max-topic-candidates` to cap pool sizes with
deterministic sampling:

```bash
python -m AuthBench.eval.runner \
  --split test \
  --models e5-large-v2 \
  --candidate-pool topic \
  --max-topic-candidates 5000
```

## LLM Embedding Flow

For causal LLM checkpoints used through `embedder.py`, the normal embedding path does
not autoregressively generate output tokens. The flow is:

1. Tokenize the input text.
2. Run one forward pass through the model.
3. Read the final `last_hidden_state` tensor with shape `(batch_size, seq_len, hidden_dim)`.
4. Pool across the token dimension to get one vector per document.
5. Normalize the pooled vector before scoring.

With the default `--pooling mean`, the document embedding is the mean over the
non-padding token hidden states. Other supported pooling choices are `cls` and `last`.

TF-IDF baseline (cosine on TF-IDF vectors) is available via the dedicated runner:

```bash
python -m AuthBench.eval.tfidf_runner \
  --dataset-root /path/to/outputs/official_ttl300k_cap10M_sf10k_postprocessed \
  --split test \
  --output-json eval/results/tfidf.json
```

Additional non-transformer baselines are available via the shared baseline runner:

- `tfidf` – existing character `3-5` gram TF-IDF cosine baseline.
- `ngram` – hashed character/word n-gram stylometric features with a train-split logistic pair calibrator.
- `ppm` – fixed-order hashed character language-model approximation of PPM-style cross-entropy scoring with a train-split logistic calibrator.

Run one or more baselines directly:

```bash
python -m AuthBench.eval.baseline_runner \
  --dataset-root processing/outputs/combined_phase1_phase2 \
  --split test \
  --baselines tfidf ngram ppm \
  --output-json eval/results/baselines/all_baselines.json
```

Or use the shell wrapper that writes one JSON per baseline and defaults to the combined
phase1+phase2 benchmark:

```bash
eval/scripts/eval_all_baselines.sh
```

Useful overrides for the shell wrapper:

- `BASELINES="tfidf ppm"` to evaluate only a subset.
- `OUTPUT_DIR=eval/results/baselines_topic CANDIDATE_POOL=topic MAX_TOPIC_CANDIDATES=5000` for topic-matched evaluation.

Analyze zero-shot result JSONs, merge in the three baselines from
`eval/results/baselines`, export organized tables, and generate bar-chart figures:

```bash
eval/scripts/analyze_results.sh
```

The wrapper defaults to:

- `RESULTS_DIR=eval/results`
- `BASELINES_DIR=eval/results/baselines`
- `DATASET_ROOT=processing/outputs/authbench`
- `SPLIT=test`
- `CANDIDATE_POOL=all`
- `OUTPUT_DIR=eval/results/analysis`

Useful overrides:

- `METRICS="success@10 recall@10 ndcg@10 mrr roc_auc eer"` to restrict exported metrics.
- `CANDIDATE_POOL=topic MAX_TOPIC_CANDIDATES=5000` to analyze topic-controlled runs with the matching random-reference normalization.
- `MAX_QUERIES=2000 MAX_CANDIDATES=5000 SEED=13` to reproduce analysis for capped evaluation runs.
- `SKIP_PLOTS=1` to export tables and reports only.

The analysis output is organized as:

- `eval/results/analysis/metadata/`
  - `analysis_config.json`
  - `random_reference/*.csv` with random-guess expectations and pool-size stats by split/language/genre/length bucket
- `eval/results/analysis/tables/summary/`
  - `leaderboard_overall.csv`
  - `best_by_metric.csv`
  - `by_model_type.csv`
  - `best_model_by_slice.csv`
  - `slice_difficulty.csv`
- `eval/results/analysis/tables/long/`
  - `overall_metrics_long.csv`
  - `grouped_metrics_long.csv`
- `eval/results/analysis/tables/wide/`
  - `language/{raw,normalized}/*.csv`
  - `primary_genre/{raw,normalized}/*.csv`
  - `length_bucket/{raw,normalized}/*.csv`
- `eval/results/analysis/plots/`
  - horizontal bar charts for overall metrics and macro-by-slice metrics, with dashed reference lines for `tfidf`, `ngram`, and `ppm`
- `eval/results/analysis/reports/fine_grained_analysis.md`

The exported tables include every metric present in the JSON results, including:

- Retrieval: `success@K`, `recall@K`, `ndcg@K`, `mrr`
- Attribution: `roc_auc`, `eer`

They also include candidate-pool statistics:

- `num_candidates` when the pool size is fixed
- `min_num_candidates` / `max_num_candidates` when the pool varies per query
- random-reference files with `avg_num_candidates`, `min_num_candidates`, `max_num_candidates`, and `avg_num_positives`

## Metric normalization used in analysis

For one query `q`, let:

- `N_q` = candidate-pool size
- `R_q` = number of relevant candidates
- `K` = cutoff

The analyzer reconstructs `N_q` and `R_q` from the benchmark split plus the same
pooling settings used during evaluation, then averages the random-guess expectation over
the evaluated queries in the relevant slice.

Random-guess expectations:

- `E[Recall@K | q] = min(K, N_q) / N_q`
- `E[Success@K | q] = 1 - prod_{i=0}^{K-1} (N_q - R_q - i) / (N_q - i)`
- `IDCG(R_q, K) = sum_{i=1}^{min(R_q, K)} 1 / log2(i + 1)`
- `E[nDCG@K | q] = ((R_q / N_q) * sum_{i=1}^{min(K, N_q)} 1 / log2(i + 1)) / IDCG(R_q, K)`
- `E[MRR | q] = sum_{j=1}^{N_q - R_q + 1} (1 / j) * C(N_q - j, R_q - 1) / C(N_q, R_q)`
- `E[ROC-AUC] = 0.5`
- `E[EER] = 0.5`

Chance-adjusted normalized metrics:

- For higher-is-better metrics (`success`, `recall`, `ndcg`, `mrr`, `roc_auc`):
  - `normalized = (score - random) / (1 - random)`
- For `eer`:
  - `normalized_eer = (0.5 - eer) / 0.5 = 1 - 2 * eer`

This normalization maps random guessing to `0`, perfect performance to `1`, and
worse-than-random performance to negative values. It is the quantity exported as
`normalized_<metric>` in the summary tables and under the `normalized/` wide-table
directories.

For a full topic-leakage sweep (topic-matched pools + TF-IDF baseline + CSV exports),
see `eval/scripts/run_topic_leakage.sh`.

Both `runner.py` and `train.py` emit fine-grained breakdowns by language, genre, and
token-length bucket under `by_language`, `by_genre`, and `by_length_bucket` in addition
to the overall metrics. These are written to stdout, any `--output-json`, optional
JSONL logs, and W&B (if enabled) for downstream analysis.

## Training + evaluation

`train.py` fine-tunes an embedding model with an in-batch InfoNCE-style loss over
(query, positive-candidate) pairs derived from `ground_truth.jsonl`. Evaluation runs at
step 0, every `--eval-every` steps, and at the end on both dev and test.

The trainer also supports three authorship-specific recipes via `--authorship-method`:
- `part` – an AuthBench-adapted PART recipe: the base encoder is frozen by default, a
  BiLSTM head is trained over token states, and same-author document pairs are optimized
  with a PART-style learnable-temperature contrastive objective. Evaluation remains
  document-level, so every query/candidate gets one embedding. (https://arxiv.org/pdf/2209.15373)
- `luar` – an AuthBench-adapted LUAR recipe: each author is represented by variable-size
  episodes of `32`-token windows, self-attention + max-pooling aggregate windows into an
  authorship embedding, and training uses supervised contrastive learning over two
  sampled episodes per author. At evaluation time, one document embedding is built from
  all `32`-token windows in the document by default, optionally capped for runtime.
  (https://aclanthology.org/2021.emnlp-main.70.pdf)
- `stel` – an AuthBench-adapted STEL/CAV-style recipe: the encoder is tuned with a
  content-controlled triplet objective where negatives are chosen from the same
  `source+genre`, then same `source`, then same `genre`, before falling back to a random
  different-author negative. The public method name stays `stel`, but this is a training
  recipe inspired by the paper's content-control setup rather than the STEL evaluation
  framework itself. (https://aclanthology.org/2022.repl4nlp-1.26.pdf)
- `standard` – the query/candidate InfoNCE baseline (default) and the recommended control
  recipe for direct comparison to the authorship-specific adaptations.

Example: fine-tune and evaluate `bge-base-en-v1.5` with periodic metrics:

```bash
python -m AuthBench.eval.train \
  --model bge-base-en-v1.5 \
  --batch-size 16 \
  --epochs 1 \
  --eval-every 500 \
  --dataset-root /path/to/outputs/official_ttl300k_cap10M_sf10k_postprocessed \
  --output-dir checkpoints/bge-base \
  --log-file logs/bge_base.jsonl
```

Useful flags:
- `--query-prefix/--doc-prefix` to add model-specific prompts (e.g., E5's `query:` /
  `passage:`).
- `--max-eval-queries/--max-eval-candidates` to cap evaluation size.
- `--authorship-method standard|part|luar|stel` to switch between the baseline and the
  three authorship-specific training recipes.
- `--max-train-authors` to cap author-centric training for PART/LUAR/STEL.
- `--part-freeze-encoder` / `--part-train-encoder` to keep PART paper-aligned by
  freezing the encoder by default, or explicitly allow encoder tuning.
- `--part-temperature-init` to set PART's learnable temperature/logit-scale
  initialization.
- `--luar-window-size` to control LUAR's excerpt length (defaults to `32` total tokens).
- `--luar-episode-length` and `--luar-samples-per-author` to control LUAR's
  supervised-contrastive episode construction.
- `--luar-max-eval-windows` to cap LUAR's eval-time full-document window aggregation for
  long documents. If omitted, LUAR uses every `32`-token window in the document.
- `--luar-max-episode-docs` / `--luar-eval-episode-docs` remain accepted as backward-compatible
  aliases for the newer LUAR flags.
- `--stel-control-keys` to change the metadata priority used for STEL/CAV-style
  content control. The default order is `source genre`, which yields
  `source+genre -> source -> genre -> random`.
- `--negatives-per-query` and `--negative-strategy` to balance attribution EER runtime.
- `--trust-remote-code` to pre-approve HF repos that ship custom modeling code
  (scripts will also auto-retry with `trust_remote_code=True` when transformers
  explicitly requests it; disable that fallback with `--no-auto-trust-remote-code`).
- Some checkpoints only publish PyTorch `.bin` weights. Transformers now blocks
  unsafe `torch.load` on torch<2.6; either upgrade torch or install `safetensors`
  so the loaders can grab `.safetensors` weights when available.
- `--late-interaction` for max-sim scoring during eval (memory-heavy; pair with caps).
- `--candidate-chunk-size` to control candidate token batch size during late interaction.
- Late interaction pads to `--max-length` to keep token tensors alignable; lower the
  length or subset queries/candidates if memory spikes.
- `--wandb-project` (runner/train) to push metrics to Weights & Biases; combine with
  `--wandb-run-name/--wandb-entity/--wandb-tags` as needed.
- `--eval-fraction-epoch` to trigger evals at a fraction of each epoch (e.g., 0.5 for mid-epoch) and
  `--eval-every-epoch` to always evaluate at epoch boundaries.
- `--lora-rank` to enable LoRA adapter tuning (`0` disables LoRA). Pair with
  `--lora-alpha/--lora-dropout/--lora-bias/--lora-target-modules` for custom setups.

Scripts:
- `eval/scripts/train_model.sh <model-name>` – run one model for 1 epoch with mid-epoch eval and LoRA (rank 16 by default; override with `LORA_RANK`). Results are written under `eval/results/training_summary/<model>/`.
- `eval/scripts/train_all_models.sh` – train the default top-2 models per group (LLM-base, LLM-instruct, Embedding, Embedding-instruct) with LoRA rank 16. Override the list with `MODELS="m1 m2 ..."`.
- `eval/scripts/train_eval_authorship_methods.sh` – train and evaluate `standard`, `part`,
  `luar`, and `stel` on a single base model (defaults to `qwen3-emb-4b`) using one
  consistent AuthBench dataset root for direct comparison.
- `eval/scripts/eval_all_models.sh` – evaluate a broad set of embedding models (or override via `MODELS="m1 m2"`) and store per-model JSON outputs with fine-grained breakdowns for leaderboard building.
Checkpoints are saved under `--output-dir/<model>` with a `training_summary.json` that
captures the final dev/test metrics for quick comparison to pre-trained baselines.

For historical result folders, note that older authorship-method runs and the generic
`train_model.sh` workflow may not be directly comparable if they used a different
dataset root, batch regime, or pre-adaptation LUAR/STEL defaults.
