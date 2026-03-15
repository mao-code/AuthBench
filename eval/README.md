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

Export metric tables from JSON results:

```bash
python -m AuthBench.eval.export_results \
  --results-dir eval/results \
  --output-dir eval/results/analysis \
  --metrics success@10 recall@10 ndcg@10 eer@10
```

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
- `eval/scripts/eval_all_models.sh` – evaluate a broad set of embedding models (or override via `MODELS="m1 m2"`) and store per-model JSON outputs with fine-grained breakdowns for leaderboard building.
Checkpoints are saved under `--output-dir/<model>` with a `training_summary.json` that
captures the final dev/test metrics for quick comparison to pre-trained baselines.
