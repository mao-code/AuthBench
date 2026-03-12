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

For generation-capable causal LLMs, you can enable self-consistency score aggregation:
sample multiple style descriptions with top-k sampling, embed each sample separately,
sum the per-sample query-candidate similarities, and rerank the full candidate pool:

```bash
python -m AuthBench.eval.runner \
  --split test \
  --models qwen2.5-3b-instruct \
  --self-consistency \
  --self-consistency-samples 4 \
  --self-consistency-top-k 50 \
  --self-consistency-temperature 0.8 \
  --self-consistency-max-new-tokens 96
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

For `--self-consistency`, the flow is different and does require autoregressive
generation:

1. Prompt the LLM to write a short authorship style profile for the document.
2. Sample `N` style profiles with top-k / temperature decoding.
3. Decode the generated continuation tokens into text.
4. Re-tokenize each generated style profile and run a standard forward pass.
5. Pool each generated profile into one embedding.
6. Compare query profile embedding `i` only against candidate profile embedding `i`.
7. Sum the per-sample query-candidate similarity scores and rerank with that summed score.

`--self-consistency-include-original` optionally appends one direct embedding of the
original document as an extra term in that summed-score aggregation.

TF-IDF baseline (cosine on TF-IDF vectors) is available via the dedicated runner:

```bash
python -m AuthBench.eval.tfidf_runner \
  --dataset-root /path/to/outputs/official_ttl300k_cap10M_sf10k_postprocessed \
  --split test \
  --output-json eval/results/tfidf.json
```

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
- `--self-consistency` to sample multiple style descriptions from supported causal LLMs
  and sum/rerank the resulting per-sample retrieval scores instead of averaging vectors. Tune with `--self-consistency-samples`,
  `--self-consistency-top-k`, `--self-consistency-temperature`, and
  `--self-consistency-max-new-tokens`. Add `--self-consistency-include-original` if you
  want to add the direct document embedding as one extra sampled embedding in the score sum.
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
- `eval/scripts/self-consistency/eval_model.sh <model-name>` – run one causal LLM with generation-based, summed-score self-consistency retrieval.
- `eval/scripts/self-consistency/eval_all_llms.sh` – sweep the registry’s supported causal LLMs with the same self-consistency settings.

Checkpoints are saved under `--output-dir/<model>` with a `training_summary.json` that
captures the final dev/test metrics for quick comparison to pre-trained baselines.
