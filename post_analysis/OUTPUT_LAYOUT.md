## Post-analysis output layout

This note documents the intended structure for generated benchmark-analysis artifacts under `post_analysis/outputs/`.

Recommended layout:

```text
post_analysis/outputs/
├── combined_phase1_phase2/
│   ├── statistics/
│   ├── qualitative/
│   ├── benchmark_profile/
│   └── leakage_audit/
├── phase1_vs_phase2/
│   ├── per_benchmark/
│   └── comparison/
└── phase1_official_plus_phase2_all4_all_docs/   # legacy combined run already present
```

Conventions:

- Use `combined_phase1_phase2/` for new runs on `processing/outputs/combined_phase1_phase2`.
- Inside `combined_phase1_phase2/`, keep:
  - `statistics/` for the main quantitative tables and figures
  - `qualitative/` for qualitative diagnostics and leakage checks
  - `benchmark_profile/` for the supplementary authorship-benchmark balance, stage-flow, and leakage-risk outputs
  - `leakage_audit/` for the reviewer-facing topic/language shortcut audit and metadata-only baselines
- Use `phase1_vs_phase2/` for side-by-side comparison outputs between:
  - `processing/outputs/pipeline_phase1_official`
  - `processing/second_phase_web_crawling/outputs/pipeline_phase2_official`
- Keep each benchmark run in its own directory so CSVs, figures, and markdown reports stay grouped together.
- Treat `phase1_official_plus_phase2_all4_all_docs/` as a historical run directory rather than the canonical name going forward.
