# Qualitative Analysis Report

- dataset_dir: `processing/outputs/combined_phase1_phase2_1m`
- splits: `dev, test, train`
- generated_at_utc: `2026-02-24T15:22:50.978927+00:00`

## Basic Statistics
- total_docs: `14688`
- total_docs_with_author: `14688`
- unique_authors: `7344`
- docs_per_author_min: `2`
- docs_per_author_avg: `2.0000`
- docs_per_author_max: `2`

## Per-Author Genre Entropy
- authors_total: `7344`
- authors_multi_genre: `0`
- pct_authors_multi_genre: `0.00%`
- avg_normalized_genre_entropy: `0.0000`

## Cross-Genre Author Analysis
- authors_multi_genre: `0`
- genre_pairs_nonzero: `0`

## Topic-Controlled Same-Genre Pairs
- topic_groups_processed: `18`
- topic_clusters_summarized: `136`
- pairs_exported: `674`
- same_author_pairs: `266`
- different_author_pairs: `408`

## Embedding Visualization
- points_used: `3000`
- vectorizer_features: `12000`
- svd_components: `64`
- tsne_perplexity: `35`

## Key Output Files
- csv/basic/summary_overview.json
- csv/entropy/author_genre_entropy.csv
- csv/cross_genre/cross_genre_author_pairs.csv
- csv/topic_pairs/topic_controlled_same_genre_pairs.csv
- csv/embedding/embedding_projection_points.csv
- figures/author_genre_entropy_histogram.png
- figures/cross_genre_author_overlap_heatmap.png
- figures/embedding_projection_by_language.png
- figures/embedding_projection_by_primary_genre.png