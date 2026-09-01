# v0.2.0 research gate

The next runtime phase is allowed to begin only after route collection answers
these questions with observed data:

1. With 18 cached Experts per layer, what is the decode
   `per_selection_hit_rate` for each domain?
2. With 18 cached Experts per layer, what is the decode
   `all_selected_hit_rate` for each layer and domain?
3. How much do the selected Expert sets overlap between consecutive tokens?
4. Do the hottest Experts differ materially across coding, defensive security,
   mathematics, and general prompts?
5. How many warm-up tokens are needed before the top-N set stabilizes?
6. What fraction of events would require RAM or NVMe service under static,
   session-adaptive, and mixed cache policies?
7. Does tracing itself change generation throughput enough to require a lower
   sampling rate or a compact binary collector in v0.3?

## Promotion criteria for v0.3 runtime work

The first dynamic-cache prototype should proceed when at least one of the
following is observed across multiple sessions:

- top-18 per-selection hit rate >= 70%;
- top-18 all-selected hit rate >= 40%;
- consecutive-token overlap >= 50%;
- a mixed domain/session policy improves all-selected hit rate by >= 10 percentage
  points over a global static cache.

These are engineering gates, not claims that such rates are sufficient for a
specific token/s target.
