# Phase 2 Progress

## Current State

The repo has moved past the original ABSA smoke-test setup.

Completed:

- Added a prefilled annotation workflow to reduce manual labeling work.
- Expanded the working ABSA dataset to 1,029 labeled rows.
- Retrained the ABSA baseline on the merged dataset.
- Integrated model-based aspect inference into the API and dashboard, with rule-based fallback.
- Added a review-priority workflow so the highest-risk rows can be corrected first.

## Current Assessment

- The ABSA pipeline is now end-to-end functional.
- The 1,029-row dataset is usable as a working training resource.
- The current metrics are still better treated as an internal milestone than as final academic evidence, because many labels originated from prefill or rapid correction.

## Latest Baseline Notes

Working dataset:

- `data/absa_annotation_master.csv`
- 1,029 labeled rows

Observed pattern after fast high-risk review:

- Delivery and price labels became more semantically consistent in several rows.
- Product labels remain the noisiest area.
- Service remains sparse and sensitive to definition boundaries.

Conclusion:

- The current 1,029-row set should remain the main working training set.
- A smaller, cleaner subset is needed for trusted model comparison.

## Recommended Next Step

Build a `gold subset` of about 200 to 300 reviews with stricter manual review.

Use that subset as the trusted evaluation set for:

- ABSA baseline
- Bi-LSTM + POS
- PhoBERT

Keep the larger 1,029-row set as a broader training resource rather than the only evaluation source.

## Immediate Next Actions

1. Create a `gold subset` from the current annotation pool.
2. Review and clean that subset more carefully than the broader prefilled set.
3. Freeze a train/test strategy based on:
   - large working training set
   - small trusted gold evaluation set
4. Re-run ABSA baseline with the new split before moving to deep learning.
