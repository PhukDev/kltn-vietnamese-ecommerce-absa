# Phase 2 Progress

## Current State

The repo has moved past the original ABSA smoke-test setup.

Completed:

- Added a prefilled annotation workflow to reduce manual labeling work.
- Expanded the working ABSA dataset to 1,029 labeled rows.
- Retrained the ABSA baseline on the merged dataset.
- Integrated model-based aspect inference into the API and dashboard, with rule-based fallback.
- Added a review-priority workflow so the highest-risk rows can be corrected first.
- Created `data/absa_gold_subset_candidate.csv` as the next-stage trusted evaluation candidate set.

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

## Gold Subset Candidate

Prepared artifact:

- `data/absa_gold_subset_candidate.csv`
- `data/absa_working_train.csv`
- `data/absa_gold_eval.csv`
- `data/absa_gold_eval_review_queue.csv`

Current size:

- 250 reviews

Selection intent:

- Includes all currently high-risk rows.
- Includes the medium-risk rows that are most useful for error reduction.
- Keeps strong coverage for `service`, `app`, `delivery`, `price`, and `product`.

Recommended handling:

1. Review this 250-row candidate set carefully.
2. Treat it as the trusted evaluation subset once the labels are cleaned.
3. Keep the broader 1,029-row dataset as the larger working training pool.

## Split Prepared

Current split:

- Working train set: 779 rows
- Gold eval candidate: 250 rows

Prepared workflow:

- `data/absa_working_train.csv` is the broader training pool.
- `data/absa_gold_eval.csv` is the evaluation candidate snapshot.
- `data/absa_gold_eval_review_queue.csv` is the review-first file with `review_status` and `review_note`.

Recommended next manual action:

1. Open `data/absa_gold_eval_review_queue.csv`.
2. Review the rows in order from top to bottom.
3. Change `review_status` from `pending` to `done` as each row is confirmed.
4. Record short notes when a label was corrected or remained ambiguous.

## Gold Review Progress

Current reviewed rows in `data/absa_gold_eval_review_queue.csv`:

- 250 rows marked as `done`

Notes:

- The first reviewed block focuses on the highest-priority rows at the top of the queue.
- Review notes are being written directly into `review_note` so later passes do not re-open the same decisions.

## Gold Eval Result

The first split-aware ABSA evaluation has been run with:

- train set: `data/absa_working_train.csv` (779 rows)
- eval set: 250 reviewed rows exported from `data/absa_gold_eval_review_queue.csv`

Observed result:

- micro F1: `0.6238`
- macro F1: `0.6279`

Interpretation:

- The current working train set does not generalize nearly as well to the cleaner gold-eval subset as it did on the earlier mixed evaluation setup.
- The main weakness is still label quality and label consistency, especially around:
  - `product`
  - `service`
  - `app`

Practical conclusion:

- This confirms that the gold-eval workflow is necessary.
- The next work should continue cleaning more rows in the gold-eval queue before moving to deep learning.

## Bi-LSTM Milestone

Deep learning environment has now been installed successfully:

- `torch`
- `transformers`
- related requirements from `requirements-dl.txt`

The first Bi-LSTM run on the current split has been completed with:

- train set: 779 rows
- eval set: 250 rows

Observed result:

- micro F1: `0.6018`
- macro F1: `0.6010`

Interpretation:

- The first Bi-LSTM result is slightly below the current ABSA baseline on the same gold-eval split.
- This means the next deep-learning step should move to PhoBERT rather than spending too long tuning this initial Bi-LSTM baseline.

## PhoBERT Milestone

The first PhoBERT run has completed successfully.

Current run conditions:

- CPU only
- 1 epoch
- train set: 779 rows
- eval set: 250 rows

Observed result:

- aspect micro F1: `0.4325`
- aspect macro F1: `0.4193`
- sentiment macro F1: `0.4297`

Interpretation:

- PhoBERT has been proven runnable in the local environment.
- This run is only an initial CPU checkpoint, not a fair final PhoBERT comparison.
- To evaluate PhoBERT properly, more epochs and preferably GPU training are still needed.

## PhoBERT GPU Milestone & Live Dashboard Integration

The official deep learning milestone has been completed successfully on GPU:

- Platform: Kaggle Notebook (GPU P100)
- Epochs: 5
- Learning Rate: `2e-5`
- Batch Size: `16`
- Train set: 779 rows
- Eval set: 250 rows (clean gold evaluation subset)

Observed GPU Training Results:

- **Aspect micro F1:** `0.5669`
- **Aspect macro F1:** `0.5465`
- **Sentiment Accuracy:** `0.6200`
- **Sentiment macro F1:** `0.5819`

Live Dashboard & API Integration:

- **Unified Prediction Service:** Upgraded `src/ecommerce_absa/api.py` to seamlessly load PyTorch PhoBERT multitask models (`.pt`) using CPU fallback, while preserving backward compatibility for baseline models (`.joblib`).
- **Interactive Streamlit App:** Re-architected `dashboard/streamlit_app.py` to unify prediction logic and allow selecting the PhoBERT `.pt` model directly from the sidebar. Shows `Aspect source: phobert_multitask` live.
- **Verification:** Unit tests passed cleanly (`Ran 6 tests - OK`). API and Dashboard verified with live PhoBERT inference in real-time.

This successfully completes the Giai đoạn 4 (Modeling) and Giai đoạn 5 (Deployment) milestones of the KDD pipeline for the ABSA graduation thesis!

## PowerPoint & Thesis Document Integration

- **PowerPoint Slide Upgrade:** Successfully upgraded `BAO_CAO_TIEN_DO_KLTN.pptx` to **11 slides** using Premium Light Theme (brand color `0E8C61`), featuring actual PowerPoint column charts with percentage data labels and a beautiful vector flowchart for sentence splitting on Slide 8.
- **Word Document Audit:** Analyzed the official thesis outline document `DCKL_KsorPhuk_V1.docx` (confirming that the other draft `KsorPhuk_2301010014_DeCuongKLTN.docx` was an unused alternative).
- **100% Perfect Alignment:** Confirmed that the completed codebase, trained models (SVM, Bi-LSTM, PhoBERT), API, live Streamlit Dashboard, and PowerPoint slides **align 100% perfectly** with the official outline in `DCKL_KsorPhuk_V1.docx` (covering the 5 KDD phases, teencode/emoji preprocessing, ABSA modeling, and the RACE/eWOM business framework).
- **Progress Report:** Updated the main progress file `THESIS_PROGRESS_AND_OUTLINE.md` to record the completed milestones, actual metrics, and the structured outline mapping.
