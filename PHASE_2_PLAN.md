# Phase 2 Plan: ABSA Annotation and Baseline Integration

## Objective

Phase 2 moves the project from a rule-based aspect demo to a real ABSA baseline backed by manually labeled data.

The minimum successful outcome for this phase is:

- At least 1,000 fully labeled ABSA reviews.
- A reproducible multi-label ABSA baseline trained on the expanded dataset.
- Metrics and confusion matrix per aspect.
- API and dashboard prepared to consume model-predicted aspects instead of relying only on lexicon rules.

## Current Status

The project already has:

- Sentiment preprocessing, feature extraction, and baseline training.
- Flask API and Streamlit dashboard.
- A seed ABSA annotation file at `data/absa_annotation_sample.csv`.
- A training script for aspect detection at `scripts/train_absa_baseline.py`.

The current ABSA baseline is only a smoke test because it was trained on 30 labeled rows.

## Scope

This phase includes:

- Creating new annotation batches from the raw review dataset.
- Labeling ABSA aspect columns for five target aspects:
  - `product`
  - `price`
  - `delivery`
  - `service`
  - `app`
- Combining labeled batches into a training-ready dataset.
- Re-training and evaluating the ABSA baseline.
- Preparing the integration path for API/dashboard.

This phase does not include:

- Bi-LSTM experiments
- PhoBERT fine-tuning
- Final comparative deep learning results

## Deliverables

1. Annotation assets

- `data/absa_annotation_batch_001.csv`
- `data/absa_annotation_batch_002.csv`
- `data/absa_annotation_batch_003.csv`
- Optional additional batches if needed to reach 1,000 labeled rows.

2. Consolidated labeled dataset

- `data/absa_annotation_master.csv`

3. Model and evaluation outputs

- `artifacts/models/absa_aspect_baseline.joblib`
- `artifacts/reports/absa_baseline_metrics.json`
- `artifacts/reports/confusion_matrix_absa_aspects.csv`

4. Integration updates

- API path for model-based aspect prediction
- Dashboard path for model-based aspect display with fallback if model is unavailable

## Execution Steps

### Step 1: Lock annotation rules

Use `docs/absa_annotation_guide.md` as the source of truth.

Before large-scale annotation:

- Keep labels binary: `1` means the aspect appears, `0` means it does not.
- Leave no blank aspect columns after annotation is complete.
- Allow multiple aspects in the same review.
- Keep all-zero rows when the review is generic and does not mention a specific aspect.

Exit criteria:

- The labeling rules are stable enough that repeated annotation does not drift.

### Step 2: Generate annotation batches

Create balanced samples from the raw dataset and split them into manageable batch files.

Recommended batch layout:

- Batch 001: 300 reviews
- Batch 002: 300 reviews
- Batch 003: 400 reviews

Exit criteria:

- Enough batches exist to reach the phase minimum of 1,000 labeled rows.

### Step 3: Annotate and review

For each batch:

- Fill `product`, `price`, `delivery`, `service`, `app`.
- Review 10% of rows after each batch for consistency.
- Pay attention to common confusions:
  - `service` vs `app`
  - `product` vs `price`
  - `delivery` vs general complaint wording

Exit criteria:

- Each batch is fully labeled with no blanks in aspect columns.

### Step 4: Consolidate labeled data

Merge all completed batches into one master annotation file:

- `data/absa_annotation_master.csv`

Validation requirements:

- No duplicate `reviewid`
- No missing `content`
- No blank aspect columns
- Only `0` or `1` in aspect columns

Exit criteria:

- The master file is clean enough to train without dropping most rows.

### Step 5: Retrain ABSA baseline

Train the multi-label baseline using the consolidated dataset.

Recommended command:

```powershell
.\.venv\Scripts\python.exe scripts\train_absa_baseline.py --input data\absa_annotation_master.csv --segmenter none
```

Track:

- `subset_accuracy`
- `hamming_loss`
- `precision_micro`, `recall_micro`, `f1_micro`
- `precision_macro`, `recall_macro`, `f1_macro`
- Per-aspect metrics

Exit criteria:

- Metrics are generated from a labeled dataset large enough to report academically.

### Step 6: Prepare inference integration

Update the inference flow so aspect prediction can come from the ABSA model.

Target behavior:

- Predict sentiment as before.
- Predict aspects from the ABSA baseline.
- Map predicted aspects plus sentiment and engagement signals into RACE and alert logic.
- Keep lexicon rules only as fallback or comparison baseline.

Exit criteria:

- The code path for model-based aspects is ready and testable.

### Step 7: Freeze the phase

Before moving to deep learning:

- Re-run tests
- Verify training outputs
- Update project docs with current ABSA dataset size and baseline metrics

Exit criteria:

- The repo has a stable ABSA baseline milestone.

## Suggested Timeline

### Week 1

- Finalize guide and generate all annotation batches.

### Week 2

- Label 600 to 1,000 reviews.

### Week 3

- Consolidate labels and retrain ABSA baseline.

### Week 4

- Integrate model-based aspects into API/dashboard and freeze results.

## Immediate Next Actions

1. Generate at least two new annotation batches from the raw dataset.
2. Start labeling batches until the project reaches 1,000 fully labeled rows.
3. Create `absa_annotation_master.csv`.
4. Re-train the ABSA baseline and record the new metrics.

## Progress Update

Completed after the initial Phase 2 setup:

- Added a prefilled annotation workflow to reduce manual work.
- Expanded the working ABSA dataset to 1,029 labeled rows.
- Retrained the ABSA baseline on the merged dataset.
- Integrated model-based aspect inference into the API and dashboard, with rule-based fallback.
- Added a review-priority workflow so the highest-risk rows can be corrected first.

Current assessment:

- The ABSA pipeline is now end-to-end functional.
- The 1,029-row dataset is usable as a working training resource.
- The current metrics are still better treated as an internal milestone than as final academic evidence, because many labels originated from prefill or rapid correction.

Revised next step:

- Build a smaller `gold subset` of about 200 to 300 reviews with stricter manual review.
- Use that subset as the trusted evaluation set for ABSA baseline, Bi-LSTM, and PhoBERT comparisons.
