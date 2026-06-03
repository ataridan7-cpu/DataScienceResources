# CLAUDE.md

Working guidance for Claude Code on data science / ML tasks in this repository.

## Core principle: optimize for the metric that matches the goal

- Pick the scoring metric from the **real-world objective**, not the library
  default. If one class matters more (e.g. the rare/positive class), weight the
  metric toward it rather than defaulting to `f1_macro`, `accuracy`, etc.
- Define that target **once** and reuse it everywhere. A single `SCORING`
  variable (e.g. a `make_scorer`-wrapped custom function) should drive **all**
  optimization decisions:
  - feature selection
  - resampling / SMOTE ratio tuning
  - hyperparameter search (Grid/RandomizedSearchCV)
  - decision-threshold tuning
  - final best-model selection
- **Optimize for one metric, report many.** Keep familiar metrics
  (`f1_macro`, precision/recall per class, accuracy) in result tables for
  interpretability, but never let a reported-only metric silently drive a choice.
  Be explicit in code/comments about which metric is the target and which are
  just displayed.

## Class imbalance

Treat imbalance deliberately and stack the techniques that apply:

- **Resampling** (e.g. SMOTE) — put it *inside* the CV pipeline
  (`imblearn.Pipeline`) so it only sees training folds; never resample before
  splitting (causes leakage).
- **Decision-threshold tuning** — tune the probability cutoff on a
  validation set toward the target metric. Apply it to **every** model that
  exposes `predict_proba`, not just one. Store each model's chosen threshold and
  reuse it at test time.
- **`class_weight`** — only where the estimator supports it. Verify support by
  introspection instead of assuming (scikit-learn's KNN and MLP have no
  `class_weight`; SVC/trees/linear models do). When unsure whether `balanced`
  helps, make it a tunable choice (`[None, 'balanced']`) so CV decides.

## Validation discipline

- Use **stratified** splits/folds for classification. Prefer nested
  cross-validation (inner = tuning, outer = unbiased performance estimate).
- Keep a held-out test set untouched until final evaluation. Tune thresholds on
  validation, not on test.
- Set `random_state` for reproducibility.
- Watch for leakage: scaling, imputation, resampling, and feature selection all
  belong inside the CV pipeline, fit on training data only.

## Working style

- **Integrate, don't bolt on.** Prefer folding a step into the existing
  pipeline/cell over adding a parallel one-off cell that can drift out of sync.
- **Apply changes consistently across all models**, not just the first one, when
  a request is general ("do it for all models").
- **Verify claims about libraries** (parameter existence, behavior) by checking
  the actual API rather than guessing.
- Remove dead fallbacks once a value is always computed — prefer direct access
  over defensive `.get(..., <stale default>)`.
- Match the existing notebook/codebase language, style, and naming.

## Jupyter notebooks

- Edit `.ipynb` files with the **`NotebookEdit`** tool, not `Edit`/`Write`;
  it needs a fresh read of the file first.
- Large notebooks exceed the `Read` token limit — read specific cells via
  `python3 -c "import json; ..."` in Bash instead of reading the whole file.
- Validate JSON parses before committing:
  `python3 -c "import json; json.load(open('notebook.ipynb'))"`.
- Stored cell outputs go **stale** after edits; they only refresh on re-run
  (e.g. in Colab). Don't present stale outputs as current results.

## Git

- Commit completed work with clear, descriptive messages.
- Push with `git push -u origin <branch>`; retry network failures with
  exponential backoff.
- Do **not** open a pull request unless explicitly asked.
