# CLAUDE.md

Guidance for Claude Code when working in this repository.

## Repository

A data science portfolio / resources repo. The active project is a **student
dropout prediction** ML pipeline built as a Jupyter notebook (Hebrew-language
write-up).

- `Copy_of_dropout_improved_1.ipynb` — the main project notebook (large, ~900 KB).
- `README.md` — project documentation.

## Working with the notebook

- It is a `.ipynb` (JSON) file. **Edit cells with the `NotebookEdit` tool, not
  `Edit`/`Write`.** `NotebookEdit` requires a fresh read of the file first.
- The file is too large to `Read` whole (exceeds the token limit). Read specific
  parts with `python3 -c "import json; ..."` via Bash, or extract a single
  cell's source to a temp file and `Read` that.
- After editing, validate the JSON still parses before committing:
  `python3 -c "import json; json.load(open('Copy_of_dropout_improved_1.ipynb'))"`.
- Saved cell outputs go stale after edits — they only refresh when the notebook
  is re-run (e.g. in Colab). Don't treat stored outputs as current results.

## Project conventions (dropout pipeline)

- **Custom scorer is the optimization target everywhere.** A custom F1 of
  `0.6 × F1(minority/Dropout=1) + 0.4 × F1(majority/Dropout=0)` is defined near
  the top as `custom_f1` / `custom_scorer`, with `SCORING = custom_scorer`. Use
  `scoring=SCORING` for all optimization: feature selection, SMOTE-ratio tuning,
  hyperparameter search (GridSearchCV / RandomizedSearchCV), threshold tuning,
  and best-model selection (highest `custom_f1`).
- **F1-macro is reported, not optimized.** Keep `f1_macro` in display tables and
  final test reports for comparison, but never use it to drive a choice.
- **Class imbalance is handled three ways:** SMOTE in the `ImbPipeline` (all
  models), decision-threshold tuning toward custom F1 (all models, range
  0.20–0.80), and tunable `class_weight=[None, 'balanced']` (SVM only — KNN and
  MLP have no `class_weight` parameter in scikit-learn).
- Threshold tuning uses `predict_proba`; `SVC` must be built with
  `probability=True`.
- Cross-validation uses `StratifiedKFold` (nested inner/outer).

## Git

- Develop on branch `claude/blissful-pasteur-wEWHl`.
- Push with `git push -u origin <branch>`; retry network failures with
  exponential backoff.
- Commit when work is complete with clear messages. Do **not** open a pull
  request unless explicitly asked.
