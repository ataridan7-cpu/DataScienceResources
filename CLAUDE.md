# CLAUDE.md

Working guidance for Claude Code in this repository. This is a data science /
ML repo (mostly Jupyter notebooks). Read this before starting any task.

## How to collaborate

- **Ask before starting.** Clarify the goal, scope, and any assumptions before
  writing or changing code. Don't guess at ambiguous requirements — ask. If
  several reasonable approaches exist, lay out the options and let me choose.
- **Plan first, then act.** For anything non-trivial, outline what you intend to
  do and wait for a go-ahead before making changes.
- **Offer improvements.** When you notice better approaches — cleaner code, a
  more suitable metric, a bug, a leakage risk, a faster method — point them out
  and suggest them. Don't silently change direction; propose, then let me decide.
- **Make minimal, scoped changes.** Do what was asked; don't refactor unrelated
  code or expand scope without checking first.
- **Explain what you did** in plain terms after a change, including trade-offs
  and anything I should verify.

## Reviewing vs. running

- **When asked to review or read code, do NOT run it.** No executing notebooks,
  no training runs, no installing packages. Read, analyze, and report.
- **Only run code when explicitly asked** to run, execute, train, or test.
- Training/execution can be slow and expensive — never kick one off as a
  side effect of another task.

## Editing safely

- Make changes incrementally and keep them reviewable.
- Don't delete or overwrite work you didn't create without flagging it first.
- Preserve existing style, structure, language, and naming conventions.
- Verify claims (library behavior, parameter support, APIs) by checking, not
  guessing.

## Jupyter notebooks

- Edit `.ipynb` files with the **`NotebookEdit`** tool, not `Edit`/`Write`.
- Large notebooks exceed the read limit — read specific cells with
  `python3 -c "import json; ..."` instead of the whole file.
- Validate JSON parses before committing:
  `python3 -c "import json; json.load(open('notebook.ipynb'))"`.
- Stored outputs go **stale** after edits and only refresh on re-run. Don't
  present stale outputs as current results, and don't re-run just to refresh
  them unless asked.

## Data science defaults

When these are relevant, follow them — but raise it first if a task implies
otherwise:

- Optimize for the metric that matches the real goal, defined once and reused
  for feature selection, tuning, thresholds, and model selection. Report the
  familiar metrics too, but optimize for the one that matters.
- Guard against data leakage: scaling, imputation, resampling, and feature
  selection go inside the CV pipeline, fit on training folds only.
- Use stratified splits, keep a held-out test set untouched until the end, and
  set `random_state` for reproducibility.

## Git

- Commit completed work with clear, descriptive messages — only when the work is
  done and I've confirmed, not mid-task.
- Push with `git push -u origin <branch>`; retry network failures with backoff.
- Do **not** open a pull request unless explicitly asked.
