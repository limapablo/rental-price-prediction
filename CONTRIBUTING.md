# Contributing

Suggestions, bug reports and improvements are welcome.

This project is maintained as an applied machine-learning portfolio project, so contributions should favor reproducibility, clear evaluation and production-minded practices.

## Guidelines

- Keep preprocessing and inference behavior consistent.
- Document changes to features, training data or evaluation metrics.
- Prefer reusable pipeline code over duplicated notebook logic.
- Add tests when changing preprocessing or prediction behavior.
- Do not commit credentials or private data.
- Keep model limitations explicit.

## Commit style

```text
feat: add prediction intervals
fix: align inference preprocessing with training
refactor: move transformations into sklearn pipeline
docs: update model limitations
test: validate unseen category handling
```
