# NLSExplorer agent notes

- For model download, checkpoint loading, CPU/GPU portability, ESM offline loading, or a report
  involving `data.pkl`, read [docs/checkpoint_and_inference_troubleshooting.md](docs/checkpoint_and_inference_troubleshooting.md) before changing code.
- Maintained user-facing code lives in `src/nlsexplorer/`; the large historical scripts under
  `A2KA_train/`, `Recommendation_system/`, and `NLSEXplorer_A2KA/` are research artifacts.
- Before committing user-facing changes, run `python -m pytest -q` and
  `python -m ruff check src tests A2KA.py SCNLS.py`.
