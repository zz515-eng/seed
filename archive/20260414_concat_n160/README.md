# 2026-04-14 archive

This folder is the previous release. It is the concatenated model trained on the avoid50 N=160 table, with dropout raised to 0.5.

Do not load these weights with the current `model/model.py`. That file is the later FiLM model and reads the center base.

- `weights/best_seed_model.pt`: fine-tuned weights for the N=160 table.
- `weights/base_human_model.pt`: human checkpoint used before that fine-tune.
- `data/train.csv`, `val.csv`, `test.csv`: 160 / 40 / 248 rows. `sample_train_ultimate.csv` is the same training table.
