# Task 1 patch: export model bundle for Task 3

This folder contains a patched `model.py` that adds:

- `--export-model <path>`: saves a torch bundle with `state_dict` + `threshold` (and `hidden`)
- `--no-predict`: train/export only (local convenience)

## How to use

In your Task 1 repo, replace your `model.py` with this one (or manually apply the small diff).

Then run (inside the Task 1 container, or with the same deps installed):

```bash
python model.py --input=/data/test.csv --output=/data/aki.csv --export-model=/data/aki_model.pt
```

Commit the resulting `aki_model.pt` into your Task 3 repo under `./model/aki_model.pt`
(or mount it at runtime and pass `--model-bundle`).
