# Reproducibility

Someone with the data bundle and this repository reproduces the submission file byte for byte. This
page says what that takes.

## Environment

| | version |
|---|---|
| Python | 3.14.3, 64-bit (CPython) |
| numpy / pandas / scipy | 2.4.3 / 3.0.6 / 1.18.1 |
| scikit-learn / joblib / threadpoolctl | 1.9.1 / 1.6.0 / 3.7.0 |
| numba / llvmlite | 0.68.0 / 0.50.0 |
| verified on | Windows 11 x64 (Intel, 12 logical cores, 7.3 GB RAM) |

All pinned in `requirements.txt` (`requirements-dev.txt` adds pytest). `pyproject.toml` declares the same
packages with compatible ranges, for installing the library elsewhere; the exact versions are the ones
verified to reproduce the file byte for byte.

## Determinism

- **Seeds.** Every random draw derives from `runtime.seed` (20271117): the DFL degradations (`seed`, and
  `seed + 1` for the synthetic stoppage ball) and the gradient-boosted trees (`seed`, `seed + 1`,
  `seed + 2` for the team model's three members).
- **Threads.** The entry points fix the number of OpenMP threads with threadpoolctl to
  `runtime.threads = 4`, whatever the machine.
- **Stored model outputs.** The decoders read the model probabilities as stored, in float32
  (`outputs/model_outputs.npz`).
- **Line endings.** The CSV is written with `\n` on every platform; `.gitattributes` keeps git from
  converting the committed CSV files.
- **Caches** (`cache/`) are keyed on the source code of the modules that produce them, the inputs and the
  parameters; deleting the folder is always safe.

## Run times (the machine above)

| step | first run | with the cache |
|---|---|---|
| `train/train.py` | ~28 min (7 DFL matches parsed, degraded twice, featurised; 4 tree fits) | ~5 min |
| `inference/predict.py` | ~3 min (the challenge match featurised) | ~1.5 min |

## The submitted file

`outputs/submission.csv`, 199,319 rows, SHA-256

```
46c69c3e559135a0e716eb316b67f13c953f1fc18378b3bace57fda242c21716
```

`python scripts/reproduce.py` downloads the DFL open data (files already present and verified are kept),
checks the match files and the DFL files against their checksums, trains from the data, predicts, and compares the file's SHA-256
with the value above. `--skip-train` reuses the models in `artifacts/`, `--skip-download` uses the DFL
files already in place.

## Downloads

| data | script | source | size | check |
|---|---|---|---|---|
| DFL open data | `scripts/download_dfl.py` | figshare article 28196177 (public, CC BY 4.0) | ~2.6 GB | size and MD5 published by figshare |

The script retries failed downloads and skips files that are already present and verified. The match
bundle is the organisers' own and is not downloaded.
