# Ball possession from tracking: More than a Hack 2027, Phase 1

Our submission to the [More than a Hack 2027](docs/challenge/README.md) ball-possession challenge: for
each of the 199,319 frames of FC Barcelona vs Athletic Club, who has the ball (`H` home, `A` away) or
whether the ball is dead (`D`).

The model works in two stages. **Stage 1** finds when the ball is dead (out of play, a foul, a goal),
from football rules read off the tracking and a learned detector. **Stage 2** decides which team has the
ball during the live play that is left, from the shape of the two teams. Both learned parts are trained on
seven other matches, the free open data of the German Bundesliga, first made as noisy as our match's
tracking. The model never sees an answer for this match.

How it works, in plain words and with a diagram: [docs/MODEL.md](docs/MODEL.md).

## Quick start

Python **3.14.3** (64-bit). Every package is pinned in `requirements.txt`.

```bash
python -m venv .venv
.venv/Scripts/activate            # Windows;  source .venv/bin/activate on macOS / Linux
pip install -r requirements.txt
pip install -e .
```

Put the organisers' match bundle in place (not redistributable): `data/tactical/` must hold
`BAR-ATH_tactical_atd.txt`, `BAR-ATH_tactical_atd.xml` and `BAR-ATH_tactical_smart_tagging.xml`, or set
`MTAH_DATA_DIR` to the folder that contains `tactical/` (not to `tactical/` itself). Nothing else of the
bundle is needed. Then one command does the rest:

```bash
python scripts/reproduce.py
```

It downloads the training data (the DFL open data, public on figshare, CC BY 4.0, ~2.6 GB, no account
needed), checks the match files and the DFL files against the checksums the submission was made from, trains both models,
writes **`outputs/submission.csv`** and compares it with the SHA-256 of the submitted file.

The same steps one by one:

1. `python scripts/download_dfl.py`: the DFL open data into `data/external/dfl/`, MD5-checked against figshare.
2. `python scripts/check_data.py`: verifies the match files and the DFL files.
3. `python train/train.py`: trains the two models into `artifacts/` (~30 min the first time; DFL parsing
   and degradation are cached in `cache/`).
4. `python inference/predict.py`: writes `outputs/submission.csv` and runs the official format check.

## Repository layout

```
configs/default.toml        every parameter of the pipeline (one file, all keys required)
lib/possession/             the library
  io/                       organisers' ATD + Smart Tagging, DFL open data, submission CSV
  features/                 team-shape features, dead-ball cues and the hand-weighted cue rules
  rules/                    dead-ball rules read from the tracking (observed, forced, formations)
  degradation/              the challenge tracking's profile and its simulation on DFL tracking
  models/                   the dead/alive head and the team model (gradient-boosted trees)
  decoding/                 explicit-duration Viterbi, forced layers, metric playbook
  pipeline/                 stage 1, stage 2, training tables, train / predict orchestration
train/train.py              entry point: fit both models on DFL -> artifacts/
inference/predict.py        entry point: label the match -> outputs/submission.csv
scripts/                    data download and checks, end-to-end reproduction
evaluation/                 the organisers' scorer, validator and public 5-minute ground truth (unchanged)
tests/                      unit tests (pytest), no match data needed
docs/                       how it works, the data, reproducibility, the technical report
docs/challenge/             the organisers' challenge documents (unchanged)
```

## Documentation

- [docs/PIPELINE.md](docs/PIPELINE.md): the pipeline end to end in Simplified Technical English: every input file, its source and check, every step.
- [docs/MODEL.md](docs/MODEL.md): how the model works, in plain words, with a diagram of the pipeline. Start here.
- [docs/PARAMETERS.md](docs/PARAMETERS.md): every configuration value, what it does and why.
- [docs/DATA.md](docs/DATA.md): every input, its source, licence and checksum; what is committed and why.
- [docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md): environment, determinism, run times, verification.
- [docs/TECHNICAL_REPORT.md](docs/TECHNICAL_REPORT.md): the technical document of the submission.

## Credits and licences

Code: MIT (see `LICENSE`). The evaluation code and challenge documents are the organisers'. DFL open data:
Bassek et al. (2025), [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). The match data is licensed by the organisers for the competition only and is not included.
