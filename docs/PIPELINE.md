# Ball possession pipeline: end-to-end description

## 1. Purpose

The pipeline writes one file, `outputs/submission.csv`. The file has one row for each of the 199,319 video frames of the challenge match. Each row has the columns `frame` and `label`. The label `H` tells that the home team has the ball. The label `A` tells that the away team has the ball. The label `D` tells that the frame is a stoppage. The label `X` (abstain) is permitted, but the pipeline does not write it.

### Terminology

| Term | Meaning in this document |
|---|---|
| frame | One video frame of a match, 1/25 s. Row *i* of the CSV is video frame *i* (ATD `frame_id` *i* + 1). |
| graded frame | A frame inside one of the two halves that the organisers score (`GRADED_HALVES`). |
| stoppage | A frame where the ball is dead. Label `D`. |
| live frame | A frame where the ball is in play. Label `H` or `A`. |
| possession | A continuous run of live frames with the same team label. |
| live block | A continuous run of live frames between two stoppages, inside one half. Also called a live stretch. |
| challenge match | The match that the pipeline labels. Its labels are not an input. |
| DFL data | The DFL open data: seven Bundesliga matches with per-frame labels from the data provider. |
| degraded DFL tracking | DFL tracking that the pipeline changes so that it has the statistics of the challenge tracking. |
| cue rules | A logistic function of 18 dead-ball cues with fixed weights. |
| forced layer | A set of frames that a rule sets to stoppage or to live, before and after the decoder. |
| decoder | The explicit-duration Viterbi decoder in `decoding/viterbi.py`. |

## 2. Inputs

| Path | Source | How the pipeline gets it | Check | Use |
|---|---|---|---|---|
| `data/tactical/BAR-ATH_tactical_atd.txt` | Organisers' bundle | The operator puts it in place (or sets `MTAH_DATA_DIR`). | Size and SHA-256 (`scripts/check_data.py`) | Player and ball positions, 25 fps |
| `data/tactical/BAR-ATH_tactical_atd.xml` | Organisers' bundle | As above | Size and SHA-256 | Teams, goalkeepers, attack direction |
| `data/tactical/BAR-ATH_tactical_smart_tagging.xml` | Organisers' bundle | As above | Size and SHA-256 | `SET PIECES` and `GOALS` tags |
| `data/external/dfl/DFL_02_01_matchinformation_*.xml` (7 files) | DFL open data on figshare | `scripts/download_dfl.py` | Size and MD5 (figshare values, again in `scripts/check_data.py`) | Teams, goalkeepers, pitch size |
| `data/external/dfl/DFL_04_03_positions_raw_observed_*.xml` (7 files) | DFL open data on figshare | `scripts/download_dfl.py` | Size and MD5 | Positions, `BallPossession` and `BallStatus` labels |
| `configs/default.toml` | Committed in repo | Not applicable | Not applicable | All parameters |
| `evaluation/validate_submission.py`, `evaluation/possession_metrics.py`, `games.py` | Committed in repo (organisers' code) | Not applicable | Not applicable | Format check of the CSV |
| `cache/*` | Calculated at run time | Parse and feature steps | Key from source code, inputs and parameters | Parsed tracking, feature tables, training rows |
| `artifacts/deadball_head.joblib`, `artifacts/team_model.joblib` | Calculated at run time | `train/train.py` | Final SHA-256 of the CSV | The two trained models |

The DFL download also gets the DFL events files. The pipeline does not read them.

## 3. Procedure

Start the full procedure with one command: `python scripts/reproduce.py`. The script does the steps that follow.

### 3.1 Download and check

1. Download the 14 DFL files with `scripts/download_dfl.py`. The script keeps the files that are already correct.
2. Make sure that each DFL file has the figshare size and MD5. The script tries again up to five times.
3. Run `scripts/check_data.py`. It checks the match files and the DFL files.
4. Stop if one file is missing or different.

### 3.2 Prepare the challenge match

1. Read the ATD positions. Multiply the 0–1 pitch units by the pitch size, 105.3 m × 68.0 m.
2. Read the ATD metadata and the Smart Tagging file.
3. Find the kick-off frame of each half from the tracking and the kick-off tags.
4. Calculate the team features of each frame (`features/team.py`).
5. Calculate the dead-ball cues of each frame (`features/deadball.py`).

The team features describe how each team stands and moves in its attack direction. The cues use the simpler ball-out rules (`[stage1.cues]`).

### 3.3 Make the DFL training data

1. Read the seven DFL matches.
2. Move the DFL origin from the pitch centre to a corner.
3. Scale each match from its own pitch size to the challenge pitch size.
4. Put the two halves end to end, with a gap of 750 frames that have the label `X`.
5. Convert the DFL labels: `BallStatus` dead gives `D`; live frames get `H` or `A` from `BallPossession`.

Then measure the statistics of the challenge tracking. These steps use the tracking file and the Smart Tagging file only. They do not use labels.

6. Measure the speed-bin profile: the ball gaps by player-speed bin, and the track runs of the player slots.
7. Divide the challenge frames into live frames and stoppages. Use the cue rules, the forced layers and the stage-1 decode (`stoppage_state`).
8. Measure the game-state profile: the ball gaps and ball speeds in live frames and in stoppages.

Then make two degraded copies of each DFL match.

9. For the team model, apply `degrade_by_speed_bin`. It adds ball gaps by player-speed bin.
10. For the dead/alive head, replace the frozen DFL stoppage ball with a synthetic handled-ball path (`stoppage_ball`).
11. Then apply `degrade_by_game_state`. It adds ball gaps by game state and near the lines.
12. In both copies, smooth the ball, shift it forward, and remove short visible runs.
13. In both copies, break the player tracks into runs like those of the challenge tracking.
14. Calculate the team features and the dead-ball cues of each degraded match.

### 3.4 Train stage 1: the dead/alive head

1. Keep the graded DFL frames that do not have the label `X`.
2. Keep one frame in 15.
3. Use the 40 columns in `[stage1.head] columns`. They include the cues 1 s and 2 s ahead.
4. Fit one `HistGradientBoostingClassifier` on `BallStatus`, with `random_state = seed`.

### 3.5 Train stage 2: the team model

1. Keep the DFL frames that do not have the label `X`, one frame in 15.
2. Keep only the shape columns that also exist on the challenge match.
3. Add each frame again with the home and away teams exchanged and the label exchanged.
4. Give the classes `H`, `A` and `D` equal total weight.
5. Fit three `HistGradientBoostingClassifier` members, with seeds `seed`, `seed + 1` and `seed + 2`.
6. Write both models and `training_manifest.json` to `artifacts/`.

The team model has no ball columns.

### 3.6 Inference, stage 1: stoppage or live

1. Calculate P(dead) of the head on the challenge match.
2. Calculate P(dead) of the cue rules.
3. Mix them: P(dead) = 0.75 × head + 0.25 × cue rules.
4. Set P(home | live) to 0.5. Stage 1 does not select a team.
5. Give the hard-dead frames a near-certain `D` emission.
6. Give the certainly live frames a near-zero `D` emission.
7. Calibrate the `D` bias to a stoppage share of 44 %. This share is close to the DFL training data.
8. Calibrate the switch penalty to an upper bound of 340 live blocks. The penalty must not join stoppages.
9. Decode each half with the decoder.
10. Set the hard-dead frames to `D`.
11. Change live runs shorter than 3 s between two stoppages to `D`.
12. Set all frames outside play to `D`.
13. Extend each possession by 1 s into each adjacent stoppage.
14. Set the observed-dead frames to `D` again.
15. Change live runs of 1 s or less between stoppages to `D`.
16. Apply the keeper rule (L5): a goalkeeper holds an untracked ball, then releases it hard. These frames become live.
17. Change live runs of 1 s or less between stoppages to `D` again.

The forced layers are in `rules/forced.py` and `rules/observed.py`. The hard-dead layer includes the ball out until the restart, goal to kick-off, and Smart Tagging set-piece windows. The certainly live layer includes clear play and the 2 s after a restart strike. It also includes a keeper with the ball in his box.

### 3.7 Inference, stage 2: home or away

1. Find the live blocks of stage 1 in each half.
2. Calculate P(home | live) = P(H) / (P(H) + P(A)) from the team model.
3. Keep the stage-1 stoppages as hard-dead frames.
4. Set the switch penalty from the change cost: 1 s × 25 frames × ln 9 / 2.
5. Decode each half with the decoder. Do not calibrate.
6. Apply the same forced layers and the same sliver rule as in stage 1.

### 3.8 Write and check the output

1. Write `outputs/submission.csv` with LF line endings.
2. Write the stored model outputs to `outputs/model_outputs.npz`.
3. Run `evaluation/validate_submission.py` on the CSV.
4. Calculate the SHA-256 of the CSV.
5. Compare it with `SUBMITTED_SHA256` in `scripts/reproduce.py`. The script stops with exit code 1 if they are different.

## 4. Fixed parameters

All parameters are in `configs/default.toml`. The rule thresholds are next to each rule in `lib/possession/rules/` and `features/deadball.py`. Each value comes from the Laws of the Game, from physical scales, or from generic football priors. Examples are pitch distances, walk and sprint speeds, and the minimum length of a stoppage. The cue-rule weights are approximate log-likelihood ratios. The training data is public.

These values are reasonable defaults. Other matches with the same camera and tracking system, and with official ground truth, can make them more accurate by cross-validation. To do this, change the values in `configs/default.toml`. The code does not change.

## 5. What the pipeline does not use

- The pipeline does not read labels of the challenge match. No code in `lib/`, `train/`, `inference/` or `scripts/` reads `evaluation/gt/`.
- The pipeline does not use the video, the homography or the broadcast feed.
- The pipeline does not use the Smart Tagging team phases or the DFL events files.
- The pipeline does not use manual adjustment. The decoder calibration uses only generic priors.

## 6. Reproducibility

- Seeds: all random draws come from `runtime.seed` (20271117). The synthetic stoppage ball uses `seed + 1`.
- Threads: `threadpoolctl` sets the BLAS and OpenMP threads to `runtime.threads` (4).
- Stored outputs: the decoders read the model outputs after a float32 round trip.
- Dependencies: Python 3.14.3; exact versions are in `requirements.txt`.
- Caches: keys include the source code, the inputs and the parameters. You can delete `cache/` safely.
- Expected SHA-256 of `outputs/submission.csv`: `46c69c3e559135a0e716eb316b67f13c953f1fc18378b3bace57fda242c21716`.
