# Data: what the model reads, where it comes from, how to get it

| # | data | used for | in this repository? | how to get it |
|---|---|---|---|---|
| 1 | Organisers' match bundle, tactical source: ATD tracking + metadata, Smart Tagging | the match we label (unlabelled input) | **no** (not redistributable) | the organisers' download (see [challenge/DATA.md](challenge/DATA.md)) |
| 2 | DFL open data, 7 Bundesliga matches | training data of both learned models | no (2.6 GB) | `python scripts/download_dfl.py` |
| 3 | `evaluation/gt/BAR-ATH_possession_gt_public.csv` | not read by the pipeline | yes (the organisers ship it) | — |

Nothing else is read. In particular: no hand labels of any kind, no event data of the match, no
corrected tracking, and neither the video, the homography nor the broadcast feed.

## 1. The organisers' match bundle

Three files of the bundle, unpacked with their original names into `data/` (or point `MTAH_DATA_DIR`
at the folder that contains `tactical/`, as in the official repository):

```
data/tactical/BAR-ATH_tactical_atd.xml              EPTS metadata: teams, goalkeepers, attack direction
data/tactical/BAR-ATH_tactical_atd.txt              EPTS tracking, players and ball, 25 fps
data/tactical/BAR-ATH_tactical_smart_tagging.xml    Smart Tagging: set pieces, goals (team phases not used)
```

SHA-256 of the files the submission was produced from (`python scripts/check_data.py` verifies them):

| file | bytes | sha256 |
|---|---|---|
| `BAR-ATH_tactical_atd.txt` | 195,714,285 | `f130e350d2bfdf734b7e32f072931fccc6d015c1e837d082064875fe0a700431` |
| `BAR-ATH_tactical_atd.xml` | 59,057 | `327e35e32e46a647389b88f883d4f9a2ab931e3e6148aef2fcf8f22b1e93c43a` |
| `BAR-ATH_tactical_smart_tagging.xml` | 261,055 | `0594d35e3df01b9c5002372d8615e2e8ef5379ac8f4e84c5ef7a53f4304c9019` |

The data is gitignored (`data/*`) and must never be committed: the submission repository is public.

## 2. DFL open data (training data)

Bassek, M., Rein, R., Weber, H., Memmert, D. (2025). *An integrated dataset of spatiotemporal and event
data in elite soccer.* Scientific Data. figshare, doi:[10.6084/m9.figshare.28196177](https://doi.org/10.6084/m9.figshare.28196177).
Licence: **CC BY 4.0** (https://creativecommons.org/licenses/by/4.0/).

Seven matches of the German Bundesliga / 2. Bundesliga, season 2022/23 (DFL match ids J03WMX, J03WN1,
J03WOH, J03WOY, J03WPY, J03WQQ, J03WR9). Per match the pipeline reads two files:

- `DFL_02_01_matchinformation_*.xml`: teams, players, playing positions (goalkeepers), pitch size;
- `DFL_04_03_positions_raw_observed_*.xml`: 25 Hz optical tracking of every player and the ball. Every
  ball frame carries `BallPossession` (1 home, 2 away) and `BallStatus` (1 alive, 0 dead).

**Labels**: `BallStatus` and `BallPossession` as delivered by the data provider (the DFL's official
tracking/data provider), i.e. not labelled by us. They are the same quantity as the challenge's ground
truth: per frame, alive with home / away in possession, or dead. Amount: ~13 hours of labelled match
play (7 matches x ~100 min, ~1.0 M labelled frames); training keeps one labelled frame in 15.

`python scripts/download_dfl.py` downloads the article's files (~2.6 GB) into `data/external/dfl/`
(`DFL_DATA_DIR` overrides), checking each file's size and MD5 against figshare. The events files are
part of the article but are not read.

## 3. The public 5 minutes

`evaluation/gt/BAR-ATH_possession_gt_public.csv` ships with the organisers' repository. The pipeline never
reads it.
