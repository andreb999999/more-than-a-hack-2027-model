# Parameters

Every value in `configs/default.toml`, what it does and why it has that value. The constants that define
the dead-ball rules themselves (metres, m/s, seconds) are physical scales or Laws-of-the-Game distances,
documented next to each rule in `lib/possession/rules/` and `features/deadball.py` (§8).

## 1. Runtime and data

| parameter | value | meaning |
|---|---|---|
| `runtime.seed` | 20271117 | seed of every random draw (degradation, gradient boosting); any fixed integer |
| `runtime.threads` | 4 | OpenMP threads, fixed for bit-exact gradient boosting |
| `dfl.matches` | all 7 | the DFL matches used for training |
| `dfl.row_stride` | 15 | one labelled frame in 15 is kept for training: neighbouring frames are near-copies |

## 2. Decoder (`[viterbi]`, both stages)

| parameter | value | meaning |
|---|---|---|
| `min_duration` | 12 / 12 / 100 frames | minimum length of an H, A, D run: possessions >= 0.5 s; stoppages >= 4 s, because a short false D splits a possession |
| `switch` | 1 off the diagonal | relative cost of leaving a label; its scale is `penalty_scale` |
| `prior` | 1/3 each | class prior of the scaled-likelihood emissions |
| `floor` | 1e-4 | probability floor before taking logs |
| `penalty_scale` | 8.0 | placeholder: calibrated in stage 1, set from the change cost in stage 2 |
| `dead_bias` | 0.0 | start of stage 1's calibration; stage 2 keeps 0 |
| `max_sandwiched_alive` | 75 frames | an alive run shorter than 3 s between two stoppages is a false restart and becomes D |

## 3. Stage 1 (`[stage1]`)

| parameter | value | meaning |
|---|---|---|
| `head_weight` | 0.75 | P(dead) = 0.75 x DFL head + 0.25 x cue rules: the learned head carries most of the weight; the rules add the Smart Tagging cues, which DFL lacks, as a transparent prior |
| `overshoot_frames` | 25 | possessions are extended 1 s into each neighbouring stoppage: under temporal IoU an undershoot of x frames costs x/L, an overshoot only x/(L+x) |
| `sliver_frames` | 26 | alive slivers of at most 1 s between observed D and a stoppage become D |
| `keeper_hold_sliver_frames` | 26 | the same, after the goalkeeper-holds-the-ball rule (L5) |
| `cues.flight_veto`, `cues.inward_throw` | false, false | the cue features use the ball-out rules without R2a (airborne-ball veto) and R4e (throw seen in mid-air); the forced and observed layers use both. R2a and R4e correct artefacts of camera-based tracking (an airborne ball projected beyond a line, a throw first seen in flight) that the DFL tracking does not have; the cue features are computed in the same way on DFL and on the challenge match, so they leave them out |

### Calibration (`[stage1.calibration]`)

| parameter | value | meaning |
|---|---|---|
| `target_live_stretches` | 340 | target number of live stretches between stoppages. Stage 1 decodes with a neutral team split, so its H / A runs are these stretches, not possessions. 340 is about a typical match's possession count; every live stretch holds at least one possession, so it is an upper bound: when the decoder cannot reach it, the bisection settles at the lowest switch penalty, and the number of stoppages is set by the minimum durations and the forced layers, not by the penalty |
| `target_dead_share` | 0.44 | prior of the D bias: share of dead-ball time in match play, close to the DFL training data (`head_dead_share` in `artifacts/training_manifest.json`: 0.45). It is calibrated on the soft decode, before the forced layers and the 1 s overshoot |
| `rounds` | 2 | the two bisections interact weakly; two rounds converge |
| `penalty_range`, `penalty_iterations`, `live_stretch_tolerance` | [0.01, 400], 24, 3 | geometric bisection of the switch penalty, stopping within 3 live stretches |
| `dead_bias_range`, `dead_bias_iterations`, `dead_share_tolerance` | [-8, 8] nats, 16, 0.005 | linear bisection of the D bias |

### Dead/alive head (`[stage1.head]`)

| parameter | value | meaning |
|---|---|---|
| `max_iter`, `learning_rate`, `max_leaf_nodes`, `l2_regularization` | 300, 0.05, 31, 1.0 | gradient-boosting settings (no early stopping) |
| `row_stride` | 15 | as `dfl.row_stride` |
| `lookahead_s` | 1, 2 | the cues are also read 1 s and 2 s ahead: the scene after a whistle shows that it was one |
| `columns` | 40 | the head's input columns: 10 feature groups, each a cue with its look-ahead copies or a shape column with its context. The groups cover the signs of a stoppage that exist in both data sets: collective walking and speed, the ball at rest, out or carried, a throw-in formation, and each team's velocity towards goal |

### Cue rules (`[stage1.cue_rules]`)

A logistic of 18 cue terms (`features/deadball.py: rule_terms`). `bias` (-1.5) sets the rules' base rate before
they are mixed with the head; the decoder's D bias then calibrates the overall dead share. Each weight (nats) is roughly the log of the cue's
likelihood ratio, judged from the Laws of the Game and from how rare the configuration is in open play:

| cue | weight | cue | weight |
|---|---|---|---|
| `slow` (everyone walking 3 s) | 3.0 | `ball_placed` (still ball, players walking) | 1.0 |
| `fast` (somebody sprinting) | -2.5 | `sp_open` (inside a set-piece window) | 2.0 |
| `ball_out` (out / vanished / at rest / player outside) | 8.0 | `goal_pending` (goal to kick-off) | 3.0 |
| `since_out` (held until the restart) | 4.0 | `restart_passed` (restart in the last 5 s) | -1.5 |
| `kickoff` formation | 3.0 | `ball_moving` (> 4 m/s) | -1.0 |
| `corner` formation | 2.0 | `lost_slow` (ball untracked >= 3 s, everyone walking) | 2.0 |
| `wall` formation | 2.0 | `keeper_holds` (keeper in his box with the ball) | -2.0 |
| `goal_kick` formation | 1.0 | `player_carries` (ball within 1 m of a player) | -1.5 |
| `throw_in` formation | 1.5 | `walk` (everyone walking >= 3 s) | 1.0 |

## 4. Team model (`[team_model]`)

| parameter | value | meaning |
|---|---|---|
| `max_iter`, `learning_rate`, `max_leaf_nodes`, `l2_regularization` | 300, 0.05, 31, 1.0 | gradient-boosting settings (no early stopping) |
| `n_members` | 3 | estimators averaged (seeds `seed`, `seed + 1`, `seed + 2`) |
| `swap_augment` | true | every training frame is added again with home and away exchanged: the model learns geometry, not "the home side usually has the ball" |
| `balance` | true | the total weight of H, A and D is equalised, as the scaled-likelihood decoder expects |

## 5. Stage 2 (`[stage2]`)

| parameter | value | meaning |
|---|---|---|
| `change_cost_seconds` | 1.0 | a change of team must be paid for by 1 s of clear evidence for the other team ... |
| `clear_evidence_odds` | 9 | ... "clear" meaning 0.9 / 0.1 per frame: penalty = 1 x 25 x ln 9 / 2 per change |
| `overshoot_frames` | 0 | stage 1 already extended the possessions |
| `sliver_frames` | 26 | as in stage 1 |

## 6. Fixed internals

Values that are part of a component's definition rather than settings: the forced-D emission
(0.001 / 0.001 / 0.998) and the certainly-alive masses (0.999 / 0.001) of `decoding/layered.py`; the
probability clips (0.001 / 0.999) and stage 2's D emission of `pipeline/stage2.py`; the sensor constants of
`SpeedBinProfile` (ball smoothing 6 frames, ball lead 4 frames, minimum visible run 15 frames, speed bins).

## 6a. Choice of the hand-set values

The hand-set values (the rule thresholds, the cue-rule weights, `head_weight`, the head's columns, the
decoder's minimum durations and priors, the sensor constants above) are reasonable defaults from the
Laws of the Game, physical scales and the public training data. Given other matches recorded with the same camera and
tracking system, with official ground truth, they could be chosen more accurately by cross-validation; the configuration file makes that a change of values, not of code.

## 7. Feature naming

Team features: `h_` home, `a_` away, `s_` home minus away, `xh_` / `yh_` coordinates in the home attacking
frame. Context suffixes, windows in frames (25 = 1 s): `__m25` / `__m75` / `__m250` centred means,
`__delta` next 1 s minus last 1 s, `__past62` / `__fut62` last / next 2.5 s, `__d62` their difference.
Dead-ball cues start with `db_`, and their look-ahead copies end in `__ahead1s` / `__ahead2s`.

## 8. Rule constants

In metres, m/s or seconds, next to each rule (`rules/observed.py`, `rules/forced.py`, `rules/formations.py`,
`features/deadball.py`): Laws-of-the-Game distances (9.15 m, penalty and goal areas), human speeds
(walking < 2 m/s, sprinting > 4.5 m/s, slot switches > 12 m/s), the speeds of a struck or placed ball, and
minimum durations.
