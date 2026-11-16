"""Match constants and label codes.

Every number here is published by the organisers (DATA.md, FAQ.md, `games.py`) or read from the ATD
metadata. None comes from a ground truth. `tests/test_basics.py` checks them against `games.py`.
"""

GAME = "BAR-ATH"
FPS = 25
N_FRAMES = 199_319                       # frames of the tactical video = rows of a submission
PITCH_LENGTH, PITCH_WIDTH = 105.3, 68.0  # metres

# Graded halves of the challenge match as half-open [start, stop) video-frame ranges (DATA.md: 11185..90651 and
# 110808..190197 inclusive).
GRADED_HALVES = ((11185, 90652), (110808, 190198))

# Label codes, identical to evaluation/possession_metrics.py.
H, A, D, X = 0, 1, 2, 3
LABELS = ("H", "A", "D", "X")
N_STATES = 3  # H, A, D

# Team names as they appear in the Smart Tagging `Team` field.
HOME_TAG, AWAY_TAG = "Barcelona", "Atletic Club"
