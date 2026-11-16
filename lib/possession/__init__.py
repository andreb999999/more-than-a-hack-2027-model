"""Frame-level ball possession (H / A / D) for the More than a Hack 2027 challenge.

Two stages: stage 1 decides dead ball vs alive on every frame, stage 2 decides home vs away inside the
live blocks. Both learned components are trained on the DFL open data, never on the challenge match.
See docs/MODEL.md.
"""

__version__ = "1.0.0"
