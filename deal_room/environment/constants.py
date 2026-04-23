"""Shared environment constants for DealRoom v3."""

REWARD_WEIGHTS = {
    "goal": 0.25,
    "trust": 0.20,
    "info": 0.20,
    "risk": 0.20,
    "causal": 0.15,
}

TERMINAL_REWARDS = {
    "deal_closed": 1.0,
    "veto": -1.0,
    "max_rounds": -0.5,
    "stage_regression": -0.5,
    "impasse": -0.75,
}

DEFAULT_MAX_ROUNDS = 10
SUMMARY_TIMEOUT_SECONDS = 5.0
VETO_WARNING_THRESHOLD_RATIO = 0.70
