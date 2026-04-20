"""
Pareto efficiency checker for DealRoom v3 - terminal reward determination.
"""

from typing import Dict, List, Tuple


TERMINAL_REWARDS = {
    "deal_closed": 1.0,
    "veto": -1.0,
    "max_rounds": 0.0,
    "stage_regression": -0.5,
    "impasse": -0.75,
}


def check_pareto_optimality(
    all_utilities: Dict[str, float],
    cvar_losses: Dict[str, float],
    thresholds: Dict[str, float],
) -> bool:
    for stakeholder_id, utility in all_utilities.items():
        cvar = cvar_losses.get(stakeholder_id, 0.0)
        threshold = thresholds.get(stakeholder_id, 0.4)
        if cvar > threshold:
            return False

    pareto_frontier = []
    for sid, util in all_utilities.items():
        dominated = False
        for other_sid, other_util in all_utilities.items():
            if other_sid == sid:
                continue
            if other_util >= util:
                cvar_sid = cvar_losses.get(sid, 0.0)
                cvar_other = cvar_losses.get(other_sid, 0.0)
                if cvar_other <= cvar_sid:
                    dominated = True
                    break
        if not dominated:
            pareto_frontier.append(sid)

    return len(pareto_frontier) > 0


def compute_terminal_reward(
    deal_closed: bool,
    veto_triggered: bool,
    veto_stakeholder: str,
    max_rounds_reached: bool,
    stage_regressions: int,
    all_utilities: Dict[str, float],
    cvar_losses: Dict[str, float],
    thresholds: Dict[str, float],
) -> Tuple[float, str]:
    if deal_closed:
        return TERMINAL_REWARDS["deal_closed"], "deal_closed"

    if veto_triggered:
        return TERMINAL_REWARDS["veto"], f"veto_by_{veto_stakeholder}"

    if max_rounds_reached:
        is_pareto = check_pareto_optimality(all_utilities, cvar_losses, thresholds)
        if is_pareto:
            return 0.0, "max_rounds_pareto"
        return TERMINAL_REWARDS["max_rounds"], "max_rounds_no_deal"

    if stage_regressions > 0:
        penalty = TERMINAL_REWARDS["stage_regression"] * min(stage_regressions, 3)
        return penalty, f"stage_regression_{stage_regressions}"

    return TERMINAL_REWARDS["impasse"], "impasse"


def get_pareto_frontier_stakeholders(
    all_utilities: Dict[str, float], cvar_losses: Dict[str, float]
) -> List[str]:
    frontier = []
    for sid, util in all_utilities.items():
        dominated = False
        for other_sid, other_util in all_utilities.items():
            if other_sid == sid:
                continue
            if other_util >= util:
                cvar_sid = cvar_losses.get(sid, 0.0)
                cvar_other = cvar_losses.get(other_sid, 0.0)
                if cvar_other <= cvar_sid:
                    dominated = True
                    break
        if not dominated:
            frontier.append(sid)
    return frontier
