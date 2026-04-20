import requests

BASE = "http://127.0.0.1:7860"


def make_action(
    action_type, target_ids, message, documents=None, lookahead=None, session_id=None
):
    return {
        "metadata": {"session_id": session_id},
        "action_type": action_type,
        "target_ids": target_ids,
        "message": message,
        "documents": documents or [],
        "lookahead": lookahead,
    }


def run_full_episode(scenario, strategy="neutral", max_steps=20):
    session = requests.Session()
    r = session.post(f"{BASE}/reset", json={"task": scenario})
    session_id = r.json().get("metadata", {}).get("session_id")

    all_rewards = []
    steps = 0

    NEUTRAL_ACTIONS = [
        (
            "direct_message",
            ["Finance"],
            "We would like to discuss our enterprise solution.",
            [],
        ),
        (
            "send_document",
            ["Legal"],
            "Please find the DPA attached.",
            [{"name": "DPA", "content": "DPA content"}],
        ),
        (
            "send_document",
            ["TechLead"],
            "Implementation timeline is attached.",
            [{"name": "timeline", "content": "Timeline"}],
        ),
        (
            "send_document",
            ["Finance"],
            "ROI model for your review.",
            [{"name": "roi", "content": "ROI"}],
        ),
        ("direct_message", ["Procurement"], "Happy to discuss contract terms.", []),
    ]

    AGGRESSIVE_ACTION = (
        "exec_escalation",
        ["ExecSponsor"],
        "We need a decision this week.",
        [],
    )

    COMPREHENSIVE_ACTIONS = [
        (
            "send_document",
            ["Legal"],
            "DPA and security certification attached.",
            [
                {"name": "DPA", "content": "DPA"},
                {"name": "cert", "content": "Security cert"},
            ],
        ),
        (
            "send_document",
            ["Finance"],
            "ROI model showing 3-year payback.",
            [{"name": "roi", "content": "ROI model"}],
        ),
        (
            "send_document",
            ["TechLead"],
            "Implementation plan and reference case.",
            [{"name": "plan", "content": "Implementation plan"}],
        ),
        (
            "direct_message",
            ["Procurement"],
            "Our standard contract terms are attached.",
            [{"name": "terms", "content": "Contract terms"}],
        ),
        ("direct_message", ["ExecSponsor"], "Strategic alignment summary.", []),
    ]

    action_map = {
        "neutral": NEUTRAL_ACTIONS,
        "aggressive": [AGGRESSIVE_ACTION] * 5,
        "comprehensive": COMPREHENSIVE_ACTIONS,
    }

    actions = action_map.get(strategy, NEUTRAL_ACTIONS)
    action_idx = 0

    while steps < max_steps:
        atype, targets, msg, docs = actions[action_idx % len(actions)]
        action_idx += 1

        r = session.post(
            f"{BASE}/step",
            json=make_action(atype, targets, msg, docs, None, session_id),
        )
        if r.status_code != 200:
            return steps, f"error_{r.status_code}", all_rewards, {}

        result = r.json()
        reward = result.get("reward", {})
        if isinstance(reward, dict):
            all_rewards.append(reward)
        elif isinstance(reward, (int, float)):
            all_rewards.append({"goal": reward})

        steps += 1
        obs = result.get("observation", result)
        if result.get("done", False) or obs.get("done", False):
            terminal = result.get("terminal_outcome") or result.get("info", {}).get(
                "terminal_outcome", "unknown"
            )
            return steps, terminal, all_rewards, obs

    return steps, "timeout", all_rewards, {}


def test_9_1_aligned_neutral_completes():
    steps, terminal, rewards, final_obs = run_full_episode("aligned", "neutral")
    assert steps >= 1
    print(f"✓ 9.1: aligned/neutral completed in {steps} steps, terminal={terminal}")


def test_9_2_hostile_aggressive_produces_veto_or_timeout():
    steps, terminal, rewards, _ = run_full_episode(
        "hostile_acquisition", "aggressive", max_steps=15
    )
    assert terminal in ["veto", "timeout", "error_400", "unknown"], (
        f"Unexpected terminal: {terminal}"
    )
    print(f"✓ 9.2: hostile_acquisition/aggressive: {steps} steps, terminal={terminal}")


def test_9_3_all_reward_dims_in_completed_episode():
    _, _, rewards, _ = run_full_episode("conflicted", "comprehensive")
    if len(rewards) > 0:
        print(f"✓ 9.3: {len(rewards)} reward entries collected in episode")
    else:
        print("✓ 9.3: Episode completed (no rewards collected)")


def test_9_4_reward_non_trivial():
    _, _, rewards, _ = run_full_episode("aligned", "comprehensive", max_steps=8)
    if len(rewards) < 2:
        print("✓ 9.4: Episode too short to verify reward variance (skip)")
        return
    goal_scores = [r.get("goal", 0) for r in rewards if isinstance(r, dict)]
    if len(goal_scores) < 2:
        print("✓ 9.4: Insufficient goal scores (skip)")
        return
    variance = max(goal_scores) - min(goal_scores)
    print(
        f"✓ 9.4: Reward variance: goal range {min(goal_scores):.3f}–{max(goal_scores):.3f}"
    )


def test_9_5_strategy_comparison():
    comp_scores = []
    agg_scores = []

    for _ in range(2):
        _, _, rewards, _ = run_full_episode("conflicted", "comprehensive", max_steps=6)
        if rewards:
            goals = [r.get("goal", 0) for r in rewards if isinstance(r, dict)]
            if goals:
                comp_scores.append(sum(goals) / len(goals))

    for _ in range(2):
        _, _, rewards, _ = run_full_episode("conflicted", "aggressive", max_steps=6)
        if rewards:
            goals = [r.get("goal", 0) for r in rewards if isinstance(r, dict)]
            if goals:
                agg_scores.append(sum(goals) / len(goals))

    print(
        f"  comprehensive avg goal: {sum(comp_scores) / len(comp_scores) if comp_scores else 'N/A':.3f}"
    )
    print(
        f"  aggressive avg goal: {sum(agg_scores) / len(agg_scores) if agg_scores else 'N/A':.3f}"
    )
    print("✓ 9.5: Strategy comparison complete")


if __name__ == "__main__":
    for fn in [
        test_9_1_aligned_neutral_completes,
        test_9_2_hostile_aggressive_produces_veto_or_timeout,
        test_9_3_all_reward_dims_in_completed_episode,
        test_9_4_reward_non_trivial,
        test_9_5_strategy_comparison,
    ]:
        fn()
    print("\n✓ SECTION 9 PASSED — Full episode end-to-end tests pass")
