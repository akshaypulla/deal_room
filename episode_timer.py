#!/usr/bin/env python3
"""Precise timing of environment interactions across multiple episodes."""

import time
import json
import requests
import statistics

BASE_URL = "http://127.0.0.1:7860"
HEADERS = {"Content-Type": "application/json"}


def reset_session(task, stakeholder_id=0):
    start = time.perf_counter()
    resp = requests.post(
        f"{BASE_URL}/reset",
        headers=HEADERS,
        json={"task": task, "stakeholder_id": stakeholder_id},
    )
    latency = (time.perf_counter() - start) * 1000
    return resp.json(), latency


def step_session(session_id, action, stakeholder, utterance):
    start = time.perf_counter()
    resp = requests.post(
        f"{BASE_URL}/step",
        headers={**HEADERS, "X-Session-ID": session_id},
        json={
            "action": action,
            "parameters": {"stakeholder": stakeholder, "utterance": utterance},
        },
    )
    latency = (time.perf_counter() - start) * 1000
    return resp.json(), latency


def run_episode(task, stakeholder_id=0, stakeholder="Legal", max_steps=15):
    print(f"\n{'=' * 80}")
    print(f"EPISODE: task={task}")
    print(f"{'=' * 80}")

    obs, reset_latency = reset_session(task, stakeholder_id)
    session_id = obs["metadata"]["session_id"]

    print(f"\n[RESET] latency={reset_latency:.3f}ms")
    print(f"  Session: {session_id}")
    print(f"  Stage: {obs['deal_stage']}")
    print(f"  Momentum: {obs['deal_momentum']}")
    print(f"  Days to deadline: {obs['days_to_deadline']}")
    print(f"  Engagement: {[round(v, 3) for v in obs['engagement_level'].values()]}")

    step_times = []
    rewards = []
    rounds = []
    stages = []
    blockers_list = []
    engagements = []
    prev_eng = obs["engagement_level"].copy()

    for step_num in range(1, max_steps + 1):
        action = "propose" if step_num % 3 != 0 else "inquire"
        utterances = [
            "We should proceed with the acquisition at the proposed valuation.",
            "What are the key concerns from your team?",
            "The terms appear favorable for both parties.",
            "Can we schedule a follow-up discussion?",
            "We are committed to moving forward.",
        ]
        utterance = utterances[step_num % len(utterances)]

        step_resp, step_latency = step_session(
            session_id, action, stakeholder, utterance
        )
        step_times.append(step_latency)

        obs = step_resp["observation"]
        reward = step_resp.get("reward")
        rewards.append(reward)

        round_num = obs.get("round_number")
        stage = obs.get("deal_stage")
        done = step_resp.get("done")
        blockers = obs.get("active_blockers", [])
        eng = obs.get("engagement_level", {})
        momentum = obs.get("deal_momentum")

        rounds.append(round_num)
        stages.append(stage)
        blockers_list.append(blockers)
        engagements.append(eng)

        eng_deltas = {k: round(eng.get(k, 0) - prev_eng.get(k, 0), 4) for k in eng}
        prev_eng = eng.copy()

        print(f"\n[STEP {step_num}] latency={step_latency:.3f}ms")
        print(f"  Round: {round_num} | Stage: {stage} | Done: {done}")
        print(f"  Momentum: {momentum}")
        print(f"  Reward: {reward}")
        print(f"  Blockers: {blockers}")
        print(f"  Engagement deltas: {eng_deltas}")

        if done:
            print(f"\n  >> EPISODE TERMINATED at step {step_num}")
            break

    print(f"\n{'─' * 80}")
    print("EPISODE SUMMARY")
    print(f"{'─' * 80}")
    print(f"  Total steps: {len(step_times)}")
    print(f"  Total episode time: {sum(step_times):.2f}ms")
    if len(step_times) > 1:
        print(
            f"  Avg step latency: {statistics.mean(step_times):.3f}ms (stdev={statistics.stdev(step_times):.3f}ms)"
        )
    else:
        print(f"  Avg step latency: {step_times[0]:.3f}ms")
    print(f"  Min/Max step latency: {min(step_times):.3f}ms / {max(step_times):.3f}ms")
    print(f"  Reset latency: {reset_latency:.3f}ms")
    print(f"  Total interaction time: {sum(step_times) + reset_latency:.2f}ms")
    print(f"  Rewards: {[round(r, 4) for r in rewards]}")
    print(f"  Stages: {stages}")
    print(f"  Final blockers: {blockers_list[-1] if blockers_list else []}")

    return {
        "task": task,
        "steps": len(step_times),
        "total_time_ms": sum(step_times) + reset_latency,
        "reset_latency_ms": reset_latency,
        "avg_step_ms": statistics.mean(step_times),
        "stdev_ms": statistics.stdev(step_times) if len(step_times) > 1 else 0,
        "min_step_ms": min(step_times),
        "max_step_ms": max(step_times),
        "rewards": rewards,
        "final_stage": stages[-1] if stages else None,
        "final_blockers": blockers_list[-1] if blockers_list else [],
        "all_stages": stages,
        "all_blockers": blockers_list,
    }


def run_comparison():
    print("\n" + "=" * 80)
    print("MULTI-TASK COMPARISON")
    print("=" * 80)

    tasks = ["aligned", "hostile_acquisition", "conflicted"]
    results = []

    for task in tasks:
        result = run_episode(task)
        results.append(result)

    print(f"\n\n{'=' * 80}")
    print("OVERALL COMPARISON TABLE")
    print(f"{'=' * 80}")
    print(
        f"{'Task':<25} {'Steps':>6} {'Total(ms)':>12} {'Avg(ms)':>10} {'Stdev(ms)':>10} {'Final Stage':>15}"
    )
    print("-" * 80)
    for r in results:
        print(
            f"{r['task']:<25} {r['steps']:>6} {r['total_time_ms']:>12.2f} {r['avg_step_ms']:>10.3f} {r['stdev_ms']:>10.3f} {str(r['final_stage']):>15}"
        )

    print(f"\n{'=' * 80}")
    print("REWARD & STAGE PROFILES BY TASK")
    print(f"{'=' * 80}")
    for r in results:
        print(f"\n  {r['task']}:")
        for i, (rew, stage, blk) in enumerate(
            zip(r["rewards"], r["all_stages"], r["all_blockers"])
        ):
            print(f"    Step {i + 1}: reward={rew:.4f} stage={stage} blockers={blk}")


if __name__ == "__main__":
    print(f"Starting environment interaction test")
    print(f"Base URL: {BASE_URL}")

    health = requests.get(f"{BASE_URL}/health").json()
    print(f"Health: {health}")

    run_comparison()

    print(f"\n{'=' * 80}")
    print("TEST COMPLETE")
    print(f"{'=' * 80}")
