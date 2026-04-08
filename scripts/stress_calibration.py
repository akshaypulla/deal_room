#!/usr/bin/env python3
"""Stress, determinism, and calibration checks for DealRoom V2.5."""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import statistics
import sys
import time
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import inference as baseline
from server.deal_room_environment import DealRoomEnvironment


TASKS = ("aligned", "conflicted", "hostile_acquisition")


baseline.API_KEY = None


@dataclass
class EpisodeResult:
    task: str
    seed: int
    score: float
    steps: int
    success: bool
    done: bool
    stage: str
    rewards: List[float]
    blockers: List[str]
    known_constraints: List[str]
    approval_bands: Dict[str, str]
    semantic_backend: str
    error: str | None
    elapsed_s: float

    @property
    def fingerprint(self) -> str:
        payload = {
            "task": self.task,
            "seed": self.seed,
            "score": round(self.score, 4),
            "steps": self.steps,
            "success": self.success,
            "done": self.done,
            "stage": self.stage,
            "rewards": [round(value, 4) for value in self.rewards],
            "blockers": self.blockers,
            "known_constraints": self.known_constraints,
            "approval_bands": self.approval_bands,
            "semantic_backend": self.semantic_backend,
            "error": self.error,
        }
        encoded = json.dumps(payload, sort_keys=True).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()[:16]


def run_episode(task: str, seed: int, env: DealRoomEnvironment | None = None) -> EpisodeResult:
    local_env = env or DealRoomEnvironment()
    obs = local_env.reset(seed=seed, task_id=task)
    policy = baseline.ProtocolPolicy()
    rewards: List[float] = []
    done = False
    final_score = 0.0
    error: str | None = None
    start = time.perf_counter()

    try:
        while not obs.done and len(rewards) < obs.max_rounds + 2:
            action = policy.build_action(obs)
            obs, reward, done, info = local_env.step(action)
            rewards.append(float(reward))
            if done:
                final_score = float(reward)
                break
    except Exception as exc:  # pragma: no cover - diagnostic path
        error = str(exc)
        info = {"approval_bands": {}, "semantic_backend": "error"}
        final_score = 0.0
        done = True

    elapsed_s = time.perf_counter() - start
    info = obs.info if hasattr(obs, "info") else {}
    approval_bands = dict(info.get("approval_bands", {}))
    semantic_backend = str(info.get("semantic_backend", "unknown"))
    known_constraints = [item.get("id", "") for item in obs.known_constraints]

    return EpisodeResult(
        task=task,
        seed=seed,
        score=final_score,
        steps=len(rewards),
        success=final_score >= 0.35,
        done=done or obs.done,
        stage=obs.deal_stage,
        rewards=rewards,
        blockers=list(obs.active_blockers),
        known_constraints=known_constraints,
        approval_bands=approval_bands,
        semantic_backend=semantic_backend,
        error=error,
        elapsed_s=elapsed_s,
    )


def summarize(task: str, results: List[EpisodeResult]) -> Dict[str, Any]:
    scores = [result.score for result in results]
    steps = [result.steps for result in results]
    successes = sum(1 for result in results if result.success)
    failures = [result for result in results if result.error]
    semantic_backends = sorted({result.semantic_backend for result in results})
    return {
        "task": task,
        "episodes": len(results),
        "mean_score": round(statistics.fmean(scores), 4),
        "median_score": round(statistics.median(scores), 4),
        "min_score": round(min(scores), 4),
        "max_score": round(max(scores), 4),
        "stdev_score": round(statistics.pstdev(scores), 4),
        "success_rate": round(successes / len(results), 4),
        "mean_steps": round(statistics.fmean(steps), 2),
        "max_steps": max(steps),
        "semantic_backends": semantic_backends,
        "errors": len(failures),
    }


def run_distribution(tasks: Iterable[str], seeds: Iterable[int]) -> Dict[str, List[EpisodeResult]]:
    by_task: Dict[str, List[EpisodeResult]] = {task: [] for task in tasks}
    for task in tasks:
        for seed in seeds:
            by_task[task].append(run_episode(task, seed))
    return by_task


def run_determinism(tasks: Iterable[str], seeds: Iterable[int], repeats: int) -> List[Dict[str, Any]]:
    checks: List[Dict[str, Any]] = []
    for task in tasks:
        for seed in seeds:
            runs = [run_episode(task, seed) for _ in range(repeats)]
            fingerprints = {result.fingerprint for result in runs}
            checks.append(
                {
                    "task": task,
                    "seed": seed,
                    "repeats": repeats,
                    "stable": len(fingerprints) == 1,
                    "fingerprints": sorted(fingerprints),
                    "score": round(runs[0].score, 4),
                    "steps": runs[0].steps,
                }
            )
    return checks


def run_reuse_env_stress(tasks: Iterable[str], episodes: int, seed_start: int) -> Dict[str, Any]:
    env = DealRoomEnvironment()
    results: List[EpisodeResult] = []
    reset_ok = True
    start = time.perf_counter()

    for index in range(episodes):
        task = tuple(tasks)[index % len(tuple(tasks))]
        seed = seed_start + index
        result = run_episode(task, seed, env=env)
        results.append(result)
        reset_ok = reset_ok and env.state.task_id == task and env.state.round_number <= env.state.max_rounds

    elapsed = time.perf_counter() - start
    return {
        "episodes": episodes,
        "errors": sum(1 for result in results if result.error),
        "mean_score": round(statistics.fmean(result.score for result in results), 4),
        "mean_steps": round(statistics.fmean(result.steps for result in results), 2),
        "mean_episode_s": round(elapsed / max(episodes, 1), 4),
        "max_episode_s": round(max(result.elapsed_s for result in results), 4),
        "reset_integrity_ok": reset_ok,
    }


def difficulty_ladder(summary_rows: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    aligned = summary_rows["aligned"]["mean_score"]
    conflicted = summary_rows["conflicted"]["mean_score"]
    hostile = summary_rows["hostile_acquisition"]["mean_score"]
    return {
        "aligned_gt_conflicted": aligned > conflicted,
        "conflicted_gt_hostile": conflicted > hostile,
        "target_shape_ok": aligned >= 0.75 and conflicted >= 0.4 and hostile <= conflicted,
    }


def print_report(
    summaries: Dict[str, Dict[str, Any]],
    determinism: List[Dict[str, Any]],
    reuse_stress: Dict[str, Any],
) -> None:
    print("== DealRoom Stress + Calibration Report ==")
    print("\nTask distribution:")
    for task in TASKS:
        row = summaries[task]
        print(
            f"- {task}: mean={row['mean_score']:.4f} median={row['median_score']:.4f} "
            f"min={row['min_score']:.4f} max={row['max_score']:.4f} "
            f"stdev={row['stdev_score']:.4f} success_rate={row['success_rate']:.2%} "
            f"mean_steps={row['mean_steps']:.2f} backends={','.join(row['semantic_backends'])}"
        )

    print("\nDeterminism:")
    for row in determinism:
        status = "stable" if row["stable"] else "UNSTABLE"
        print(
            f"- {row['task']} seed={row['seed']}: {status} "
            f"score={row['score']:.4f} steps={row['steps']} fingerprints={','.join(row['fingerprints'])}"
        )

    print("\nRepeated reset stress:")
    print(
        f"- episodes={reuse_stress['episodes']} errors={reuse_stress['errors']} "
        f"mean_score={reuse_stress['mean_score']:.4f} mean_steps={reuse_stress['mean_steps']:.2f} "
        f"mean_episode_s={reuse_stress['mean_episode_s']:.4f} max_episode_s={reuse_stress['max_episode_s']:.4f} "
        f"reset_integrity_ok={reuse_stress['reset_integrity_ok']}"
    )

    ladder = difficulty_ladder(summaries)
    print("\nDifficulty ladder:")
    for key, value in ladder.items():
        print(f"- {key}={value}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, default=8, help="Number of sequential seeds per task.")
    parser.add_argument(
        "--seed-start",
        type=int,
        default=42,
        help="Starting seed for distribution and reuse checks.",
    )
    parser.add_argument(
        "--determinism-repeats",
        type=int,
        default=3,
        help="How many times to rerun the same seed for determinism checks.",
    )
    parser.add_argument(
        "--determinism-seeds",
        type=int,
        default=3,
        help="How many seeds per task to check for exact reproducibility.",
    )
    parser.add_argument(
        "--stress-episodes",
        type=int,
        default=24,
        help="How many episodes to run through one reused env instance.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit the report as JSON instead of human-readable text.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    seeds = list(range(args.seed_start, args.seed_start + args.seeds))
    determinism_seed_values = list(
        range(args.seed_start, args.seed_start + args.determinism_seeds)
    )

    by_task = run_distribution(TASKS, seeds)
    summaries = {task: summarize(task, results) for task, results in by_task.items()}
    determinism = run_determinism(TASKS, determinism_seed_values, args.determinism_repeats)
    reuse_stress = run_reuse_env_stress(TASKS, args.stress_episodes, args.seed_start + 1000)

    report = {
        "summaries": summaries,
        "determinism": determinism,
        "reuse_stress": reuse_stress,
        "difficulty_ladder": difficulty_ladder(summaries),
    }

    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
        return

    print_report(summaries, determinism, reuse_stress)


if __name__ == "__main__":
    main()
