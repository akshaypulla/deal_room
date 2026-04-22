"""
DealRoom v3.6 — Standalone LLM Inference Demo

This script demonstrates the fine-tuned DealRoom negotiation agent
without requiring the full Gradio UI.

Usage:
    # Train first (from Colab or locally):
    #   python notebooks/05_llm_training.ipynb  # exports to dealroom-qwen-3b-negotiation/

    # Then run inference:
    python demo_inference.py --scenario hostile_acquisition --seed 7 --max-steps 8
    python demo_inference.py --scenario aligned --seed 42 --max-steps 10
    python demo_inference.py --scenario conflicted --seed 99 --max-steps 10

Optional: Load a model from HuggingFace Hub:
    python demo_inference.py --model-path username/dealroom-qwen-3b --scenario hostile_acquisition

The fine-tuned model learns to:
1. Prefer send_document over direct_message (higher immediate reward)
2. Target Finance/Legal first (higher trust/info scores)
3. Avoid triggering vetoes (higher terminal rewards)
"""

import argparse
import sys
import os
from pathlib import Path
from typing import Optional

try:
    import torch
    from unsloth import FastLanguageModel

    UNSLOTH_AVAILABLE = True
except ImportError:
    UNSLOTH_AVAILABLE = False

from deal_room.environment.dealroom_v3 import DealRoomV3
from deal_room.environment.text_env import DealRoomTextEnv
from deal_room.environment.prompts import build_situation_prompt, parse_action_text


# Well-known good seeds for demo stability
SEED_PRESETS = {
    "hostile_acquisition": {
        7: "Known good: agent correctly sequences Finance → Legal → TechLead",
        13: "Known good: agent uses lookahead before group_proposal",
        42: "Reproducible baseline",
        99: "Stress test: multiple blockers",
    },
    "aligned": {
        42: "Smooth negotiation, high Pareto efficiency",
        7: "Quick resolution expected",
        99: "Extended deliberation",
    },
    "conflicted": {
        42: "Cost cluster vs risk cluster conflict",
        13: "Multi-cluster tension",
        99: "High uncertainty scenario",
    },
}


def load_model(model_path: str, max_seq_length: int = 512):
    """Load fine-tuned model or fall back to base model."""
    if not UNSLOTH_AVAILABLE:
        raise RuntimeError("Unsloth not installed. Run: pip install unsloth")

    if not Path(model_path).exists():
        print(f"[INFO] Model path '{model_path}' not found locally.")
        print(f"[INFO] Attempting to load from HuggingFace Hub...")
        try:
            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name=model_path,
                max_seq_length=max_seq_length,
                dtype=torch.float16,
                load_in_4bit=True,
            )
        except Exception as e:
            print(f"[WARN] Could not load from Hub: {e}")
            print("[INFO] Falling back to base Qwen2.5-3B...")
            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name="Qwen/Qwen2.5-3B-Instruct",
                max_seq_length=max_seq_length,
                dtype=torch.float16,
                load_in_4bit=True,
            )
    else:
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_path,
            max_seq_length=max_seq_length,
            dtype=torch.float16,
            load_in_4bit=True,
        )

    FastLanguageModel.for_inference(model)
    return model, tokenizer


def run_episode(
    model,
    tokenizer,
    scenario: str = "hostile_acquisition",
    seed: int = 7,
    max_steps: int = 8,
    temperature: float = 0.7,
) -> dict:
    """Run one episode with the fine-tuned LLM agent."""
    env = DealRoomTextEnv(task_id=scenario, seed=seed)
    prompt = env.reset()

    trajectory = {
        "steps": [],
        "total_reward": 0.0,
        "terminal_outcome": None,
        "pareto_efficiency": 0.0,
    }

    print(f"\n{'=' * 65}")
    print(f"  DealRoom v3.6 — {scenario} (seed={seed})")
    print(f"{'=' * 65}\n")

    for step in range(max_steps):
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        outputs = model.generate(
            **inputs,
            max_new_tokens=120,
            temperature=temperature,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )
        response = tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1] :],
            skip_special_tokens=True,
        ).strip()

        parsed = parse_action_text(response)

        print(f"Step {step + 1}:")
        print(f"  LLM:    {response[:80]}{'...' if len(response) > 80 else ''}")
        print(f"  Action: [{parsed.action_type}] → {parsed.target}")
        if parsed.message:
            print(
                f"  Message: {parsed.message[:60]}{'...' if len(parsed.message) > 60 else ''}"
            )

        prompt, reward, done, info = env.step(response)
        trajectory["total_reward"] += reward
        trajectory["steps"].append(
            {
                "step": step + 1,
                "action_type": parsed.action_type,
                "target": parsed.target,
                "reward": reward,
                "done": done,
            }
        )

        reward_components = info.get("reward_components", {})
        if reward_components:
            rc_str = " | ".join(f"{k}={v:.2f}" for k, v in reward_components.items())
            print(f"  Reward: {reward:+.3f} ({rc_str})")
        else:
            print(f"  Reward: {reward:+.3f}")

        if done:
            trajectory["terminal_outcome"] = info.get("terminal_outcome", "unknown")
            trajectory["pareto_efficiency"] = info.get("pareto_efficiency", 0.0)
            trajectory["causal_accuracy"] = info.get("causal_accuracy", 0.0)
            trajectory["veto_stakeholder"] = info.get("veto_stakeholder", None)
            print(f"\n  → Terminal: {trajectory['terminal_outcome']}")
            print(f"  → Pareto Efficiency: {trajectory['pareto_efficiency']:.3f}")
            break
        print()

    env.close()
    return trajectory


def print_episode_summary(trajectory: dict):
    """Print a formatted episode summary."""
    print(f"\n{'─' * 65}")
    print(f"  Episode Summary")
    print(f"{'─' * 65}")
    print(f"  Total Reward:    {trajectory['total_reward']:+.3f}")
    print(f"  Steps:           {len(trajectory['steps'])}")
    print(f"  Terminal:        {trajectory.get('terminal_outcome', 'max_rounds')}")
    print(f"  Pareto Eff:      {trajectory.get('pareto_efficiency', 0):.3f}")
    print(f"  Causal Acc:      {trajectory.get('causal_accuracy', 0):.3f}")
    if trajectory.get("veto_stakeholder"):
        print(f"  Veto by:         {trajectory['veto_stakeholder']}")
    print(f"{'─' * 65}\n")


def main():
    parser = argparse.ArgumentParser(description="DealRoom v3.6 LLM Inference Demo")
    parser.add_argument(
        "--scenario",
        default="hostile_acquisition",
        choices=["aligned", "conflicted", "hostile_acquisition"],
        help="Scenario to run",
    )
    parser.add_argument("--seed", type=int, default=7, help="Random seed")
    parser.add_argument(
        "--max-steps", type=int, default=8, help="Maximum steps per episode"
    )
    parser.add_argument(
        "--temperature", type=float, default=0.7, help="LLM sampling temperature"
    )
    parser.add_argument(
        "--model-path",
        default="dealroom-qwen-3b-negotiation",
        help="Path to fine-tuned model (local or HF hub)",
    )
    parser.add_argument(
        "--list-seeds",
        action="store_true",
        help="List known-good seeds for each scenario",
    )

    args = parser.parse_args()

    if args.list_seeds:
        print("\nKnown-good seeds for demo stability:")
        for scenario, seeds in SEED_PRESETS.items():
            print(f"\n  {scenario}:")
            for seed, desc in seeds.items():
                print(f"    seed={seed}: {desc}")
        return

    print(f"\n[INFO] Scenario: {args.scenario}, Seed: {args.seed}")
    print(f"[INFO] Max steps: {args.max_steps}, Temperature: {args.temperature}")

    if args.seed in SEED_PRESETS.get(args.scenario, {}):
        print(f"[INFO] Seed note: {SEED_PRESETS[args.scenario][args.seed]}")

    if not UNSLOTH_AVAILABLE:
        print(
            "[ERROR] Unsloth is required for inference. Install with: pip install unsloth"
        )
        print("[INFO] Falling back to rule-based policy...")
        env = DealRoomTextEnv(task_id=args.scenario, seed=args.seed)
        prompt = env.reset()
        print("\n[Rule-based episode with random fallback actions]\n")
        # Simple rule-based fallback
        fallback_actions = [
            "send_document Finance roi_model Here is our ROI model.",
            "direct_message Legal Let me address your compliance concerns.",
        ]
        for i in range(args.max_steps):
            action = fallback_actions[i % len(fallback_actions)]
            prompt, reward, done, info = env.step(action)
            print(f"Step {i + 1}: [{action[:50]}...] reward={reward:.3f} done={done}")
            if done:
                break
        env.close()
        return

    print(f"\n[INFO] Loading model from: {args.model_path}")
    model, tokenizer = load_model(args.model_path)
    print("[INFO] Model loaded successfully.")

    trajectory = run_episode(
        model=model,
        tokenizer=tokenizer,
        scenario=args.scenario,
        seed=args.seed,
        max_steps=args.max_steps,
        temperature=args.temperature,
    )

    print_episode_summary(trajectory)


if __name__ == "__main__":
    main()
