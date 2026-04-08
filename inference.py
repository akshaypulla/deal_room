"""
DealRoom Inference Script — Baseline
Imports deal_room directly (no HTTP). Strict [START][STEP][END] format.
"""

import os
import json
import re

from openai import OpenAI
from models import DealRoomAction
from server.deal_room_environment import DealRoomEnvironment

API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY", "")
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
BENCHMARK = "deal-room"

client = OpenAI(api_key=API_KEY, base_url=API_BASE_URL)

SYSTEM_PROMPT = """You are an enterprise software sales negotiator closing a $2M+ contract.
You must build consensus across 5 stakeholders: CFO, CTO, Legal, Procurement, Ops.

Respond ONLY with a JSON object:
{
  "action_type": "direct_message",
  "target": "CFO",
  "message": "Your message here",
  "documents": [{"type": "roi_model", "specificity": "high"}],
  "channel": "formal"
}

action_type: direct_message|send_document|backchannel|group_proposal|concession|reframe_value_prop|exec_escalation
target: CFO|CTO|Legal|Procurement|Ops|all|cto_cfo|legal_procurement
document types: roi_model|security_cert|implementation_timeline|dpa|reference_case
specificity: high|med|low

Watch veto_precursors carefully — act on them immediately with backchannel.
Watch momentum_direction in info: 0 means stalling, act before it becomes -1.
Build consensus systematically. Do not rely on one champion."""


def get_action(obs_dict: dict) -> dict:
    content = (
        f"Round {obs_dict['round_number']}/{obs_dict['max_rounds']} | "
        f"Stage: {obs_dict['deal_stage']} | Momentum: {obs_dict['deal_momentum']}\n"
        f"Blockers: {obs_dict.get('active_blockers', [])} | "
        f"Days left: {obs_dict.get('days_to_deadline', '?')}\n\n"
        f"Stakeholder messages:\n{json.dumps(obs_dict.get('stakeholder_messages', {}), indent=2)}\n\n"
        f"Engagement levels (delayed, noisy):\n{json.dumps(obs_dict.get('engagement_level', {}), indent=2)}\n\n"
        f"Veto precursors (act on these immediately):\n{json.dumps(obs_dict.get('veto_precursors', {}), indent=2)}\n"
        f"Competitor events: {obs_dict.get('competitor_events', [])}\n"
    )
    if obs_dict.get("scenario_hint"):
        content += f"\nSCENARIO HINT: {obs_dict['scenario_hint']}\n"
    content += "\nRespond with your JSON action:"

    resp = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": content},
        ],
        max_tokens=400,
        temperature=0.3,
    )
    raw = resp.choices[0].message.content.strip()
    for pattern in [r"```json\s*(.*?)\s*```", r"```\s*(.*?)\s*```", r"(\{.*\})"]:
        m = re.search(pattern, raw, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(1))
            except json.JSONDecodeError:
                pass
    return {
        "action_type": "direct_message",
        "target": "all",
        "message": raw[:200],
        "channel": "formal",
    }


def run_task(task_id: str, seed: int = 42) -> dict:
    env = DealRoomEnvironment()
    obs = env.reset(seed=seed, task_id=task_id)
    short_model = MODEL_NAME.split("/")[-1] if "/" in MODEL_NAME else MODEL_NAME
    print(f"[START] task={task_id} env={BENCHMARK} model={short_model}")

    rewards, step_num, final_score, success = [], 0, 0.0, False
    try:
        while not obs.done and step_num < obs.max_rounds + 2:
            step_num += 1
            ad = get_action(obs.dict())
            action = DealRoomAction(
                action_type=ad.get("action_type", "direct_message"),
                target=ad.get("target", "all"),
                message=ad.get("message", ""),
                documents=ad.get("documents", []),
                channel=ad.get("channel", "formal"),
                mode=ad.get("mode", "async_email"),
            )
            obs, reward, done, info = env.step(action)
            rewards.append(reward)
            err = info.get("error", None)
            print(
                f"[STEP] step={step_num} "
                f"action={action.action_type}(target={action.target}) "
                f"reward={reward:.2f} done={str(done).lower()} "
                f"error={err if err else 'null'}"
            )
            if done:
                final_score = reward
                success = reward > 0.05
                break
    except Exception as e:
        print(
            f"[STEP] step={step_num} action=error reward=0.00 done=true error={str(e)[:80]}"
        )
        rewards.append(0.0)

    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={step_num} "
        f"score={final_score:.2f} rewards={rewards_str}"
    )
    return {
        "task": task_id,
        "score": final_score,
        "steps": step_num,
        "success": success,
    }


if __name__ == "__main__":
    for task in ["aligned", "conflicted", "hostile_acquisition"]:
        run_task(task, seed=42)
