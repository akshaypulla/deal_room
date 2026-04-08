"""Baseline inference script for DealRoom V2.5."""

from __future__ import annotations

import json
import os
import re
from typing import Dict, List, Optional

from openai import OpenAI

from models import DealRoomAction, DealRoomObservation
from server.deal_room_environment import DealRoomEnvironment

API_KEY = os.getenv("HF_TOKEN") or os.getenv("OPENAI_API_KEY") or os.getenv("API_KEY")
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
BENCHMARK = "deal-room"

client = OpenAI(api_key=API_KEY or "missing", base_url=API_BASE_URL)

ARTIFACT_MESSAGES = {
    "roi_model": "Here is the ROI model with explicit payback assumptions and downside cases.",
    "implementation_timeline": "Here is the implementation timeline with milestones, owners, and delivery guardrails.",
    "security_cert": "Here are the requested security materials, audit artifacts, and control summaries.",
    "dpa": "Here is the DPA with GDPR-aligned privacy commitments and review-ready clauses.",
    "vendor_packet": "Here is the supplier onboarding packet including process, insurance, and vendor details.",
    "reference_case": "Here is a reference case from a comparable deployment with measurable outcomes.",
    "support_plan": "Here is the support plan with named coverage, escalation paths, and ongoing ownership.",
}

ROLE_PROBE_MESSAGES = {
    "finance": "Help me understand the budget ceiling or board payback requirement we need to respect so I can tailor the commercial terms responsibly.",
    "technical": "What delivery window or implementation constraint is truly non-negotiable for your team?",
    "legal_compliance": "Which compliance or privacy obligation is the real approval blocker right now?",
    "procurement": "What supplier-process or onboarding requirement do we still need to satisfy to move this forward cleanly?",
    "operations": "What rollout window or support commitment is the real operational blocker?",
    "executive_sponsor": "What internal approval risk do we need to de-risk before this is safe to sponsor?",
}

SYSTEM_PROMPT = """You are the lead negotiator for an enterprise software vendor.
Return only JSON with keys:
action_type, target_ids, target, message, documents, proposed_terms, channel, mode
Keep the message concise, credible, collaborative, and role-aware."""


class ProtocolPolicy:
    def __init__(self):
        self.handled_precursors: set[str] = set()

    def build_action(self, obs: DealRoomObservation) -> DealRoomAction:
        stakeholders = obs.stakeholders
        progress = obs.approval_path_progress
        requested = obs.requested_artifacts
        known_constraints = {item["id"] for item in obs.known_constraints}
        blockers = obs.active_blockers

        if obs.veto_precursors:
            target_id = next(iter(obs.veto_precursors))
            if target_id not in self.handled_precursors and not requested.get(target_id):
                self.handled_precursors.add(target_id)
                return action_with_message(
                    DealRoomAction(
                        action_type="backchannel",
                        target=target_id,
                        target_ids=[target_id],
                        channel="backchannel",
                        mode="formal_meeting",
                    ),
                    obs,
                    f"Address the rising internal risk with {target_id} directly.",
                )

        for stakeholder_id, payload in progress.items():
            if payload.get("mandatory") and requested.get(stakeholder_id):
                artifact = requested[stakeholder_id][0]
                return action_with_message(
                    DealRoomAction(
                        action_type="send_document",
                        target=stakeholder_id,
                        target_ids=[stakeholder_id],
                        documents=[{"type": artifact, "specificity": "high"}],
                    ),
                    obs,
                    f"Send the requested {artifact.replace('_', ' ')} to {stakeholder_id}.",
                    fallback_message=ARTIFACT_MESSAGES.get(artifact, "Here is the requested material."),
                )

        for stakeholder_id, artifacts in requested.items():
            if artifacts:
                artifact = artifacts[0]
                return action_with_message(
                    DealRoomAction(
                        action_type="send_document",
                        target=stakeholder_id,
                        target_ids=[stakeholder_id],
                        documents=[{"type": artifact, "specificity": "high"}],
                    ),
                    obs,
                    f"Clear the remaining requested artifact for {stakeholder_id}.",
                    fallback_message=ARTIFACT_MESSAGES.get(artifact, "Here is the requested material."),
                )

        if (not known_constraints or any(obs.weak_signals.values())) and obs.deal_stage in {
            "evaluation",
            "negotiation",
            "legal_review",
        }:
            target_id = choose_probe_target(obs)
            prompt = "Probe for the highest-probability hidden constraint using a precise, low-pressure question."
            role = obs.stakeholders[target_id]["role"]
            return action_with_message(
                DealRoomAction(
                    action_type="direct_message",
                    target=target_id,
                    target_ids=[target_id],
                ),
                obs,
                prompt,
                fallback_message=ROLE_PROBE_MESSAGES.get(
                    role,
                    "Help me understand the real approval constraint we need to respect so I can tailor the proposal correctly.",
                ),
            )

        if blockers:
            target_id = blockers[0]
            role = obs.stakeholders[target_id]["role"]
            return action_with_message(
                DealRoomAction(
                    action_type="direct_message",
                    target=target_id,
                    target_ids=[target_id],
                ),
                obs,
                f"Reduce resistance with {target_id} using a specific and credible message.",
                fallback_message=ROLE_PROBE_MESSAGES.get(
                    role,
                    "I want to address the remaining risk directly and make sure the proposal matches your internal constraints.",
                ),
            )

        if obs.deal_stage in {"legal_review", "final_approval", "closed"} and not obs.active_blockers:
            return action_with_message(
                DealRoomAction(
                    action_type="group_proposal",
                    target="all",
                    target_ids=list(stakeholders.keys()),
                    proposed_terms={
                        "price": 180000,
                        "timeline_weeks": 14,
                        "security_commitments": ["gdpr", "audit rights"],
                        "support_level": "named_support_lead",
                        "liability_cap": "mutual_cap",
                    },
                ),
                obs,
                "Attempt closure only because approval and feasibility are ready.",
                fallback_message="I believe we have enough alignment to move to final approval on concrete, reviewable terms.",
            )

        target_id = choose_probe_target(obs)
        return action_with_message(
            DealRoomAction(
                action_type="direct_message",
                target=target_id,
                target_ids=[target_id],
            ),
            obs,
            f"Advance the conversation with {target_id} using a role-aware, specific message.",
            fallback_message="I want to make sure we are solving the right internal concern for your team before we push this forward.",
        )


def build_protocol_action(obs: DealRoomObservation) -> DealRoomAction:
    return ProtocolPolicy().build_action(obs)


def choose_probe_target(obs: DealRoomObservation) -> str:
    weakest = None
    weakest_score = 10.0
    for stakeholder_id, payload in obs.approval_path_progress.items():
        rank = {"blocker": 0, "neutral": 1, "workable": 2, "supporter": 3}[payload["band"]]
        score = rank - (0.2 if payload.get("mandatory") else 0.0)
        if score < weakest_score:
            weakest_score = score
            weakest = stakeholder_id
    return weakest or next(iter(obs.stakeholders))


def action_with_message(
    action: DealRoomAction,
    obs: DealRoomObservation,
    instruction: str,
    fallback_message: Optional[str] = None,
) -> DealRoomAction:
    message = fallback_message or "I want to make this easy to evaluate and safe to approve."
    if API_KEY:
        llm_message = maybe_generate_message(obs, action, instruction)
        if llm_message:
            message = llm_message
    action.message = message
    return action


def maybe_generate_message(
    obs: DealRoomObservation,
    action: DealRoomAction,
    instruction: str,
) -> Optional[str]:
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            temperature=0.1,
            max_tokens=180,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": json.dumps(
                        {
                            "instruction": instruction,
                            "stage": obs.deal_stage,
                            "weak_signals": obs.weak_signals,
                            "known_constraints": obs.known_constraints,
                            "requested_artifacts": obs.requested_artifacts,
                            "action": action.model_dump(),
                        }
                    ),
                },
            ],
        )
    except Exception:
        return None

    raw = (response.choices[0].message.content or "").strip()
    for pattern in [r"```json\s*(.*?)\s*```", r"(\{.*\})"]:
        match = re.search(pattern, raw, re.DOTALL)
        if not match:
            continue
        try:
            payload = json.loads(match.group(1))
            return str(payload.get("message", "")).strip() or None
        except json.JSONDecodeError:
            continue
    return raw[:320] if raw else None


def run_task(task_id: str, seed: int = 42) -> Dict[str, object]:
    env = DealRoomEnvironment()
    obs = env.reset(seed=seed, task_id=task_id)
    policy = ProtocolPolicy()
    short_model = MODEL_NAME.split("/")[-1] if "/" in MODEL_NAME else MODEL_NAME
    print(f"[START] task={task_id} env={BENCHMARK} model={short_model}")

    rewards: List[float] = []
    final_score = 0.0
    success = False
    step_num = 0

    try:
        while not obs.done and step_num < obs.max_rounds + 2:
            step_num += 1
            action = policy.build_action(obs)
            obs, reward, done, info = env.step(action)
            rewards.append(reward)
            error = info.get("last_action_error") or "null"
            print(
                f"[STEP] step={step_num} action={action.action_type}(target={','.join(action.target_ids) or action.target}) "
                f"reward={reward:.2f} done={str(done).lower()} error={error}"
            )
            if done:
                final_score = reward
                success = reward >= 0.35
                break
    except Exception as exc:
        print(
            f"[STEP] step={step_num} action=error reward=0.00 done=true error={str(exc)[:120]}"
        )
        rewards.append(0.0)

    reward_str = ",".join(f"{value:.2f}" for value in rewards)
    print(
        f"[END] success={str(success).lower()} steps={step_num} score={final_score:.2f} rewards={reward_str}"
    )
    return {
        "task": task_id,
        "score": final_score,
        "steps": step_num,
        "success": success,
    }


if __name__ == "__main__":
    for task_name in ["aligned", "conflicted", "hostile_acquisition"]:
        run_task(task_name, seed=42)
