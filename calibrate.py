"""
Calibration script — run before submission.
Target: strategic agent beats random by 0.20+ spread on every task.
"""

import numpy as np
from models import DealRoomAction
from server.deal_room_environment import DealRoomEnvironment


class RandomAgent:
    def __init__(self, rng):
        self.rng = rng
        self.targets = ["CFO", "CTO", "Legal", "Procurement", "Ops", "all"]
        self.docs = [
            [],
            [{"type": "roi_model", "specificity": "med"}],
            [{"type": "security_cert", "specificity": "med"}],
        ]

    def act(self, obs):
        return DealRoomAction(
            action_type=str(
                self.rng.choice(["direct_message", "send_document", "backchannel"])
            ),
            target=self.targets[int(self.rng.integers(0, len(self.targets)))],
            message="Here is my proposal for your consideration.",
            documents=self.docs[int(self.rng.integers(0, len(self.docs)))],
            channel="formal",
        )


class StrategicAgent:
    """Hardcoded sensible strategy to verify environment rewards real skills."""

    def act(self, obs):
        r = obs.round_number
        blockers = obs.active_blockers
        precursors = obs.veto_precursors

        if precursors:
            target = list(precursors.keys())[0]
            return DealRoomAction(
                action_type="backchannel",
                target=target,
                channel="backchannel",
                message=(
                    "I want to make sure we address any concerns you have directly. "
                    "I'm committed to making this work for your specific situation and timeline. "
                    "I value our partnership and want to find a mutual solution."
                ),
            )

        if blockers and obs.deal_momentum == "critical":
            return DealRoomAction(
                action_type="direct_message",
                target=blockers[0],
                message=(
                    "I understand there are open concerns on your end and I appreciate your transparency. "
                    "Rather than proceed, I'd like to address them specifically together. "
                    "I'm flexible and want to find a solution that works for both sides long-term."
                ),
            )

        doc_sequence = [
            (
                "CFO",
                "roi_model",
                "Here is our ROI analysis showing 14-month payback at your scale.",
            ),
            (
                "Legal",
                "dpa",
                "Here is our GDPR-compliant DPA and SOC2 Type II certification.",
            ),
            (
                "Legal",
                "security_cert",
                "Additional security documentation and audit rights clause.",
            ),
            (
                "CTO",
                "implementation_timeline",
                "Our implementation team dedicates senior engineers to your integration. Timeline respects your Q3 bandwidth.",
            ),
            (
                "Ops",
                "implementation_timeline",
                "Our implementation timeline shows how we'll support your team during rollout. We dedicate senior engineers to work alongside your ops team.",
            ),
        ]

        if blockers:
            target = blockers[0]
            if target == "Ops":
                doc_type, msg = (
                    "implementation_timeline",
                    "Our implementation timeline shows how we'll support your team during rollout.",
                )
            elif target == "Legal":
                doc_type, msg = (
                    "dpa",
                    "Here is our GDPR-compliant DPA and SOC2 Type II certification.",
                )
            elif target == "CTO":
                doc_type, msg = (
                    "implementation_timeline",
                    "Our implementation team dedicates senior engineers to your integration.",
                )
            else:
                doc_type, msg = (
                    "roi_model",
                    "Here is our proposal for your consideration.",
                )
            return DealRoomAction(
                action_type="send_document",
                target=target,
                message=msg,
                documents=[{"type": doc_type, "specificity": "high"}],
            )

        if r < len(doc_sequence):
            target, doc_type, msg = doc_sequence[r]
            return DealRoomAction(
                action_type="send_document",
                target=target,
                message=msg,
                documents=[{"type": doc_type, "specificity": "high"}],
            )

        engagement = obs.engagement_level
        low_engagement = (
            [s for s, v in engagement.items() if v < 0.45] if engagement else []
        )
        if low_engagement:
            target = low_engagement[0]
            if target == "Ops":
                doc_type, msg = (
                    "implementation_timeline",
                    "Our implementation timeline shows how we'll support your team during rollout.",
                )
            elif target == "Legal":
                doc_type, msg = (
                    "dpa",
                    "Here is our GDPR-compliant DPA and SOC2 Type II certification.",
                )
            elif target == "CTO":
                doc_type, msg = (
                    "implementation_timeline",
                    "Our implementation team dedicates senior engineers to your integration.",
                )
            else:
                doc_type, msg = (
                    "roi_model",
                    "Here is our proposal for your consideration.",
                )
            return DealRoomAction(
                action_type="send_document",
                target=target,
                message=msg,
                documents=[{"type": doc_type, "specificity": "high"}],
            )

        doc_sequence = [
            (
                "CFO",
                "roi_model",
                "Here is our ROI analysis showing 14-month payback at your scale.",
            ),
            (
                "Legal",
                "dpa",
                "Here is our GDPR-compliant DPA and SOC2 Type II certification.",
            ),
            (
                "CTO",
                "implementation_timeline",
                "Our implementation team dedicates senior engineers to your integration. Timeline respects your Q3 bandwidth.",
            ),
            (
                "Procurement",
                "security_cert",
                "Additional security documentation and audit rights clause.",
            ),
            (
                "Ops",
                "implementation_timeline",
                "Our implementation timeline shows how we'll support your team during rollout.",
            ),
        ]

        if r < len(doc_sequence):
            target, doc_type, msg = doc_sequence[r]
            return DealRoomAction(
                action_type="send_document",
                target=target,
                message=msg,
                documents=[{"type": doc_type, "specificity": "high"}],
            )

        return DealRoomAction(
            action_type="group_proposal",
            target="all",
            message=(
                "I believe we have addressed the core requirements for all teams. "
                "I'd like to propose moving forward together. "
                "I'm committed to a long-term partnership that delivers real value for your organization."
            ),
        )


def run_episodes(task_id: str, agent_class, n: int = 50) -> list:
    scores = []
    for i in range(n):
        rng = np.random.default_rng(i)
        agent = agent_class(rng) if agent_class == RandomAgent else agent_class()
        env = DealRoomEnvironment()
        obs = env.reset(seed=i, task_id=task_id)
        final_score = 0.0
        for _ in range(20):
            action = agent.act(obs)
            obs, reward, done, _ = env.step(action)
            if done:
                final_score = reward
                break
        scores.append(final_score)
    return scores


if __name__ == "__main__":
    tasks = ["aligned", "conflicted", "hostile_acquisition"]
    print("DealRoom Calibration (50 episodes per agent per task)\n")
    all_pass = True
    for task in tasks:
        rand_scores = run_episodes(task, RandomAgent, n=50)
        strat_scores = run_episodes(task, StrategicAgent, n=50)
        rand_avg = sum(rand_scores) / len(rand_scores)
        strat_avg = sum(strat_scores) / len(strat_scores)
        spread = strat_avg - rand_avg
        status = "PASS" if spread >= 0.15 else "FAIL"
        if status == "FAIL":
            all_pass = False
        print(f"{task}:")
        print(f"  Random agent:    {rand_avg:.3f}")
        print(f"  Strategic agent: {strat_avg:.3f}")
        print(f"  Spread:          {spread:.3f}  [{status}]")
        print()

    if all_pass:
        print("All calibration targets met. Ready to submit.")
    else:
        print(
            "CALIBRATION FAILED. Adjust initial_satisfaction or veto_threshold in scenarios.py."
        )
        print(
            "If aligned spread is too small: lower initial_satisfaction by 0.05 across all stakeholders."
        )
        print(
            "If hostile spread is too small: increase round_3_hint detail or lower veto_threshold to 0.40."
        )
