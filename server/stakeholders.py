"""
StakeholderEngine + STAKEHOLDER_TEMPLATES + DOCUMENT_EFFECTS

All stakeholder state and response generation. Zero LLM calls. Deterministic given RNG.

Template design principles enforced:
1. Overlapping surface signals between stances — testing and delaying sometimes produce
   similar-sounding responses. Agent must use accumulated history, not per-message pattern.
2. Implicit stakeholder concerns — priorities revealed through language, never stated directly.
3. 4+ variants per bucket — prevents cycle repetition in long episodes.
"""

from typing import Dict, List, TYPE_CHECKING
import numpy as np

if TYPE_CHECKING:
    from models import DealRoomState

STAKEHOLDER_TEMPLATES: Dict[str, Dict[str, Dict[str, List[str]]]] = {
    "CFO": {
        "cooperative": {
            "high": [
                "The ROI projections align well with our Q3 cost targets. Let's discuss payment structure.",
                "We're making good progress. I need to map this to our budget cycle before sign-off.",
                "The financial case is solid. A few clarifications on payment milestones and we can move.",
                "The numbers work for me. Let's align on terms and move forward.",
                "Good traction here. I want to make sure the board can validate the payback assumptions.",
            ],
            "mid": [
                "Can you walk me through the cost-reduction assumptions in more detail?",
                "I need to see how this maps to our existing OpEx commitments for Q3.",
                "The ROI case needs tightening before I take this to the board.",
                "What's the basis for the 18-month payback? Our finance team will scrutinize this.",
                "I'm interested but the financial modeling needs another pass.",
            ],
            "low": [
                "This doesn't meet our ROI threshold. We need a significantly stronger business case.",
                "The cost structure needs to be reworked. I can't move forward on these terms.",
                "Our board is asking hard questions about spend. This needs to be much more compelling.",
                "I have serious reservations about the financial justification as it stands.",
            ],
        },
        "testing": {
            "high": [
                "Walk me through the assumptions behind your cost-saving projections.",
                "How does this compare to our current vendor spend? I need the delta to be clear.",
                "What happens to the ROI model if implementation takes 20% longer than projected?",
                "Who else in your portfolio has achieved this level of savings at our scale?",
            ],
            "mid": [
                "What's the basis for these projections? They seem optimistic given our environment.",
                "I'd want our internal finance team to validate these numbers independently.",
                "Help me understand total cost of ownership, not just the license fee.",
                "We'll need evidence of comparable results at similar-sized organizations.",
                "I need more time to review this internally before I can give you a read.",
            ],
            "low": [
                "These numbers don't hold up to scrutiny.",
                "I've seen vendors make these claims before. The reality is usually quite different.",
                "We'll need substantial justification before reconsidering this path.",
                "Our finance team has significant concerns about the methodology here.",
            ],
        },
        "delaying": {
            "high": [
                "This looks promising. I need to loop in our controller before we finalize anything.",
                "We're working through some internal budget items. Bear with us.",
                "Good progress. We'll review with the finance committee and revert by end of week.",
                "I need to revisit our Q4 commitments before locking anything in.",
            ],
            "mid": [
                "We're working through some internal approvals. This isn't the right moment.",
                "Our budget cycle is at a sensitive point. Let's revisit in a few weeks.",
                "What's the basis for these projections?",
                "I'll need to take this back to the team before I can give a clear answer.",
                "There are other priorities competing for my attention right now.",
            ],
            "low": [
                "This isn't a good time to advance this discussion.",
                "We're in a budget freeze. I'd suggest we reconnect next quarter.",
                "I'll need to get back to you. Several internal reviews are pending.",
                "Let's table this until we have more internal clarity.",
            ],
        },
        "obfuscating": {
            "high": [
                "There are a few angles we're still evaluating from a financial perspective.",
                "The picture is more complex than it appears on the surface.",
                "We appreciate the proposal. There are several factors in play.",
            ],
            "mid": [
                "It's difficult to say at this stage. We have competing priorities.",
                "The financial picture is evolving. It's not straightforward to comment.",
                "There are considerations I'm not in a position to share at this point.",
                "We're looking at this holistically — hard to comment specifically right now.",
            ],
            "low": [
                "This isn't something I can address directly right now.",
                "There are dynamics at play that I'd rather not get into.",
                "I'd prefer to keep the discussion at a higher level for now.",
            ],
        },
    },
    "CTO": {
        "cooperative": {
            "high": [
                "The architecture looks sound. I'd like to go deeper on the API integration points.",
                "My team reviewed the technical specs — this looks feasible within our stack.",
                "The implementation approach is reasonable. Timeline needs to account for our Q3 load.",
                "I'm encouraged by what I'm seeing. Let's schedule a technical deep-dive.",
                "This is coming along well technically. My main concern is my team's bandwidth.",
            ],
            "mid": [
                "Can you clarify the API response time guarantees under peak load?",
                "How does this interact with our data warehouse? The integration story needs work.",
                "The migration path from our current system isn't clearly documented.",
                "My team is stretched. I need to understand the implementation support model better.",
                "What's the rollback plan if we encounter issues post-deployment?",
            ],
            "low": [
                "I have significant technical concerns that haven't been addressed.",
                "The integration complexity is being understated. This will strain my team considerably.",
                "We've had bad experiences with vendors who overpromised on technical delivery.",
                "The timeline is unrealistic given our current architecture and team commitments.",
            ],
        },
        "testing": {
            "high": [
                "What's the actual API response time under our expected load profile?",
                "Walk me through the data migration approach in more detail.",
                "How many integrations have you completed with systems similar to ours?",
                "What does your implementation team's on-site availability look like during rollout?",
            ],
            "mid": [
                "I'm waiting on feedback from our infrastructure team before I can respond properly.",
                "The technical documentation doesn't address our specific environment.",
                "Who on your team will own the integration? I need to assess their experience.",
                "What are the known failure modes and how are they mitigated?",
                "Can you provide references from clients with similar technical complexity?",
            ],
            "low": [
                "The architecture raises more questions than it answers.",
                "My senior engineers have reviewed this and have serious concerns.",
                "The technical risk profile is higher than we're comfortable with.",
                "We'd need a full technical audit before considering this further.",
            ],
        },
        "delaying": {
            "high": [
                "I'm waiting on feedback from our infrastructure team before we can advance.",
                "We're in the middle of a sprint cycle. Give us a week to surface this properly.",
                "My team hasn't had bandwidth to do a thorough technical review yet.",
                "This needs more internal deliberation before I can give you a concrete answer.",
            ],
            "mid": [
                "There's a lot going on in our stack right now. Timing isn't ideal.",
                "We haven't been able to fully evaluate this. What's your flexibility on timeline?",
                "What's the actual API response time under load?",
                "My team needs to weigh in and they've been heads-down on other priorities.",
                "Let's pick this back up once our current release is out the door.",
            ],
            "low": [
                "My team simply doesn't have capacity for this right now.",
                "We're in a code freeze. Technical evaluations need to wait.",
                "I can't make commitments until we clear our current backlog.",
                "This will need to wait until next quarter at the earliest.",
            ],
        },
        "obfuscating": {
            "high": [
                "There are several technical considerations we're still working through.",
                "The integration landscape is more nuanced than it might appear from the outside.",
                "My team has some concerns but they're not fully crystallized yet.",
            ],
            "mid": [
                "It's a complex picture technically. Hard to give you a clear read right now.",
                "There are dependencies I'd prefer not to get into at this stage.",
                "The technical evaluation is ongoing. I don't want to pre-judge the outcome.",
                "We're seeing some things internally that are relevant but I can't share yet.",
            ],
            "low": [
                "I don't think this is going in the right direction technically.",
                "There are things I'm not in a position to discuss that are relevant here.",
                "I'd prefer to keep this vague for now.",
            ],
        },
    },
    "Legal": {
        "cooperative": {
            "high": [
                "The DPA is well-structured. A few clauses need refinement before we can sign.",
                "We're comfortable with the liability framework. Let's align on indemnification language.",
                "The GDPR compliance posture looks solid. I'll want audit rights formally documented.",
                "Good progress on the legal terms. The DPA just needs a couple of adjustments.",
                "We're close. Primarily cleanup at this point before we can move to execution.",
            ],
            "mid": [
                "The liability exposure in clause 12 is broader than we're comfortable with.",
                "We need more specificity in the data handling provisions before moving forward.",
                "Our standard DPA won't work here — we'll need a custom agreement drafted.",
                "What jurisdictions does your data processing infrastructure operate in?",
                "The indemnification terms need to be mutual, not one-directional as written.",
            ],
            "low": [
                "The contractual terms create unacceptable liability exposure. This is a non-starter.",
                "We can't sign anything with this data handling language as written.",
                "The compliance posture doesn't meet our regulatory requirements.",
                "This needs a complete legal review from the ground up.",
            ],
        },
        "testing": {
            "high": [
                "Walk me through your data residency model for EU data subjects.",
                "What's your breach notification timeline and internal process?",
                "How are sub-processors managed and contractually notified to us?",
                "I'll need to review your most recent security audit report before we advance.",
            ],
            "mid": [
                "Your standard contract terms don't address our specific regulatory context.",
                "We'll need your SOC 2 Type II report and any recent penetration test results.",
                "The limitation of liability clause needs to be negotiated significantly.",
                "We need written confirmation of your GDPR compliance program and DPO contact.",
                "I need to run this by our external counsel before we can respond.",
            ],
            "low": [
                "The contractual terms don't reflect current regulatory requirements.",
                "We've had issues with similar language in other vendor agreements. Red flag.",
                "I'll need to escalate this to our general counsel.",
                "This doesn't meet the bar we established after our last vendor audit.",
            ],
        },
        "delaying": {
            "high": [
                "I need to run this by our external counsel before we can progress on this.",
                "We're in the middle of a compliance review cycle. Timing is challenging right now.",
                "Legal reviews take time. We'll revert once we've completed our standard process.",
                "There are a few internal approvals in the queue ahead of this one.",
            ],
            "mid": [
                "Our legal team is backed up with other matters right now.",
                "I need to run this by our external counsel.",
                "We can't rush this review — the regulatory stakes are too high to shortcut.",
                "This is waiting on input from our privacy officer. No timeline yet.",
                "Let's revisit this once we're through our current compliance cycle.",
            ],
            "low": [
                "This isn't moving forward until all legal concerns are fully resolved.",
                "We need more time. I genuinely can't give you a timeline at this point.",
                "Our review process is thorough. We won't be rushed on data handling matters.",
                "This is in the queue but I can't tell you when we'll get to it.",
            ],
        },
        "obfuscating": {
            "high": [
                "There are some legal nuances we're still working through on our side.",
                "The contractual picture is more complex than the headline terms suggest.",
                "We're evaluating several angles. It's not a simple assessment.",
            ],
            "mid": [
                "There are legal considerations I'm not in a position to elaborate on right now.",
                "The compliance landscape here is complicated. Hard to be specific.",
                "We're looking at a number of factors I'd prefer to keep internal.",
                "The legal review is ongoing. I don't want to preview where it's going.",
            ],
            "low": [
                "There are things our counsel has flagged that I can't discuss publicly.",
                "The legal situation here is more complex than it appears.",
                "I'm not in a position to comment specifically on this right now.",
            ],
        },
    },
    "Procurement": {
        "cooperative": {
            "high": [
                "The compliance documentation is in good shape. Process is moving forward cleanly.",
                "We're on track with the standard evaluation process. Good progress overall.",
                "The vendor qualification requirements have been met. Next step is contract review.",
                "Everything is processually sound. Minor documentation cleanup remaining.",
                "We're aligned on the procurement requirements. Let's finalize the evaluation.",
            ],
            "mid": [
                "We need the full vendor compliance questionnaire before we can advance.",
                "Your insurance certificates don't match our standard minimum thresholds.",
                "The RFP response needs to be more detailed on implementation methodology.",
                "Have you gone through our standard onboarding process? Missing some documents.",
                "Our evaluation committee needs a formal presentation before sign-off.",
            ],
            "low": [
                "The documentation is incomplete. We can't advance through our standard process.",
                "Your vendor qualification doesn't meet our baseline requirements as written.",
                "We've identified compliance gaps that need to be resolved before we can proceed.",
                "Our procurement committee has serious concerns about the evaluation process.",
            ],
        },
        "testing": {
            "high": [
                "Can you confirm your D&B rating and business continuity plan documentation?",
                "Walk us through your standard onboarding and implementation methodology.",
                "We'll need references from three similar implementations in our sector.",
                "What does your vendor management process look like post-contract signature?",
            ],
            "mid": [
                "Your proposal doesn't follow our standard RFP format. That creates process issues.",
                "We need to validate your compliance with our supplier code of conduct.",
                "Has your organization undergone a third-party security assessment recently?",
                "We'll need to conduct a site visit as part of our standard due diligence.",
                "I need to check with our legal team on a few items before responding.",
            ],
            "low": [
                "The compliance gaps here are more significant than initially apparent.",
                "We've found inconsistencies in the documentation that need to be resolved.",
                "Our evaluation committee is not satisfied with the responses provided so far.",
                "This doesn't meet our vendor qualification standards in several areas.",
            ],
        },
        "delaying": {
            "high": [
                "We're running the standard three-bid evaluation. Results will come.",
                "The committee hasn't convened yet. We'll have clarity by end of month.",
                "Our evaluation timeline is set — we follow the process without exception.",
                "There are a few internal approvals required before we can move this forward.",
            ],
            "mid": [
                "Our standard process requires multiple review stages. This takes time.",
                "I need to check with our legal team on this.",
                "The evaluation committee meets monthly. We'll be on the next agenda.",
                "Our procurement cycle is rigid. We don't accelerate for individual vendors.",
                "We're following our standard timeline. There's no mechanism to expedite.",
            ],
            "low": [
                "We cannot deviate from our standard procurement process. Full stop.",
                "This is not moving forward until all process requirements are met.",
                "Our evaluation timeline doesn't flex based on vendor preference.",
                "The committee hasn't approved advancing this to the next stage.",
            ],
        },
        "obfuscating": {
            "high": [
                "There are a few process steps we're working through internally.",
                "The evaluation is progressing but I can't share specifics at this stage.",
                "We're following our standard process. It will conclude when it concludes.",
            ],
            "mid": [
                "The evaluation is more involved than it might appear from the outside.",
                "There are internal factors I'm not in a position to share with you.",
                "The process is moving but I genuinely can't give you a precise timeline.",
                "We're assessing several dimensions simultaneously. Hard to comment on any one.",
            ],
            "low": [
                "There are aspects of our evaluation I'm not able to discuss externally.",
                "The process has its own internal logic. I can't really elaborate.",
                "I'd prefer not to comment on where things stand internally right now.",
            ],
        },
    },
    "Ops": {
        "cooperative": {
            "high": [
                "This looks great for our Q3 rollout. The implementation timeline maps perfectly.",
                "My team is excited about this. The early milestones look completely achievable.",
                "The delivery roadmap aligns well with our internal project plan.",
                "We're fully aligned on the scope. I can get my team mobilized quickly.",
                "This is exactly what we needed. Q3 delivery is genuinely critical for us.",
            ],
            "mid": [
                "Can we get a more detailed implementation roadmap? We need to plan our involvement.",
                "The Q3 deadline is non-negotiable for us internally. Can you commit to that?",
                "We need clarity on what we're responsible for versus what your team handles.",
                "What resources do you need from our side during implementation?",
                "Our internal sponsors are counting on this landing before end of Q3.",
            ],
            "low": [
                "I'm losing confidence that the Q3 delivery is realistic at this point.",
                "Our leadership is asking questions I can't answer. The timeline is slipping.",
                "We've already communicated this timeline internally. A slip would be damaging.",
                "I'm concerned this won't be ready when we need it. That's a real problem.",
            ],
        },
        "testing": {
            "high": [
                "What early deliverables can we commit to for internal reporting purposes?",
                "Walk us through what a typical week 1 looks like during implementation.",
                "Who will be our primary point of contact throughout the rollout?",
                "What's your track record on hitting the delivery dates you commit to?",
            ],
            "mid": [
                "I need concrete milestones I can show my leadership by end of month.",
                "What happens if you miss the Q3 target? What's the contingency plan?",
                "Our internal project plan depends on your timeline. Be more precise.",
                "I'm working through the internal approvals on my side still.",
                "What does your implementation team's experience look like with similar deployments?",
            ],
            "low": [
                "The delivery commitments don't inspire confidence based on what I've seen.",
                "I've had vendors miss timelines before. What makes this situation different?",
                "Our leadership won't accept another missed deadline. I need real certainty.",
                "The timeline looks unrealistic given what I know about our environment.",
            ],
        },
        "delaying": {
            "high": [
                "I'm working through the internal approvals on my side. Give me one week.",
                "We're finalizing our internal project plan. Almost ready to commit.",
                "My team needs to review the implementation approach before we lock in.",
                "There are a few internal sign-offs I need to collect first.",
            ],
            "mid": [
                "We're still finalizing our internal resourcing plan for this.",
                "There's a leadership review happening internally that directly affects this.",
                "I need concrete milestones I can show my leadership.",
                "My hands are tied until a few internal decisions get made above me.",
                "We're waiting on some internal clarity before we can truly commit.",
            ],
            "low": [
                "We're not in a position to move forward on this right now.",
                "There are internal blockers I'm working through. Not the right moment.",
                "My leadership has put a pause on new commitments this quarter.",
                "We're reassessing our Q3 priorities. I'll be in touch when that's settled.",
            ],
        },
        "obfuscating": {
            "high": [
                "There are some internal dynamics I'm navigating on my side. It's complicated.",
                "We're working through a few things. Nothing to worry about at this stage.",
                "The internal situation is a bit fluid right now.",
            ],
            "mid": [
                "There's context I'm not in a position to share that's relevant here.",
                "It's complicated on our side. I wish I could be more specific.",
                "There are factors at play I can't elaborate on right now.",
                "Internal politics make this harder to predict than I'd like.",
            ],
            "low": [
                "I can't really get into the specifics at this point.",
                "There are things happening internally that affect this. I can't say more.",
                "It's better I don't comment on the internal situation right now.",
            ],
        },
    },
}

DOCUMENT_EFFECTS = {
    "roi_model": {
        "high": {"CFO": 0.18, "Procurement": 0.08},
        "med": {"CFO": 0.10, "Procurement": 0.05},
        "low": {"CFO": 0.04},
    },
    "security_cert": {
        "high": {"Legal": 0.20, "CTO": 0.12, "Procurement": 0.06},
        "med": {"Legal": 0.12, "CTO": 0.07},
        "low": {"Legal": 0.05},
    },
    "implementation_timeline": {
        "high": {"CTO": 0.18, "Ops": 0.16},
        "med": {"CTO": 0.10, "Ops": 0.09},
        "low": {"CTO": 0.04, "Ops": 0.04},
    },
    "dpa": {
        "high": {"Legal": 0.22, "Procurement": 0.08},
        "med": {"Legal": 0.14, "Procurement": 0.04},
        "low": {"Legal": 0.06},
    },
    "reference_case": {
        "high": {"CFO": 0.10, "Procurement": 0.14, "CTO": 0.08},
        "med": {"CFO": 0.06, "Procurement": 0.09, "CTO": 0.05},
        "low": {"Procurement": 0.04},
    },
}

COLLABORATIVE_SIGNALS = [
    "understand",
    "partnership",
    "mutual",
    "together",
    "value",
    "appreciate",
    "flexible",
    "work with",
    "long-term",
    "relationship",
    "transparent",
    "committed",
    "invested in your success",
    "your goals",
    "collaborative",
    "joint",
    "shared",
    "tailored",
]
AGGRESSIVE_SIGNALS = [
    "demand",
    "require",
    "final offer",
    "unacceptable",
    "must",
    "non-negotiable",
    "take it or leave",
    "bottom line",
    "deadline",
    "insist",
    "ultimatum",
    "last chance",
]


class StakeholderEngine:
    STAKEHOLDER_IDS = ["CFO", "CTO", "Legal", "Procurement", "Ops"]

    def __init__(self):
        self.state = None
        self.rng = None
        self._pre_action_beliefs: Dict = {}
        self._stances: Dict[str, str] = {}

    def reset(self, state, rng, scenario: dict):
        self.state = state
        self.rng = rng
        self._pre_action_beliefs = {}
        self._stances = {}
        for sid in self.STAKEHOLDER_IDS:
            sat = state.satisfaction.get(sid, 0.5)
            if sat > 0.60:
                self._stances[sid] = "cooperative"
            elif sat > 0.45:
                self._stances[sid] = "testing"
            else:
                self._stances[sid] = "delaying"

    def generate_opening(self) -> Dict[str, str]:
        return {
            "CFO": "Thanks for reaching out. Before we go further I'll need detailed ROI projections. The board will ask for a defensible payback period.",
            "CTO": "Happy to evaluate this. I'll need to review the technical architecture documentation and understand the integration approach with our current stack.",
            "Legal": "We'll require a full data processing agreement and liability review. GDPR compliance documentation is essential given our EU operations.",
            "Procurement": "Please ensure all compliance documentation is ready. Our standard vendor qualification process will need to be completed before we can advance.",
            "Ops": "We're excited about the potential here. A Q3 implementation date would align perfectly with our internal roadmap.",
        }

    def apply_action(self, action_dict: dict, rng):
        from .claims import expand_targets

        self._pre_action_beliefs = {k: dict(v) for k, v in self.state.beliefs.items()}
        targets = expand_targets(action_dict.get("target", "all"))
        message = action_dict.get("message", "")
        documents = action_dict.get("documents", [])
        rapport_delta = self._compute_rapport(message)

        for target in targets:
            if target not in self.STAKEHOLDER_IDS:
                continue
            for doc in documents:
                doc_type = doc.get("type", "")
                specificity = doc.get("specificity", "med")
                effects = DOCUMENT_EFFECTS.get(doc_type, {}).get(specificity, {})
                if target in effects:
                    self.state.satisfaction[target] = min(
                        1.0, self.state.satisfaction[target] + effects[target]
                    )
            if rapport_delta != 0:
                speed = 0.06 + abs(rapport_delta) * 0.04
                self.state.beliefs[target]["competence"] = min(
                    1.0,
                    max(
                        0.0,
                        self.state.beliefs[target]["competence"]
                        + speed * rapport_delta,
                    ),
                )
                self.state.satisfaction[target] = min(
                    1.0,
                    max(
                        self.state.trust_floors.get(target, 0.0),
                        self.state.satisfaction[target] + rapport_delta * 0.04,
                    ),
                )
            if self.state.scrutiny_mode:
                self.state.satisfaction[target] = max(
                    self.state.trust_floors.get(target, 0.0),
                    self.state.satisfaction[target] - 0.03,
                )
            self.state.rounds_since_last_contact[target] = 0
            self._update_stance(target)

        for sid in self.STAKEHOLDER_IDS:
            if sid not in targets:
                self.state.rounds_since_last_contact[sid] = (
                    self.state.rounds_since_last_contact.get(sid, 0) + 1
                )

    def generate_responses(self, action_dict: dict, state) -> Dict[str, str]:
        from .claims import expand_targets

        targets = expand_targets(action_dict.get("target", "all"))
        responses = {}
        for sid in self.STAKEHOLDER_IDS:
            if sid in targets or action_dict.get("target", "").lower() == "all":
                stance = self._stances.get(sid, "cooperative")
                sat = state.satisfaction.get(sid, 0.5)
                responses[sid] = self._generate_single_response(sid, stance, sat)
        return responses

    def get_belief_deltas(self) -> Dict[str, float]:
        deltas = {}
        for sid in self.STAKEHOLDER_IDS:
            pre = self._pre_action_beliefs.get(sid, {})
            if not pre:
                deltas[sid] = 0.0
                continue
            current = self.state.beliefs.get(sid, {})
            delta = (
                sum(
                    abs(current.get(d, 0.5) - pre.get(d, 0.5))
                    for d in ["competence", "risk_tolerance", "pricing_rigor"]
                )
                / 3.0
            )
            deltas[sid] = round(delta, 4)
        return deltas

    def _generate_single_response(self, sid: str, stance: str, sat: float) -> str:
        templates = STAKEHOLDER_TEMPLATES.get(sid, {}).get(stance, {})
        bucket = "high" if sat > 0.65 else "low" if sat < 0.35 else "mid"
        options = templates.get(
            bucket, templates.get("mid", ["Understood. Let's continue."])
        )
        return options[int(self.rng.integers(0, len(options)))]

    def _compute_rapport(self, message: str) -> float:
        msg_lower = message.lower()
        collab = sum(0.05 for w in COLLABORATIVE_SIGNALS if w in msg_lower)
        aggro = sum(0.05 for w in AGGRESSIVE_SIGNALS if w in msg_lower)
        return round(max(-0.30, min(0.30, collab - aggro)), 4)

    def _update_stance(self, sid: str):
        sat = self.state.satisfaction.get(sid, 0.5)
        if sat > 0.65:
            self._stances[sid] = "cooperative"
        elif sat > 0.50:
            self._stances[sid] = str(self.rng.choice(["testing", "cooperative"]))
        elif sat > 0.35:
            self._stances[sid] = str(self.rng.choice(["testing", "delaying"]))
        else:
            self._stances[sid] = str(self.rng.choice(["delaying", "obfuscating"]))
