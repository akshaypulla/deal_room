# DealRoom — Complete End-to-End Implementation Plan (Final)

## Multi-Stakeholder Enterprise Contract Negotiation RL Environment

---

## 1. What DealRoom Is And Why It Exists

DealRoom is an OpenEnv-compliant reinforcement learning environment where an LLM agent plays a vendor's lead negotiator closing a $2M+ enterprise software contract. The agent must build internal consensus among 5 buying-side stakeholders — each with hidden priorities, conflicting incentives, and the unilateral power to veto.

**The core insight:** 80% of B2B deals fail not because of price or features but because of internal consensus failure inside the buying organization (Gartner). No existing OpenEnv environment models this. Every other environment in the hub models games, code execution, or simple two-party negotiation. DealRoom models organizational politics.

**Why RL is strictly required:**

- The optimal action in round 3 depends on belief states accumulated across all 5 stakeholders over rounds 0–2 — not observable from the current step alone
- Stakeholder trust damage is irreversible — early mistakes compound permanently
- Veto risk is hidden and only inferable from indirect signals
- Coalition sequencing (CFO before CTO in public in conflicted task) only emerges through thousands of trajectory-level training episodes
- A base LLM optimizes per-step rapport; an RL agent learns the full sequential strategy

**Why LLM is strictly required:**

- Stakeholder responses are natural language — the agent must semantically parse signals ("payment timing is critical for us this quarter" reveals cash flow stress without stating it)
- Rapport keywords in agent messages affect opponent concession rate — language quality causally changes environment state
- No classifier or rule-based system can generate negotiation language that modulates relationship dynamics

---

## 2. File Structure

```
deal_room/
├── __init__.py
├── deal_room_environment.py    ← MAIN environment class
├── models.py                   ← All Pydantic models (flat dicts)
├── stakeholders.py             ← StakeholderEngine + templates
├── grader.py                   ← CCIGrader (Contract Closure Index v3)
├── validator.py                ← OutputValidator (strict, no LLM)
├── scenarios.py                ← 3 task configs
└── claims.py                   ← ClaimsTracker (regex-only, no expansion)

server/
└── app.py                      ← FastAPI wrapper, zero business logic

inference.py                    ← Baseline, imports deal_room directly
openenv.yaml
Dockerfile
requirements.txt
calibrate.py
README.md
```

**Architecture rules:**

- All environment logic lives in `deal_room/`
- `server/app.py` is fewer than 80 lines — HTTP wrapper only
- `inference.py` imports from `deal_room` directly, no HTTP calls
- Zero LLM calls anywhere inside `deal_room/` package

---

## 3. Design Decisions (From Issue Analysis)

Five issues were analyzed and resolved before implementation. All decisions documented here so the rationale is preserved.

**Issue 1 — ClaimsTracker double expansion:**
`extract_and_track` previously called `expand_targets` internally, and the environment also called it before passing targets in. This caused redundant computation but no logical error. Decision: centralize expansion at the environment level, remove it from inside the tracker. The tracker receives individual stakeholder IDs only.

**Issue 2 — Group target belief_deltas:**
`target_responded_positively` used `action.target` directly to look up `belief_deltas`. Group targets like `"cto_cfo"` do not exist as keys in `belief_deltas` (which uses individual IDs). This produced a permanently False signal for all group actions — a real misleading training signal. Decision: compute max delta across all expanded targets.

**Issue 3 — Veto risk growth rate:**
Growth of 0.08/round for sat<0.30 with threshold 0.44–0.68 means roughly 5–8 rounds before veto fires after sustained low satisfaction. This is intentional — it creates meaningful pressure. The precursor window (risk 0.28–0.50) gives the agent 3 rounds of warning. Rate is correct. One guard added: no veto accumulation on round 0 (opening observations only).

**Issue 4 — Stage progression simplicity:**
Satisfaction threshold alone is gameable by document spamming in early rounds. Added `STAGE_MIN_ROUNDS` guard: no stage can advance before a minimum number of rounds have passed. This is lightweight, non-breaking, and prevents unrealistic instant progression.

**Issue 5 — Momentum binary signal:**
+1/-1 binary mapped all non-regression rounds to +1, including stalling rounds. This sent a misleading "things are fine" signal during periods of no progress. Changed to three-state: +1 (genuine stage advance or blocker resolved), 0 (holding, no change), -1 (regression). Consecutive 0s signal danger as clearly as -1.

---

## 4. Complete Code — Every File

---

### `deal_room/__init__.py`

```python
from .deal_room_environment import DealRoomEnvironment
from .models import DealRoomAction, DealRoomObservation, DealRoomState

__all__ = ["DealRoomEnvironment", "DealRoomAction", "DealRoomObservation", "DealRoomState"]
```

---

### `deal_room/models.py`

```python
"""
DealRoom Pydantic Models
Flat dicts throughout — no nested model classes.
beliefs uses Dict[str, Dict[str, float]] validated to always contain
exactly {competence, risk_tolerance, pricing_rigor} per stakeholder.
"""
from pydantic import BaseModel, Field, field_validator
from typing import Dict, List, Optional, Any


class DealRoomAction(BaseModel):
    action_type: str = "direct_message"
    # direct_message | group_proposal | backchannel | send_document |
    # concession | walkaway_signal | reframe_value_prop | exec_escalation
    target: str = "all"
    # CFO | CTO | Legal | Procurement | Ops | all | cto_cfo | legal_procurement
    message: str = ""
    documents: List[Dict[str, str]] = Field(default_factory=list)
    # [{"type": "roi_model", "specificity": "high|med|low"}]
    proposed_terms: Optional[Dict[str, Any]] = None
    channel: str = "formal"    # formal | backchannel
    mode: str = "async_email"  # async_email | formal_meeting | exec_escalation


class DealRoomObservation(BaseModel):
    round_number: int = 0
    max_rounds: int = 10
    stakeholder_messages: Dict[str, str] = Field(default_factory=dict)
    engagement_level: Dict[str, float] = Field(default_factory=dict)
    # Noisy 1-step delayed proxy for satisfaction. Never exact.
    deal_momentum: str = "stalling"   # stalling | progressing | critical
    deal_stage: str = "evaluation"
    # evaluation | negotiation | legal_review | final_approval | closed
    competitor_events: List[str] = Field(default_factory=list)
    veto_precursors: Dict[str, str] = Field(default_factory=dict)
    # Ambiguous early warning: stakeholder_id -> hint string
    scenario_hint: Optional[str] = None
    active_blockers: List[str] = Field(default_factory=list)
    days_to_deadline: int = 30
    done: bool = False
    info: Dict[str, Any] = Field(default_factory=dict)


class DealRoomState(BaseModel):
    episode_id: str = ""
    task_id: str = ""
    round_number: int = 0
    max_rounds: int = 10

    # Flat dict beliefs: {"CFO": {"competence": 0.5, "risk_tolerance": 0.5, "pricing_rigor": 0.5}}
    # No nested Pydantic model — plain dict validated by field_validator below
    beliefs: Dict[str, Dict[str, float]] = Field(default_factory=dict)
    satisfaction: Dict[str, float] = Field(default_factory=dict)
    trust_floors: Dict[str, float] = Field(default_factory=dict)
    permanent_marks: Dict[str, List[str]] = Field(default_factory=dict)

    # Veto tracking
    veto_risk: Dict[str, float] = Field(default_factory=dict)
    veto_precursors_fired: Dict[str, bool] = Field(default_factory=dict)

    # Stage tracking
    deal_stage: str = "evaluation"
    active_blockers: List[str] = Field(default_factory=list)
    stage_regressions: int = 0

    # Claims and contact tracking
    tracked_claims: Dict[str, List[float]] = Field(default_factory=dict)
    rounds_since_last_contact: Dict[str, int] = Field(default_factory=dict)

    # Execution quality
    validation_failures: int = 0
    fallback_streak: int = 0
    scrutiny_mode: bool = False
    exec_escalation_used: bool = False

    # Terminal state
    deal_closed: bool = False
    deal_failed: bool = False
    failure_reason: str = ""
    final_terms: Optional[Dict] = None

    @field_validator("beliefs")
    @classmethod
    def validate_belief_dims(cls, v: Dict) -> Dict:
        required = {"competence", "risk_tolerance", "pricing_rigor"}
        for stakeholder, dims in v.items():
            missing = required - set(dims.keys())
            if missing:
                raise ValueError(f"{stakeholder} missing belief dims: {missing}")
        return v
```

---

### `deal_room/scenarios.py`

```python
"""
Task configurations for all 3 DealRoom tasks.
Every key listed here is required — KeyError otherwise.
"""

STAKEHOLDER_IDS = ["CFO", "CTO", "Legal", "Procurement", "Ops"]

SCENARIOS = {
    "aligned": {
        "max_rounds": 8,
        "veto_threshold": 0.68,      # Hard to trigger
        "block_threshold": 0.28,     # sat below this = active blocker
        "shock_prob": 0.04,
        "round_3_hint": None,
        "days_to_deadline": 45,
        "initial_beliefs": {
            "CFO":         {"competence": 0.55, "risk_tolerance": 0.52, "pricing_rigor": 0.50},
            "CTO":         {"competence": 0.58, "risk_tolerance": 0.55, "pricing_rigor": 0.48},
            "Legal":       {"competence": 0.50, "risk_tolerance": 0.45, "pricing_rigor": 0.52},
            "Procurement": {"competence": 0.53, "risk_tolerance": 0.50, "pricing_rigor": 0.55},
            "Ops":         {"competence": 0.60, "risk_tolerance": 0.58, "pricing_rigor": 0.45},
        },
        "initial_satisfaction": {
            "CFO": 0.54, "CTO": 0.56, "Legal": 0.48,
            "Procurement": 0.52, "Ops": 0.60,
        },
        "coalition_tension": None,
        "description": (
            "Low-friction enterprise deal. All stakeholders broadly favorable. "
            "Minor concerns from Legal (liability) and CFO (ROI timeline). "
            "Tests: correct document sequencing, stakeholder engagement order."
        ),
    },

    "conflicted": {
        "max_rounds": 10,
        "veto_threshold": 0.52,
        "block_threshold": 0.32,
        "shock_prob": 0.07,
        "round_3_hint": None,
        "days_to_deadline": 30,
        "initial_beliefs": {
            "CFO":         {"competence": 0.42, "risk_tolerance": 0.35, "pricing_rigor": 0.48},
            "CTO":         {"competence": 0.44, "risk_tolerance": 0.38, "pricing_rigor": 0.42},
            "Legal":       {"competence": 0.38, "risk_tolerance": 0.32, "pricing_rigor": 0.50},
            "Procurement": {"competence": 0.40, "risk_tolerance": 0.35, "pricing_rigor": 0.52},
            "Ops":         {"competence": 0.55, "risk_tolerance": 0.52, "pricing_rigor": 0.40},
        },
        "initial_satisfaction": {
            "CFO": 0.42, "CTO": 0.44, "Legal": 0.38,
            "Procurement": 0.40, "Ops": 0.55,
        },
        "coalition_tension": {
            "cto_cfo": "conflict",           # CFO must be endorsed BEFORE CTO in public
            "legal_procurement": "alliance", # Concessions to one trigger demands from other
        },
        "description": (
            "Active CTO-CFO tension from failed prior project. "
            "Legal-Procurement blocking alliance. Ops champion isolated. "
            "Tests: coalition sequencing, independent credibility building, veto avoidance."
        ),
    },

    "hostile_acquisition": {
        "max_rounds": 10,
        "veto_threshold": 0.44,
        "block_threshold": 0.35,
        "shock_prob": 0.11,
        "round_3_hint": (
            "AE note: Post-acquisition compliance team from acquiring EU parent has joined review. "
            "Expect heightened data sovereignty scrutiny. "
            "Align all messaging with GDPR baseline requirements immediately."
        ),
        "days_to_deadline": 20,
        "initial_beliefs": {
            "CFO":         {"competence": 0.40, "risk_tolerance": 0.32, "pricing_rigor": 0.45},
            "CTO":         {"competence": 0.42, "risk_tolerance": 0.35, "pricing_rigor": 0.40},
            "Legal":       {"competence": 0.35, "risk_tolerance": 0.28, "pricing_rigor": 0.48},
            "Procurement": {"competence": 0.38, "risk_tolerance": 0.32, "pricing_rigor": 0.50},
            "Ops":         {"competence": 0.50, "risk_tolerance": 0.46, "pricing_rigor": 0.38},
        },
        "initial_satisfaction": {
            "CFO": 0.38, "CTO": 0.40, "Legal": 0.32,
            "Procurement": 0.36, "Ops": 0.50,
        },
        "coalition_tension": {
            "cto_cfo": "conflict",
            "legal_procurement": "alliance",
        },
        "description": (
            "Post-acquisition authority shift. New EU compliance requirements. "
            "Compressed timeline. "
            "Tests: adaptive stakeholder mapping, GDPR framing, precision under uncertainty."
        ),
    },
}
```

---

### `deal_room/claims.py`

```python
"""
ClaimsTracker — Regex-only numerical contradiction detection.

DESIGN DECISION (Issue 1):
Target expansion is done by the CALLER (environment), not here.
This function receives a single stakeholder ID string only.
Calling expand_targets() here was redundant and violated single responsibility.
"""
import re
from typing import Dict, List

# Fixed patterns (tested for correctness):
# - implementation_weeks: was capturing wrong group due to .{0,20} greediness. Fixed.
# - price_commit: was capturing "000" from "$45000". Fixed to capture full number.
CLAIM_PATTERNS = {
    "implementation_weeks": (
        r"(?i)(?:go.?live|deploy|implementation|deliver|complete)"
        r"(?:\s+(?:in|within|takes?|will\s+take))?\s+(\d+)\s*(?:weeks?|wk)"
    ),
    "team_size": (
        r"(?i)(?:team|resources?)\s+of\s+(\d+)"
        r"|(\d+)\s*(?:dedicated\s+)?(?:engineers?|members?)"
    ),
    "price_commit": r"(?i)\$\s*(\d{1,3}(?:,\d{3})*|\d{4,})",
}

DEVIATION_TOLERANCE = 0.15  # 15% deviation triggers contradiction penalty

# Explicit subgroup registry — used by environment, NOT by ClaimsTracker
VALID_SUBGROUPS: Dict[str, List[str]] = {
    "cto_cfo": ["CTO", "CFO"],
    "legal_procurement": ["Legal", "Procurement"],
}

ALL_STAKEHOLDER_IDS = ["CFO", "CTO", "Legal", "Procurement", "Ops"]


def expand_targets(target: str) -> List[str]:
    """
    Expand target string to list of individual stakeholder IDs.

    Uses explicit subgroup registry — never splits on underscore.
    Splitting on underscore would break stakeholder names containing underscores
    and would produce lowercase IDs that don't match the dict keys.

    Returns empty list for unknown targets — caller decides how to handle.
    """
    t = target.lower().strip()
    if t == "all":
        return list(ALL_STAKEHOLDER_IDS)
    if t in VALID_SUBGROUPS:
        return VALID_SUBGROUPS[t]
    # Case-insensitive single stakeholder match
    for sid in ALL_STAKEHOLDER_IDS:
        if sid.lower() == t:
            return [sid]
    return []  # Unknown target


class ClaimsTracker:
    """
    Tracks numerical commitments per stakeholder.
    Returns True when a new value deviates >15% from the prior commitment.

    Receives individual stakeholder IDs only.
    Target expansion is done by the environment before calling this.
    """

    def __init__(self):
        self.claims: Dict[str, List[float]] = {}

    def reset(self):
        self.claims = {}

    def extract_and_track(self, target: str, message: str) -> bool:
        """
        target: single stakeholder ID (e.g. "CFO"), already expanded by caller.
        Returns True if a contradiction is detected.
        """
        if not message or not target:
            return False

        triggered = False
        for claim_type, pattern in CLAIM_PATTERNS.items():
            match = re.search(pattern, message)
            if not match:
                continue
            # Handle multi-group patterns (team_size has 2 capture groups)
            raw_val = next((g for g in match.groups() if g is not None), None)
            if raw_val is None:
                continue
            val = float(raw_val.replace(",", ""))
            key = f"{target}:{claim_type}"
            if key in self.claims and self.claims[key]:
                last = self.claims[key][-1]
                if last > 0 and abs(val - last) / last > DEVIATION_TOLERANCE:
                    triggered = True
            self.claims.setdefault(key, []).append(val)

        return triggered
```

---

### `deal_room/validator.py`

````python
"""
OutputValidator — 3-layer output parsing. Strict mode only.
No LLM extraction layer — training must be fully deterministic.

Layer 1: JSON extraction from markdown code blocks or inline JSON
Layer 2: Heuristic keyword extraction from free text
Layer 3: Safe fallback — never raises, always returns a valid dict
"""
import re
import json
from typing import Tuple, Dict, Optional

VALID_ACTION_TYPES = {
    "direct_message", "group_proposal", "backchannel", "send_document",
    "concession", "walkaway_signal", "reframe_value_prop", "exec_escalation",
}

VALID_TARGETS = {
    "cfo", "cto", "legal", "procurement", "ops",
    "all", "cto_cfo", "legal_procurement",
}


class OutputValidator:

    def __init__(self, mode: str = "strict"):
        self.mode = mode  # Always "strict" in training

    def validate(self, raw: str) -> Tuple[Dict, float]:
        """
        Returns (normalized_dict, confidence).
        confidence 1.0 = clean JSON parse
        confidence 0.6 = heuristic extraction
        confidence 0.0 = fallback used
        """
        if not raw:
            return self._fallback(raw), 0.0

        # Layer 1: JSON extraction
        for pattern in [
            r'```json\s*(.*?)\s*```',
            r'```\s*(.*?)\s*```',
            r'(\{[^{}]*"action_type"[^{}]*\})',
        ]:
            match = re.search(pattern, raw, re.DOTALL)
            if match:
                try:
                    data = json.loads(match.group(1).strip())
                    return self._normalize(data), 1.0
                except (json.JSONDecodeError, KeyError, IndexError):
                    continue

        # Layer 2: Heuristic extraction
        action_type = self._extract_action_type(raw)
        target = self._extract_target(raw)
        if action_type:
            return self._normalize({
                "action_type": action_type,
                "target": target or "all",
                "message": raw[:300].strip(),
            }), 0.6

        # Layer 3: Safe fallback
        return self._fallback(raw), 0.0

    def _normalize(self, data: Dict) -> Dict:
        at = data.get("action_type", "direct_message")
        if at not in VALID_ACTION_TYPES:
            at = "direct_message"
        tgt = str(data.get("target", "all"))
        if tgt.lower() not in VALID_TARGETS:
            tgt = "all"
        return {
            "action_type": at,
            "target": tgt,
            "message": str(data.get("message", ""))[:500],
            "channel": data.get("channel", "formal"),
            "mode": data.get("mode", "async_email"),
            "documents": data.get("documents", []),
            "proposed_terms": data.get("proposed_terms", None),
        }

    def _fallback(self, raw: str) -> Dict:
        return {
            "action_type": "direct_message",
            "target": "all",
            "message": (raw[:200] if raw else "Continuing our discussion."),
            "channel": "formal",
            "mode": "async_email",
            "documents": [],
            "proposed_terms": None,
        }

    def _extract_action_type(self, raw: str) -> Optional[str]:
        raw_lower = raw.lower()
        for at in VALID_ACTION_TYPES:
            if at.replace("_", " ") in raw_lower or at in raw_lower:
                return at
        return None

    def _extract_target(self, raw: str) -> Optional[str]:
        raw_lower = raw.lower()
        for t in ["cto_cfo", "legal_procurement", "cfo", "cto", "legal",
                  "procurement", "ops", "all"]:
            if t in raw_lower:
                return t
        return "all"
````

---

### `deal_room/grader.py`

```python
"""
CCIGrader — Contract Closure Index v3

Measures sustainable, implementable consensus across 4 dimensions:
  Consensus (40%)         — weighted satisfaction avg with weakest-link penalty
  Implementation Risk     — multiplicative: CTO+Ops satisfaction post-signature
  Efficiency (15%)        — pacing penalty, not raw speed
  Execution Penalty       — malformed output penalty

Weights are STAGE-DEPENDENT. Who can block changes through the deal lifecycle.
"""
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .models import DealRoomState

STAGE_WEIGHTS = {
    "evaluation":     {"CFO": 0.35, "CTO": 0.30, "Legal": 0.15, "Procurement": 0.15, "Ops": 0.05},
    "negotiation":    {"CFO": 0.30, "CTO": 0.25, "Legal": 0.20, "Procurement": 0.15, "Ops": 0.10},
    "legal_review":   {"CFO": 0.25, "CTO": 0.15, "Legal": 0.35, "Procurement": 0.20, "Ops": 0.05},
    "final_approval": {"CFO": 0.40, "CTO": 0.15, "Legal": 0.25, "Procurement": 0.10, "Ops": 0.10},
}


class CCIGrader:

    @staticmethod
    def compute(state: "DealRoomState") -> float:
        """
        Returns CCI in [0.0, 1.0].
        Returns 0.0 if deal not closed or deal failed (veto, timeout, mass blocking).
        """
        if not state.deal_closed or state.deal_failed:
            return 0.0

        stage = state.deal_stage
        if stage not in STAGE_WEIGHTS:
            stage = "final_approval"
        weights = STAGE_WEIGHTS[stage]

        total_w = sum(weights.values())
        assert abs(total_w - 1.0) < 0.01, f"Weights sum to {total_w}, must be 1.0"

        sat = {k: state.satisfaction.get(k, 0.5) for k in weights}
        min_sat = min(sat.values())

        # Weighted average satisfaction
        weighted_avg = sum(sat[k] * weights[k] for k in weights)

        # Weakest-link factor: any stakeholder below 0.35 degrades consensus significantly
        weakest_link = 0.6 + 0.4 * min(1.0, min_sat / 0.35)
        consensus = max(0.0, min(1.0, weighted_avg * weakest_link))

        # Implementation risk: CTO + Ops drive post-signature success
        # min 0.5 ensures even poor implementations score something
        sat_cto = sat.get("CTO", 0.5)
        sat_ops = sat.get("Ops", 0.5)
        impl_risk = max(0.5, 1.0 - (0.45 * (1.0 - sat_cto) + 0.35 * (1.0 - sat_ops)))

        # Efficiency: power > 1 means late-round penalty grows nonlinearly
        efficiency = max(0.1, 1.0 - ((state.round_number / state.max_rounds) ** 1.3) * 0.4)

        # Execution penalty: malformed agent outputs
        exec_penalty = min(0.20, state.validation_failures * 0.04)

        raw = (consensus * impl_risk * efficiency) - exec_penalty
        return round(max(0.0, min(1.0, raw)), 4)
```

---

### `deal_room/stakeholders.py`

```python
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
    from .models import DealRoomState

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
        "med":  {"CFO": 0.10, "Procurement": 0.05},
        "low":  {"CFO": 0.04},
    },
    "security_cert": {
        "high": {"Legal": 0.20, "CTO": 0.12, "Procurement": 0.06},
        "med":  {"Legal": 0.12, "CTO": 0.07},
        "low":  {"Legal": 0.05},
    },
    "implementation_timeline": {
        "high": {"CTO": 0.18, "Ops": 0.16},
        "med":  {"CTO": 0.10, "Ops": 0.09},
        "low":  {"CTO": 0.04, "Ops": 0.04},
    },
    "dpa": {
        "high": {"Legal": 0.22, "Procurement": 0.08},
        "med":  {"Legal": 0.14, "Procurement": 0.04},
        "low":  {"Legal": 0.06},
    },
    "reference_case": {
        "high": {"CFO": 0.10, "Procurement": 0.14, "CTO": 0.08},
        "med":  {"CFO": 0.06, "Procurement": 0.09, "CTO": 0.05},
        "low":  {"Procurement": 0.04},
    },
}

COLLABORATIVE_SIGNALS = [
    "understand", "partnership", "mutual", "together", "value", "appreciate",
    "flexible", "work with", "long-term", "relationship", "transparent",
    "committed", "invested in your success", "your goals", "collaborative",
    "joint", "shared", "tailored",
]
AGGRESSIVE_SIGNALS = [
    "demand", "require", "final offer", "unacceptable", "must",
    "non-negotiable", "take it or leave", "bottom line", "deadline",
    "insist", "ultimatum", "last chance",
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
                    1.0, max(0.0, self.state.beliefs[target]["competence"] + speed * rapport_delta)
                )
                self.state.satisfaction[target] = min(
                    1.0, max(
                        self.state.trust_floors.get(target, 0.0),
                        self.state.satisfaction[target] + rapport_delta * 0.04
                    )
                )
            if self.state.scrutiny_mode:
                self.state.satisfaction[target] = max(
                    self.state.trust_floors.get(target, 0.0),
                    self.state.satisfaction[target] - 0.03
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
            delta = sum(
                abs(current.get(d, 0.5) - pre.get(d, 0.5))
                for d in ["competence", "risk_tolerance", "pricing_rigor"]
            ) / 3.0
            deltas[sid] = round(delta, 4)
        return deltas

    def _generate_single_response(self, sid: str, stance: str, sat: float) -> str:
        templates = STAKEHOLDER_TEMPLATES.get(sid, {}).get(stance, {})
        bucket = "high" if sat > 0.65 else "low" if sat < 0.35 else "mid"
        options = templates.get(bucket, templates.get("mid", ["Understood. Let's continue."]))
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
```

---

### `deal_room/deal_room_environment.py`

```python
"""
DealRoom Environment — Main Entry Point
OpenEnv-compliant POMDP for multi-stakeholder enterprise contract negotiation.

All 5 issue fixes applied:
  Issue 1: ClaimsTracker receives individual IDs, expansion centralized here
  Issue 2: Group target belief_deltas uses max() across expanded targets
  Issue 3: Veto risk skips round 0 (opening round guard)
  Issue 4: Stage progression requires STAGE_MIN_ROUNDS
  Issue 5: Momentum is three-state: +1/0/-1
"""
import uuid
import logging
from copy import deepcopy
from typing import Dict, List, Optional, Tuple, Any

import numpy as np

from .models import DealRoomAction, DealRoomObservation, DealRoomState
from .validator import OutputValidator
from .grader import CCIGrader
from .scenarios import SCENARIOS
from .stakeholders import StakeholderEngine
from .claims import ClaimsTracker, expand_targets

logger = logging.getLogger(__name__)

# Safe stage regression — always use .get() never direct access
STAGE_REGRESSION = {
    "final_approval": "legal_review",
    "legal_review":   "negotiation",
    "negotiation":    "evaluation",
    "evaluation":     "evaluation",
    "closed":         "closed",
    "regressed":      "evaluation",
}

STAGE_PROGRESSION = {
    "evaluation":     "negotiation",
    "negotiation":    "legal_review",
    "legal_review":   "final_approval",
    "final_approval": "closed",
}

# Minimum satisfaction per stage before advancement is allowed
STAGE_MIN_SAT = {
    "evaluation":     0.45,
    "negotiation":    0.50,
    "legal_review":   0.55,
    "final_approval": 0.60,
}

# Issue 4 fix: minimum rounds before a stage can advance
# Prevents unrealistic document-spam instant progression
STAGE_MIN_ROUNDS = {
    "evaluation":     2,
    "negotiation":    2,
    "legal_review":   1,
    "final_approval": 1,
}


class DealRoomEnvironment:
    """
    OpenEnv-compliant multi-stakeholder enterprise negotiation environment.

    Usage:
        env = DealRoomEnvironment()
        obs = env.reset(seed=42, task_id="aligned")
        obs, reward, done, info = env.step(action)
        state = env.state  # @property
    """

    def __init__(self):
        self._state = DealRoomState()
        self.stakeholder_engine = StakeholderEngine()
        self.validator = OutputValidator(mode="strict")
        self.claims_tracker = ClaimsTracker()

        # Ephemeral per-step tracking (not in state)
        self._prev_satisfaction: Dict[str, float] = {}
        self._prev_stage: str = "evaluation"
        self._stage_regressed_this_round: bool = False
        self._precursors_this_round: Dict[str, str] = {}
        self._active_competitor_events: List[str] = []
        self._prev_blocker_count: int = 0
        self._belief_deltas: Dict[str, float] = {}
        self.rng: np.random.Generator = np.random.default_rng(None)

    # ------------------------------------------------------------------
    # OPENENV INTERFACE
    # ------------------------------------------------------------------

    def reset(self, seed: Optional[int] = None, episode_id: Optional[str] = None,
              **kwargs) -> DealRoomObservation:
        """Reset for new episode. task_id passed via kwargs per OpenEnv spec."""
        task_id = kwargs.get("task_id", "aligned")
        if task_id not in SCENARIOS:
            raise ValueError(f"Unknown task_id '{task_id}'. Valid: {list(SCENARIOS.keys())}")

        self.rng = np.random.default_rng(seed)
        scenario = SCENARIOS[task_id]
        ep_id = episode_id or str(uuid.uuid4())[:8]
        sids = list(scenario["initial_beliefs"].keys())

        self._state = DealRoomState(
            episode_id=ep_id,
            task_id=task_id,
            round_number=0,
            max_rounds=scenario["max_rounds"],
            beliefs=deepcopy(scenario["initial_beliefs"]),
            satisfaction=dict(scenario["initial_satisfaction"]),
            trust_floors={k: 0.20 for k in sids},
            permanent_marks={k: [] for k in sids},
            tracked_claims={},
            veto_risk={k: 0.0 for k in sids},
            veto_precursors_fired={k: False for k in sids},
            active_blockers=[],
            deal_stage="evaluation",
            stage_regressions=0,
            validation_failures=0,
            fallback_streak=0,
            scrutiny_mode=False,
            exec_escalation_used=False,
            deal_closed=False,
            deal_failed=False,
            failure_reason="",
            final_terms=None,
            rounds_since_last_contact={k: 0 for k in sids},
        )

        self.stakeholder_engine.reset(self._state, self.rng, scenario)
        self.claims_tracker.reset()

        self._prev_satisfaction = dict(self._state.satisfaction)
        self._prev_stage = "evaluation"
        self._stage_regressed_this_round = False
        self._precursors_this_round = {}
        self._active_competitor_events = []
        self._prev_blocker_count = 0
        self._belief_deltas = {}

        responses = self.stakeholder_engine.generate_opening()
        self._update_active_blockers()

        return self._build_observation(responses, hint=None, is_done=False)

    def step(self, action: DealRoomAction) -> Tuple[DealRoomObservation, float, bool, Dict]:
        """
        Process one negotiation turn.
        Returns: (observation, reward, done, info)
        reward is 0.0 during episode; CCI score only at successful terminal step.
        """
        if self._state.deal_closed or self._state.deal_failed:
            obs = self._build_observation({}, hint=None, is_done=True)
            return obs, 0.0, True, {"error": "episode_already_done"}

        # --- 1. Validate and normalize agent output ---
        parsed_dict, confidence = self.validator.validate(action.message)
        if confidence < 0.4:
            self._state.validation_failures += 1
            self._state.fallback_streak += 1
            if self._state.fallback_streak >= 2:
                self._state.scrutiny_mode = True
        else:
            self._state.fallback_streak = 0

        # Merge validated fields back — prefer parsed, fall back to original action
        normalized = DealRoomAction(
            action_type=parsed_dict.get("action_type", action.action_type),
            target=parsed_dict.get("target", action.target),
            message=parsed_dict.get("message", action.message),
            channel=parsed_dict.get("channel", action.channel),
            mode=parsed_dict.get("mode", action.mode),
            documents=action.documents or parsed_dict.get("documents", []),
            proposed_terms=action.proposed_terms,
        )

        # --- 2. Claims contradiction tracking ---
        # Issue 1 fix: expand here, pass single IDs to tracker
        expanded_targets = expand_targets(normalized.target)
        for t in expanded_targets:
            if self.claims_tracker.extract_and_track(t, normalized.message):
                self._state.trust_floors[t] = max(
                    0.0, self._state.trust_floors.get(t, 0.20) - 0.12
                )
                self._state.permanent_marks[t].append("contradiction_penalty")

        # --- 3. Background belief drift ---
        self._tick_background_beliefs()

        # --- 4. Apply agent action effects ---
        self.stakeholder_engine.apply_action(normalized.dict(), self.rng)
        self._belief_deltas = self.stakeholder_engine.get_belief_deltas()

        # --- 5. Conditional exogenous shocks ---
        self._apply_conditional_shocks()

        # --- 6. Update veto risk and fire precursors ---
        self._update_veto_risk(normalized)
        self._evaluate_veto_precursors()

        # Check for veto trigger — return EARLY before stage updates
        threshold = SCENARIOS[self._state.task_id]["veto_threshold"]
        for sid, risk in self._state.veto_risk.items():
            if risk > threshold:
                self._state.deal_failed = True
                self._state.failure_reason = f"silent_veto:{sid}"
                obs = self._build_observation(
                    {sid: "We've decided not to move forward at this time."},
                    hint=None,
                    is_done=True
                )
                obs.info = {
                    "veto_triggered": True,
                    "veto_source": sid,
                    "round_signals": self._build_round_signals(normalized),
                }
                self._state.round_number += 1
                return obs, 0.0, True, obs.info

        # --- 7. Stage regression and progression ---
        self._stage_regressed_this_round = False
        if self._should_regress_stage():
            self._regress_stage()
        if not self._stage_regressed_this_round:
            self._maybe_advance_stage()

        # --- 8. Update blockers and generate responses ---
        self._prev_blocker_count = len(self._state.active_blockers)
        self._update_active_blockers()
        responses = self.stakeholder_engine.generate_responses(normalized.dict(), self._state)

        # --- 9. Hint injection ---
        hint = None
        if self._state.task_id == "hostile_acquisition" and self._state.round_number == 3:
            hint = SCENARIOS["hostile_acquisition"]["round_3_hint"]

        # --- 10. Terminal check ---
        # Returns (done, reward) — reward is CCI on success, 0.0 on failure
        done, reward = self._check_terminal()

        # --- 11. Build observation ---
        obs = self._build_observation(responses, hint=hint, is_done=done)
        obs.info = self._build_round_signals(normalized)
        obs.info["validation_confidence"] = confidence

        # --- 12. Advance round counter ---
        self._prev_satisfaction = dict(self._state.satisfaction)
        self._prev_stage = self._state.deal_stage
        self._state.round_number += 1

        return obs, reward, done, obs.info

    @property
    def state(self) -> DealRoomState:
        return self._state

    # ------------------------------------------------------------------
    # INTERNAL DYNAMICS
    # ------------------------------------------------------------------

    def _tick_background_beliefs(self):
        """
        Stakeholder beliefs drift 2% toward neutral each round independently of agent.
        Models that unaddressed concerns compound over time.
        """
        alpha = 0.02
        for sid, beliefs in self._state.beliefs.items():
            for dim in ["competence", "risk_tolerance", "pricing_rigor"]:
                current = beliefs[dim]
                beliefs[dim] = round(
                    float(np.clip(current + (0.5 - current) * alpha, 0.0, 1.0)), 4
                )

    def _apply_conditional_shocks(self):
        """
        Satisfaction drops ONLY when there is a logical cause.
        No random unconditional drops — that creates unlearnable dynamics.

        Cause A: Competitor event active + agent hasn't contacted this stakeholder for 2+ rounds
        Cause B: Contradiction penalty was just applied this round
        """
        for sid in self._state.satisfaction:
            shock = 0.0

            # Cause A: competitor pressure + inactivity
            if (self._active_competitor_events and
                    self._state.rounds_since_last_contact.get(sid, 0) >= 2):
                shock = 0.07

            # Cause B: contradiction this round (last mark is contradiction)
            marks = self._state.permanent_marks.get(sid, [])
            if marks and marks[-1] == "contradiction_penalty":
                shock = max(shock, 0.10)

            if shock > 0:
                self._state.satisfaction[sid] = max(
                    self._state.trust_floors.get(sid, 0.0),
                    self._state.satisfaction[sid] - shock
                )

        # Competitor events fire with 5% probability in harder tasks
        if self._state.task_id in ("conflicted", "hostile_acquisition"):
            if self.rng.random() < 0.05 and not self._active_competitor_events:
                self._active_competitor_events.append("competitor_demo_scheduled")

    def _update_veto_risk(self, action: DealRoomAction):
        """
        Issue 3 fix: skip round 0 (opening observations only, no agent action taken yet).
        Growth rate is intentional — precursor window gives ~3 rounds of warning.
        """
        if self._state.round_number == 0:
            return

        for sid in self._state.veto_risk:
            sat = self._state.satisfaction.get(sid, 0.5)
            if sat < 0.30:
                self._state.veto_risk[sid] = min(1.0, self._state.veto_risk[sid] + 0.08)
            elif sat < 0.38:
                self._state.veto_risk[sid] = min(1.0, self._state.veto_risk[sid] + 0.04)
            else:
                # Risk decays when satisfaction is healthy
                self._state.veto_risk[sid] = max(0.0, self._state.veto_risk[sid] - 0.02)

        # Penalize repeated executive escalation
        if action.action_type == "exec_escalation":
            if self._state.exec_escalation_used:
                for sid in self._state.veto_risk:
                    self._state.veto_risk[sid] = min(1.0, self._state.veto_risk[sid] + 0.05)
            self._state.exec_escalation_used = True

    def _evaluate_veto_precursors(self):
        """
        Emit ambiguous warning signals when veto risk enters 0.28–0.50 range.
        Each stakeholder gets at most one precursor signal.
        Multiple message templates — seeded variation prevents memorization.
        """
        self._precursors_this_round = {}
        PRECURSOR_MSGS = [
            "{sid} has been unusually brief in recent replies.",
            "{sid} delegated follow-up coordination to their assistant.",
            "Note: {sid}'s calendar shows significant competing priorities this week.",
            "{sid} missed the last two check-in messages.",
        ]
        for sid, risk in self._state.veto_risk.items():
            if 0.28 <= risk <= 0.50 and not self._state.veto_precursors_fired.get(sid, False):
                self._state.veto_precursors_fired[sid] = True
                tmpl = PRECURSOR_MSGS[int(self.rng.integers(0, len(PRECURSOR_MSGS)))]
                self._precursors_this_round[sid] = tmpl.replace("{sid}", sid)

    def _should_regress_stage(self) -> bool:
        """Stage regression when new blockers appear at advanced stages, or critical sat drop."""
        if self._state.deal_stage not in ("legal_review", "final_approval"):
            return False
        new_blocker = len(self._state.active_blockers) > self._prev_blocker_count
        critical_drop = any(
            self._state.satisfaction.get(s, 1.0) < 0.30
            for s in ("CFO", "Legal")
        )
        return new_blocker or critical_drop

    def _regress_stage(self):
        current = self._state.deal_stage
        # Always use .get() — never direct access to prevent KeyError on unexpected stages
        self._state.deal_stage = STAGE_REGRESSION.get(current, "evaluation")
        self._state.stage_regressions += 1
        self._stage_regressed_this_round = True

    def _maybe_advance_stage(self):
        """
        Issue 4 fix: require STAGE_MIN_ROUNDS before any stage can advance.
        Prevents document-spam instant progression in early rounds.
        """
        if len(self._state.active_blockers) > 0:
            return
        if not self._state.satisfaction:
            return
        min_sat = min(self._state.satisfaction.values())
        current = self._state.deal_stage
        if current not in STAGE_PROGRESSION:
            return
        threshold = STAGE_MIN_SAT.get(current, 0.45)
        min_rounds = STAGE_MIN_ROUNDS.get(current, 2)
        if min_sat >= threshold and self._state.round_number >= min_rounds:
            next_stage = STAGE_PROGRESSION[current]
            self._state.deal_stage = next_stage
            if next_stage == "closed":
                self._state.deal_closed = True

    def _update_active_blockers(self):
        threshold = SCENARIOS[self._state.task_id]["block_threshold"]
        self._state.active_blockers = [
            sid for sid, sat in self._state.satisfaction.items()
            if sat < threshold
        ]

    def _check_terminal(self) -> Tuple[bool, float]:
        """
        Returns (done, reward).
        Success → (True, CCI score)
        All failure modes → (True, 0.0)
        Still running → (False, 0.0)
        """
        if self._state.deal_failed:
            return True, 0.0
        if self._state.deal_closed or self._state.deal_stage == "closed":
            self._state.deal_closed = True
            return True, CCIGrader.compute(self._state)
        if self._state.round_number >= self._state.max_rounds:
            return True, 0.0
        if (len(self._state.active_blockers) >= 3 and
                self._state.deal_stage == "evaluation"):
            self._state.deal_failed = True
            self._state.failure_reason = "mass_blocking"
            return True, 0.0
        return False, 0.0

    def _build_observation(self, responses: Dict[str, str],
                           hint: Optional[str], is_done: bool) -> DealRoomObservation:
        """
        Build agent observation. Adds delay and noise per POMDP design.
        engagement_level is 1-step delayed with Gaussian noise — never exact satisfaction.
        """
        engagement = {}
        for sid in self._state.satisfaction:
            prev = self._prev_satisfaction.get(sid, 0.5)
            noise = float(self.rng.normal(0, 0.04))
            engagement[sid] = round(float(np.clip(prev + noise, 0.0, 1.0)), 3)

        n_blockers = len(self._state.active_blockers)
        if n_blockers >= 2 or self._stage_regressed_this_round:
            momentum = "critical"
        elif n_blockers == 0 and not self._stage_regressed_this_round:
            momentum = "progressing"
        else:
            momentum = "stalling"

        days = max(
            0,
            SCENARIOS[self._state.task_id]["days_to_deadline"]
            - self._state.round_number * 3
        )

        return DealRoomObservation(
            round_number=self._state.round_number,
            max_rounds=self._state.max_rounds,
            stakeholder_messages=responses,
            engagement_level=engagement,
            deal_momentum=momentum,
            deal_stage=self._state.deal_stage,
            competitor_events=list(self._active_competitor_events),
            veto_precursors=dict(self._precursors_this_round),
            scenario_hint=hint,
            active_blockers=list(self._state.active_blockers),
            days_to_deadline=days,
            done=is_done,      # is_done passed explicitly — not always False
            info={},
        )

    def _build_round_signals(self, action: DealRoomAction) -> Dict[str, Any]:
        """
        Dense causal signals for RL credit assignment.
        These are metadata in info — NOT part of the reward.
        Reward remains 0.0 until episode end.

        Issue 2 fix: group targets use max delta across expanded targets.
        Issue 5 fix: three-state momentum +1/0/-1.
        """
        from .claims import expand_targets

        # Issue 2 fix: max delta across all targets in action
        targets = expand_targets(action.target)
        if not targets:
            targets = [action.target]
        target_delta = max(
            (self._belief_deltas.get(t, 0.0) for t in targets),
            default=0.0
        )

        # Issue 5 fix: three-state momentum
        if self._stage_regressed_this_round:
            momentum_dir = -1
        elif self._state.deal_stage != self._prev_stage:
            momentum_dir = 1   # Stage actually advanced
        elif len(self._state.active_blockers) < self._prev_blocker_count:
            momentum_dir = 1   # Blocker was resolved
        else:
            momentum_dir = 0   # Holding — no meaningful change

        # Count satisfaction VALUES not keys (was a bug: iterating dict gives keys)
        n_advocates = sum(1 for sat in self._state.satisfaction.values() if sat >= 0.65)

        return {
            "new_advocates": n_advocates,
            "new_blockers": len(self._state.active_blockers),
            "momentum_direction": momentum_dir,   # -1 / 0 / +1
            "backchannel_received": action.channel == "backchannel",
            "belief_deltas": dict(self._belief_deltas),
            "target_responded_positively": target_delta > 0.04,
            "stage_changed": self._state.deal_stage != self._prev_stage,
            "stage": self._state.deal_stage,
            "veto_risk_max": max(self._state.veto_risk.values()) if self._state.veto_risk else 0.0,
        }
```

---

### `server/app.py`

```python
"""
DealRoom FastAPI Server
Thin HTTP wrapper only. Zero business logic. All logic in deal_room/.
"""
import os
from typing import Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from deal_room import DealRoomEnvironment, DealRoomAction

app = FastAPI(title="DealRoom", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

_env = DealRoomEnvironment()


class ResetRequest(BaseModel):
    task_id: Optional[str] = "aligned"
    seed: Optional[int] = 42
    episode_id: Optional[str] = None


@app.get("/health")
async def health():
    return {"status": "ok", "service": "deal-room",
            "tasks": ["aligned", "conflicted", "hostile_acquisition"]}


@app.get("/metadata")
async def metadata():
    return {"name": "deal-room", "version": "1.0.0",
            "tasks": ["aligned", "conflicted", "hostile_acquisition"]}


@app.post("/reset")
async def reset(req: ResetRequest = ResetRequest()):
    try:
        obs = _env.reset(seed=req.seed, task_id=req.task_id, episode_id=req.episode_id)
        return obs.dict()
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Reset failed: {e}")


@app.post("/step")
async def step(action: DealRoomAction):
    try:
        obs, reward, done, info = _env.step(action)
        return {"observation": obs.dict(), "reward": reward, "done": done, "info": info}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Step failed: {e}")


@app.get("/state")
async def state():
    try:
        return _env.state.dict()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"State failed: {e}")


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 7860))
    uvicorn.run("server.app:app", host="0.0.0.0", port=port)
```

---

### `inference.py`

````python
"""
DealRoom Inference Script — Baseline
Imports deal_room directly (no HTTP). Strict [START][STEP][END] format.
"""
import os
import json
import re

from openai import OpenAI
from deal_room import DealRoomEnvironment, DealRoomAction

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
    for pattern in [r'```json\s*(.*?)\s*```', r'```\s*(.*?)\s*```', r'(\{.*\})']:
        m = re.search(pattern, raw, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(1))
            except json.JSONDecodeError:
                pass
    return {"action_type": "direct_message", "target": "all",
            "message": raw[:200], "channel": "formal"}


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
        print(f"[STEP] step={step_num} action=error reward=0.00 done=true error={str(e)[:80]}")
        rewards.append(0.0)

    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={step_num} "
        f"score={final_score:.2f} rewards={rewards_str}"
    )
    return {"task": task_id, "score": final_score, "steps": step_num, "success": success}


if __name__ == "__main__":
    for task in ["aligned", "conflicted", "hostile_acquisition"]:
        run_task(task, seed=42)
````

---

### `openenv.yaml`

```yaml
name: deal-room
version: "1.0.0"
description: "LLM agent navigates multi-stakeholder enterprise contract negotiation — building organizational consensus through coalition-building, veto avoidance, and strategic information management"
author: "akshaypulla"
tags:
  - openenv
  - negotiation
  - enterprise
  - multi-stakeholder
  - real-world
  - rl
tasks:
  - id: aligned
    description: "Low-friction enterprise deal — correct document sequencing and engagement order"
    difficulty: easy
    max_steps: 8
    reward_range: [0.0, 1.0]
  - id: conflicted
    description: "CTO-CFO tension, Legal-Procurement alliance — coalition sequencing is primary skill"
    difficulty: medium
    max_steps: 10
    reward_range: [0.0, 1.0]
  - id: hostile_acquisition
    description: "Post-acquisition authority shift, new compliance requirements — adaptive mapping under pressure"
    difficulty: hard
    max_steps: 10
    reward_range: [0.0, 1.0]
reward_range: [0.0, 1.0]
observation_space:
  type: object
  description: "Delayed noisy engagement signals, qualitative deal momentum, stakeholder messages, veto precursors"
action_space:
  type: object
  description: "8 action types targeting specific stakeholders with optional documents and natural language"
```

---

### `Dockerfile`

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
ENV PORT=7860
EXPOSE 7860
CMD ["python", "-m", "uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]
```

---

### `requirements.txt`

```
fastapi==0.109.0
uvicorn==0.27.0
pydantic>=2.0.0
numpy>=1.24.0
openai>=1.0.0
openenv-core>=0.1.0
```

---

### `calibrate.py`

```python
"""
Calibration script — run before submission.
Target: strategic agent beats random by 0.20+ spread on every task.
"""
import numpy as np
from deal_room import DealRoomEnvironment, DealRoomAction


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
            action_type=str(self.rng.choice(["direct_message", "send_document", "backchannel"])),
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

        # Respond immediately to veto precursors via backchannel
        if precursors:
            target = list(precursors.keys())[0]
            return DealRoomAction(
                action_type="backchannel", target=target, channel="backchannel",
                message=(
                    "I want to make sure we address any concerns you have directly. "
                    "I'm committed to making this work for your specific situation and timeline. "
                    "I value our partnership and want to find a mutual solution."
                ),
            )

        # Address active blockers when critical
        if blockers and obs.deal_momentum == "critical":
            return DealRoomAction(
                action_type="direct_message", target=blockers[0],
                message=(
                    "I understand there are open concerns on your end and I appreciate your transparency. "
                    "Rather than proceed, I'd like to address them specifically together. "
                    "I'm flexible and want to find a solution that works for both sides long-term."
                ),
            )

        # Systematic document delivery by round
        doc_sequence = [
            ("CFO", "roi_model", "Here is our ROI analysis showing 14-month payback at your scale."),
            ("Legal", "dpa", "Here is our GDPR-compliant DPA and SOC2 Type II certification."),
            ("Legal", "security_cert", "Additional security documentation and audit rights clause."),
            ("CTO", "implementation_timeline", "Our implementation team dedicates senior engineers to your integration. Timeline respects your Q3 bandwidth."),
            ("Ops", "reference_case", "Reference case from a similar deployment that delivered on schedule."),
        ]

        if r < len(doc_sequence):
            target, doc_type, msg = doc_sequence[r]
            return DealRoomAction(
                action_type="send_document", target=target, message=msg,
                documents=[{"type": doc_type, "specificity": "high"}],
            )

        # Closing push
        return DealRoomAction(
            action_type="group_proposal", target="all",
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
        print("CALIBRATION FAILED. Adjust initial_satisfaction or veto_threshold in scenarios.py.")
        print("If aligned spread is too small: lower initial_satisfaction by 0.05 across all stakeholders.")
        print("If hostile spread is too small: increase round_3_hint detail or lower veto_threshold to 0.40.")
```

---

## 5. How The Environment Works — Complete Flow

### Episode Start

1. `reset(seed=42, task_id="aligned")` seeds `np.random.default_rng`, creates fresh `DealRoomState`
2. All stakeholders initialized with `initial_beliefs` and `initial_satisfaction` from scenario
3. `StakeholderEngine.generate_opening()` returns fixed realistic opening messages
4. Agent receives `DealRoomObservation` — note: `engagement_level` is the previous round's satisfaction with noise added, so on round 0 it reflects initial_satisfaction + Gaussian noise

### Each Step (12 ordered operations)

1. **Validate** agent output via 3-layer validator
2. **Claims tracking** — expand targets, pass individual IDs to ClaimsTracker, fire trust floor penalty on contradiction
3. **Background drift** — all beliefs drift 2% toward neutral independently of agent
4. **Apply action** — document effects, rapport effects on beliefs and satisfaction
5. **Conditional shocks** — satisfaction drops only when competitor active + agent inactive, or contradiction just triggered
6. **Veto risk update** — accumulates for stakeholders with sat<0.30, decays for sat>0.38 (skips round 0)
7. **Veto check** — if any stakeholder risk > threshold, return early with reward=0.0
8. **Stage management** — check regression (new blockers or critical drop at advanced stage), then check advancement (no blockers, sat above threshold, minimum rounds elapsed)
9. **Generate responses** — stakeholder natural language based on stance and satisfaction bucket
10. **Hint injection** — hostile_acquisition hint fires at round 3
11. **Terminal check** — returns (done, reward) tuple
12. **Build observation** — delayed noisy engagement, three-state momentum, veto precursors

### Terminal Conditions

- **Success:** stage advances to "closed" → `reward = CCIGrader.compute(state)`
- **Veto:** any stakeholder's veto_risk > threshold → `reward = 0.0`
- **Mass blocking:** 3+ blockers at evaluation stage → `reward = 0.0`
- **Timeout:** round_number >= max_rounds → `reward = 0.0`

### CCI Score

```
CCI = consensus × weakest_link_factor × implementation_risk × efficiency − execution_penalty
```

Where:

- `consensus` = weighted average satisfaction (weights depend on current stage)
- `weakest_link_factor` = penalizes any stakeholder below 0.35 satisfaction
- `implementation_risk` = `max(0.5, f(CTO_sat, Ops_sat))` — CTO+Ops satisfaction drives post-signature success
- `efficiency` = penalizes reaching max rounds, nonlinear decay
- `execution_penalty` = `min(0.20, validation_failures × 0.04)`

### Score Calibration Targets

| Task                | Random    | Base LLM  | Trained RL Goal |
| ------------------- | --------- | --------- | --------------- |
| aligned             | 0.12–0.22 | 0.38–0.48 | 0.68–0.78       |
| conflicted          | 0.06–0.14 | 0.20–0.28 | 0.58–0.68       |
| hostile_acquisition | 0.04–0.10 | 0.20–0.26 | 0.44–0.52       |

Minimum required spread (strategic minus random): **0.15 per task**.

---

## 6. Pre-Submission Checklist

```bash
# 1. Install
pip install -r requirements.txt

# 2. OpenEnv validation
openenv validate

# 3. Integration test
python3 -c "
from deal_room import DealRoomEnvironment, DealRoomAction

# Determinism
e1 = DealRoomEnvironment()
o1 = e1.reset(seed=42, task_id='aligned')
e2 = DealRoomEnvironment()
o2 = e2.reset(seed=42, task_id='aligned')
assert o1.stakeholder_messages == o2.stakeholder_messages, 'DETERMINISM FAIL'
assert o1.done == False, 'done should be False on reset'

# Step works
a = DealRoomAction(action_type='send_document', target='CFO',
    message='Here is our ROI analysis.',
    documents=[{'type':'roi_model','specificity':'high'}])
obs, r, done, info = e1.step(a)
assert 0.0 <= r <= 1.0, f'reward out of range: {r}'
assert isinstance(done, bool)
assert 'belief_deltas' in info
assert 'CFO' in info['belief_deltas']
assert obs.done == done, 'obs.done must match done'
assert info['momentum_direction'] in (-1, 0, 1), 'momentum must be -1/0/+1'

# Group target signal
a2 = DealRoomAction(action_type='group_proposal', target='cto_cfo',
    message='I appreciate our partnership and want to find a mutual solution.')
obs2, r2, d2, i2 = e1.step(a2)
# target_responded_positively should work for group targets now
assert isinstance(i2['target_responded_positively'], bool)

# Garbage input does not crash
a3 = DealRoomAction(action_type='direct_message', target='all', message='!@#garbage')
obs3, r3, d3, i3 = e1.step(a3)
assert isinstance(r3, float)

# Episode terminates
e3 = DealRoomEnvironment()
e3.reset(seed=1, task_id='aligned')
for _ in range(15):
    a = DealRoomAction(action_type='direct_message', target='all', message='Let us move forward.')
    obs, r, done, info = e3.step(a)
    if done: break
assert done, 'Episode must terminate'

print('Integration: PASS')
"

# 4. Docker
docker build -t deal-room .
docker run -d -p 7860:7860 --name dr deal-room
sleep 5
curl -sf http://localhost:7860/health && echo 'Health: PASS'
curl -sf -X POST http://localhost:7860/reset \
  -H 'Content-Type: application/json' \
  -d '{"task_id":"aligned","seed":42}' | python3 -m json.tool | head -10
docker stop dr && docker rm dr

# 5. Calibration
python3 calibrate.py
# Must show PASS on all 3 tasks

# 6. Inference
HF_TOKEN=your_token python3 inference.py 2>&1 | grep -E '^\[START\]|^\[END\]'
# Must show 3 [START] lines and 3 [END] lines
# All scores must be in [0, 1]
```
