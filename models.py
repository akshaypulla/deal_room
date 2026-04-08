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
    channel: str = "formal"  # formal | backchannel
    mode: str = "async_email"  # async_email | formal_meeting | exec_escalation


class DealRoomObservation(BaseModel):
    round_number: int = 0
    max_rounds: int = 10
    stakeholder_messages: Dict[str, str] = Field(default_factory=dict)
    engagement_level: Dict[str, float] = Field(default_factory=dict)
    # Noisy 1-step delayed proxy for satisfaction. Never exact.
    deal_momentum: str = "stalling"  # stalling | progressing | critical
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
