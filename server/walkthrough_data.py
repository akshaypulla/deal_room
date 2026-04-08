"""Curated walkthrough data for the DealRoom OpenEnv custom tab."""

GUIDE_DATA = {
    "task": "conflicted",
    "seed": 64,
    "title": "Conflicted Committee Walkthrough",
    "summary": (
        "A seeded medium-difficulty run that shows approval-chain management, "
        "constraint discovery, coalition effects, and a feasible close."
    ),
    "steps": [
        {
            "index": 0,
            "title": "Map The Buying Committee",
            "concept": "Start with visible blockers and approval-chain ambiguity.",
            "action": None,
            "explanation": (
                "The conflicted task opens with three blockers and no revealed constraints. "
                "This is a partially observable negotiation problem, so the right move is not to close early "
                "but to sequence evidence around the most fragile approvers."
            ),
            "counterfactual": (
                "If you push to close here, mandatory stakeholders take a permanent impatience mark and the deal "
                "becomes harder to recover."
            ),
            "judge_highlights": [
                "Three blockers are visible before any proposal is safe.",
                "No hard constraint is known yet, so feasibility is still uncertain.",
                "The environment starts in evaluation, not final approval.",
            ],
        },
        {
            "index": 1,
            "title": "Finance First",
            "concept": "Clear the highest-leverage commercial blocker with a relevant artifact.",
            "action": {
                "action_type": "send_document",
                "target": "finance",
                "target_ids": ["finance"],
                "message": "Here is the ROI model with explicit payback assumptions and downside cases.",
                "documents": [{"type": "roi_model", "specificity": "high"}],
            },
            "explanation": (
                "The baseline starts by reducing finance resistance with ROI evidence. "
                "That does not close the deal, but it narrows uncertainty and removes one source of committee drag."
            ),
            "counterfactual": (
                "Sending a random artifact here wastes a turn and leaves the commercial concern unresolved."
            ),
            "judge_highlights": [
                "Dense reward comes from satisfying a real stakeholder request.",
                "Blockers shrink, but the episode is still far from terminal.",
                "The move is document-specific, not keyword-matched generic chatter.",
            ],
        },
        {
            "index": 2,
            "title": "Legal Unblocks Stage Progression",
            "concept": "Sequencing matters: legal review cannot progress without review-ready compliance artifacts.",
            "action": {
                "action_type": "send_document",
                "target": "legal_compliance",
                "target_ids": ["legal_compliance"],
                "message": "Here is the DPA with GDPR-aligned privacy commitments and review-ready clauses.",
                "documents": [{"type": "dpa", "specificity": "high"}],
            },
            "explanation": (
                "Providing the DPA to legal removes a formal blocker and lets the deal move from evaluation "
                "toward negotiation. This is a stage-dependent environment: the same move later would be less valuable."
            ),
            "counterfactual": (
                "Ignoring legal often looks harmless for one round, but the deal cannot close if the approval path is incomplete."
            ),
            "judge_highlights": [
                "Stage advancement is explicit and reward-bearing.",
                "Mandatory approvers must be handled before closure is possible.",
                "The deal can regress if blockers are mishandled later.",
            ],
        },
        {
            "index": 3,
            "title": "Operations Reveals The Hidden Constraint",
            "concept": "Ambiguous signals become a known hard constraint only after the right evidence arrives.",
            "action": {
                "action_type": "send_document",
                "target": "operations",
                "target_ids": ["operations"],
                "message": "Here is the implementation timeline with milestones, owners, and delivery guardrails.",
                "documents": [{"type": "implementation_timeline", "specificity": "high"}],
            },
            "explanation": (
                "This turn is where the hidden delivery-window constraint becomes known. "
                "The environment moves from soft concern modeling into hard feasibility gating."
            ),
            "counterfactual": (
                "If you propose commercial terms before this timeline constraint is known, the deal can look aligned while still being infeasible."
            ),
            "judge_highlights": [
                "Constraint state transitions from hidden to known.",
                "The judge lens can now show confidence rising on a specific feasibility issue.",
                "This is where partial observability becomes visible to the judge.",
            ],
        },
        {
            "index": 4,
            "title": "Procurement Completes The Process Path",
            "concept": "Not every blocker is about product or price; process compliance matters too.",
            "action": {
                "action_type": "send_document",
                "target": "procurement",
                "target_ids": ["procurement"],
                "message": "Here is the supplier onboarding packet including process, insurance, and vendor details.",
                "documents": [{"type": "vendor_packet", "specificity": "high"}],
            },
            "explanation": (
                "Procurement concerns are not negotiable away with relationship-building alone. "
                "This step demonstrates hard process constraints that must be satisfied before final approval."
            ),
            "counterfactual": (
                "A strong commercial message still fails if supplier onboarding or vendor review is incomplete."
            ),
            "judge_highlights": [
                "Process blockers are separate from sentiment blockers.",
                "The environment models different stakeholder utility functions.",
                "By now the committee is operationally and legally cleaner, not just happier.",
            ],
        },
        {
            "index": 5,
            "title": "Close Only When Feasibility Is Green",
            "concept": "A good close is earned through resolved constraints and approval readiness.",
            "action": {
                "action_type": "group_proposal",
                "target": "all",
                "target_ids": ["finance", "legal_compliance", "operations", "procurement"],
                "message": "I believe we have enough alignment to move to final approval on concrete, reviewable terms.",
                "proposed_terms": {
                    "price": 180000,
                    "timeline_weeks": 14,
                    "security_commitments": ["gdpr", "audit rights"],
                    "support_level": "named_support_lead",
                    "liability_cap": "mutual_cap",
                },
            },
            "explanation": (
                "This final proposal closes successfully because the committee is ready, the approval path is complete, "
                "and the terms satisfy the known constraints."
            ),
            "counterfactual": (
                "The same group proposal two or three turns earlier would have triggered a premature-close penalty instead of a high terminal score."
            ),
            "judge_highlights": [
                "Terminal grading rewards durable, feasible closure rather than just positive sentiment.",
                "The final score combines approval completeness, constraint satisfaction, trust durability, and efficiency.",
                "This is why dense reward is informative but not a shortcut to success.",
            ],
        },
    ],
}
