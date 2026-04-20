# DealRoom v3 — Product Requirements Document

## Problem Statement

Enterprise B2B sales negotiations fail not because individual stakeholders reject proposals, but because **buying committees are decentralized systems that coordinate autonomously between vendor interactions**. A vendor can make progress with Finance, only to find Legal has already been briefed with concerns that didn't exist before the vendor touched base. No existing RL environment models this committee coordination dynamic — treating committees instead as independent NPCs or independent LLMs that don't coordinate.

The problem is that current negotiation environments are either:
1. **Scripted NPCs** — agents learn scripts, not negotiation strategy
2. **Independent LLMs** — non-stationary, non-evaluable, can't gradient through them
3. **Flat multi-agent systems** (Sotscope, LLM-Deliberation) — agents don't coordinate between turns, no inter-agent influence graph

DealRoom v3 solves this by modeling the buying committee as a **Decentralized Partially Observable Markov Decision Process (Dec-POMDP)** with a **hidden causal influence graph G** that deliberates autonomously between vendor interactions.

---

## Solution

DealRoom v3 is a research-grade OpenEnv RL environment for enterprise B2B negotiation where:

1. **The buying committee is an autonomous multi-agent system** that deliberates between vendor touchpoints, updating beliefs through a hidden causal influence graph G
2. **CVaR-aware Bayesian stakeholders** produce tail-risk veto dynamics that fire independently of expected utility (a stakeholder can veto despite positive expected outcome)
3. **Five-dimensional utterance-level rewards** are mathematically proven non-hackable — no single strategy dimension can be optimized in isolation
4. **The vendor agent must infer G** from behavioral correlations and exploit high-centrality nodes

---

## User Stories

### Committee Dynamics and Causal Inference

1. As a vendor agent, I want to infer the hidden causal influence graph G from stakeholder behavioral correlations, so I can target high-centrality nodes that cascade influence across the committee
2. As a vendor agent, I want to observe engagement_level_delta signals, so I can detect which stakeholders are being influenced by my actions on other stakeholders
3. As a vendor agent, I want to receive cross_stakeholder_echoes, so I can infer information propagation pathways within the committee
4. As a vendor agent, I want weak_signals that probabilistically fire based on belief delta magnitude, so I have noisy but actionable evidence about hidden committee dynamics
5. As a vendor agent, I want veto_precursors warnings, so I can address CVaR concerns before they trigger a veto
6. As a vendor agent, I want engagement_history (5-round sliding window), so I can detect temporal patterns in committee sentiment shifts

### Stakeholder Belief Modeling

7. As a vendor agent, I want stakeholder responses conditioned on their Bayesian belief state, so I can predict how they will react to my messages
8. As a vendor agent, I want belief updates to propagate through G during committee deliberation, so targeting one stakeholder cascades to influence others
9. As a vendor agent, I want non-targeted stakeholder updates to be dampened (0.3x), so I can't directly observe the full effect of my actions on uninvolved parties
10. As a vendor agent, I want CVaR-aware veto dynamics, so stakeholders with high tail-risk sensitivity can veto even when expected utility is positive
11. As a vendor agent, I want observable signals of stakeholder risk profiles, so I can infer τ_i (veto thresholds) and λ_i (risk weights) over time

### Reward System

12. As a training system, I want five separate reward dimensions logged independently, so I can analyze learning signals across goal, trust, information, risk, and causal dimensions
13. As a training system, I want utterance-level rewards scored per step, so I have dense learning signal rather than only episode-terminal rewards
14. As a training system, I want r^causal computed deterministically from betweenness_centrality, so there is a stable, non-gamingable signal for graph inference
15. As a training system, I want utterance scoring cached on hash(message + stakeholder_id + belief_state_hash), so repeated evaluations don't trigger redundant LLM calls
16. As a training system, I want prediction_accuracy logged as a diagnostic metric (not added to rewards), so I can monitor world-model quality without reward hacking
17. As a training system, I want a mathematically proven non-hackability property, so any high-reward strategy must genuinely navigate multi-stakeholder dynamics

### Lookahead Tool

18. As a vendor agent, I want a simulate() lookahead tool that runs depth-2 simulations with K=2 mental state hypotheses, so I can select actions robust to uncertainty about stakeholder beliefs
19. As a vendor agent, I want lookahead to cost 0.07 from r^goal when used, so lookahead is a strategic decision rather than always-free information
20. As a vendor agent, I want minimax robustness selection across hypotheses, so I choose actions that perform well even in worst-case belief scenarios

### Training and Curriculum

21. As a training system, I want GRPO with group-relative advantage computation across 5 reward dimensions, so multi-dimensional strategies are properly credited
22. As a training system, I want a four-phase self-play loop (GRPO → committee adaptation → curriculum analysis → harder scenarios), so the committee co-evolves with the agent
23. As a training system, I want failure mode detection (F1-F6), so curriculum hardening targets specific agent weaknesses
24. As a training system, I want difficulty distribution maintenance (20% easy / 60% frontier / 20% hard), so catastrophic forgetting is prevented
25. As a training system, I want r^causal learning curve from ~0.18 to ~0.65 over 50 episodes, so graph inference learning is demonstrably occurring

### Environment Interface

26. As an OpenEnv-compatible environment, I want reset(seed, task_id) returning DealRoomObservation, so existing OpenEnv tooling works unchanged
27. As an OpenEnv-compatible environment, I want step(action) returning (observation, reward, done, info), so standard RL interfaces are supported
28. As an OpenEnv-compatible environment, I want the internal DealRoomState to contain hidden G, B_i(t), and τ_i, so the agent never directly observes committee internals

---

## Implementation Decisions

### Module Architecture

**Phase 1: Committee Foundations** (No LLM, Pure Python)
- `committee/causal_graph.py`: CausalGraph dataclass, sample_graph() for 3 scenario types, propagate_beliefs() with N=3 deliberation steps, _apply_belief_delta() with damping=0.85^step, compute_behavioral_signature() for identifiability testing, get_betweenness_centrality() for r^causal
- `committee/belief_tracker.py`: BeliefDistribution dataclass, bayesian_update() with likelihood table and non-targeted damping=0.3, compute_engagement_level()
- `committee/deliberation_engine.py`: CommitteeDeliberationEngine with Layer 1 (propagate_beliefs, pure Python) and Layer 2 (_generate_summary, MiniMax call, fails silently)

**Phase 2: Stakeholder Models** (No LLM)
- `stakeholders/archetypes.py`: 6 archetype profiles (Legal, Finance, TechLead, Procurement, Operations, ExecSponsor) with exact α, τ, λ values per locked spec
- `stakeholders/cvar_preferences.py`: compute_cvar() over empirical distribution, compute_outcome_distribution() with 4 uncertainty domains per archetype, evaluate_deal() returning (expected_utility, cvar_loss), check_veto_trigger()

**Phase 3: Reward System** (LLM Required)
- `rewards/utterance_scorer.py`: UtteranceScore dataclass (goal, trust, info, risk, causal, prediction_accuracy), UtteranceScorer with MiniMax integration and caching, LOOKAHEAD_COST=0.07 applied to r^goal, prediction_accuracy logged as diagnostic only
- `rewards/pareto_efficiency.py`: check_pareto_optimality(), compute_terminal_reward() with outcomes (+1.0, 0.0, -0.5, -0.75, -1.0)

**Phase 4: Environment** (LLM Required)
- `environment/dealroom_v3.py`: DealRoomV3 OpenEnv wrapper with reset() (samples G, initializes B_i(0) and CVaR profiles, initializes noisy engagement accumulators), step() (executes action → Bayesian updates → deliberation → veto check → scoring → observation build), _build_observation() implementing all 5 observable signals
- `environment/lookahead.py`: LookaheadSimulator with K=2 hypothesis generation (optimistic +0.15, pessimistic -0.15 on positive_mass), depth=2 simulation, minimax robustness selection

**Phase 5: Training**
- `curriculum/adaptive_generator.py`: AdaptiveCurriculumGenerator with failure mode taxonomy (F1-F6), analyze_failures(), generate_adaptive_scenario(), difficulty distribution maintenance (20/60/20)
- `training/grpo_trainer.py`: GRPOTrainer with group-relative advantage per dimension, EpisodeTrajectory with 5D rewards, run_training_loop() with verbose logging

### Schema Changes (models.py)

**DealRoomObservation additions:**
- `engagement_level_delta: Optional[float]` — noisy delta since last round (single float, first stakeholder)
- `engagement_history: List[Dict[str, float]]` — list of {sid, levels} dicts with 5-round sliding window
- `cross_stakeholder_echoes: List[Dict[str, str]]` — list of {from, to, content} for 70% probabilistic recall

**DealRoomAction additions:**
- `lookahead: Optional[LookaheadRequest]` — when present, triggers simulation and applies LOOKAHEAD_COST

**LookaheadRequest schema:**
- `depth: int = 2` — number of turns to simulate
- `n_hypotheses: int = 2` — K mental state hypotheses

### Key Invariants (Must Never Break)

1. G is never in DealRoomObservation — only in DealRoomState (hidden)
2. B_i(t) raw distributions are never in DealRoomObservation — only engagement levels
3. τ_i CVaR thresholds are never in DealRoomObservation — only veto_precursors (generic warnings at 70% of τ_i)
4. Episode reset regenerates G, B_i(0), and τ_i from scratch — no warm-starting
5. Utterance scorer caches on hash(message + stakeholder_id + belief_state_hash)
6. All 5 reward dimensions logged separately throughout training — never collapsed to scalar before analysis
7. Lookahead costs 0.07 from r^goal — no prediction accuracy bonus in any reward dimension
8. Deliberation Layer 1 is pure Python — runs in microseconds
9. Deliberation Layer 2 fails silently — returns empty string on any error
10. Engagement levels are accumulated from noisy deltas — never computed fresh from B_i

### Locked Parameters

```
ENGAGEMENT_NOISE_SIGMA = 0.03
ECHO_RECALL_PROBABILITY = 0.70
WEAK_SIGNAL_HARD_THRESHOLD = 0.12
WEAK_SIGNAL_SOFT_LOWER = 0.08
WEAK_SIGNAL_SOFT_PROBABILITY = 0.70
VETO_WARNING_THRESHOLD_RATIO = 0.70
ENGAGEMENT_HISTORY_WINDOW = 5

N_DELIBERATION_STEPS = {"aligned": 3, "conflicted": 3, "hostile_acquisition": 4}
PROPAGATION_DAMPING_BASE = 0.85

REWARD_WEIGHTS = {"goal": 0.25, "trust": 0.20, "info": 0.20, "risk": 0.20, "causal": 0.15}
TERMINAL_WEIGHT = 2.0
LOOKAHEAD_COST = 0.07
LOOKAHEAD_DEPTH = 2
LOOKAHEAD_HYPOTHESES = 2

GRPO_GROUP_SIZE = 4
GRPO_MAX_NEW_TOKENS = 256
GRPO_LEARNING_RATE = 1e-5
POLICY_MODEL = "Qwen/Qwen2.5-3B-Instruct"
LORA_RANK = 16
```

### CVaR Profiles (Locked)

| Archetype | α | τ | λ |
|-----------|---|---|---|
| Legal | 0.95 | 0.10 | 0.70 |
| Finance | 0.90 | 0.15 | 0.50 |
| TechLead | 0.80 | 0.25 | 0.30 |
| Procurement | 0.85 | 0.20 | 0.45 |
| Operations | 0.80 | 0.30 | 0.35 |
| ExecSponsor | 0.70 | 0.40 | 0.25 |

### API Contracts

**reset(seed, task_id) → DealRoomObservation**
- Samples G from scenario-conditioned prior
- Initializes B_i(0) per scenario prior
- Initializes CVaR profiles per archetype
- Initializes noisy engagement accumulators: {sid: 0.5 + N(0, 0.03)}
- Initializes engagement history: {sid: [initial_value] * 5}
- Sets deal_stage = 'evaluation', round = 0
- Returns observation with NO G, B_i, τ_i exposed

**step(action) → (DealRoomObservation, float, bool, Dict)**
- If action.lookahead: runs LookaheadSimulator, returns SimulationResult (does NOT advance state)
- Runs bayesian_update for targeted stakeholder (is_targeted=True)
- Runs bayesian_update for non-targeted stakeholders (is_targeted=False, damping=0.3)
- Runs committee deliberation engine (Layer 1 propagation + Layer 2 rendering)
- Checks veto triggers for ALL stakeholders post-deliberation
- Calls build_observation() to construct agent-visible observation
- Scores utterance across 5 dimensions
- Applies LOOKAHEAD_COST if action.lookahead was used
- Updates deal_stage, round_number
- Returns StepResult

---

## Testing Decisions

### What Makes a Good Test

- Tests external behavior only, not implementation details
- Tests module interfaces in isolation (deep module pattern)
- Uses property-based testing for mathematical properties (e.g., belief propagation boundedness)
- Statistical tests for probabilistic behavior (weak signals, echo recall)
- Integration tests verify end-to-end episode flow

### Phase 1 Tests: Committee Foundations

**test_causal_graph.py:**
- `test_propagation_direction`: Positive delta for targeted stakeholder propagates positively to neighbors (A→B edge weight 0.7, target A with delta=0.4, assert B.positive_mass > before + 0.05)
- `test_damping_prevents_runaway`: In fully-connected dense graph (weight=0.8), beliefs stay in (0, 1) after 5 propagation steps
- `test_graph_identifiability`: 20 sampled graphs produce pairwise distinguishable behavioral signatures (statistical test, p < 0.05)
- `test_authority_invariant`: ExecSponsor has >= 2 outgoing edges in every scenario type (30 graphs tested)
- `test_centrality_hub_higher_than_leaves`: In star graph, hub centrality > all leaf centralities
- `test_no_self_edges`: Graph has no self-edges
- `test_weight_range`: All edge weights in [0.05, 0.95]
- `test_behavioral_signature_distinct`: Targeting different stakeholders in same graph produces different signatures (pairwise distance > 0.02)

**test_belief_tracker.py:**
- `test_bayesian_update_concentrates_belief`: 10 consistent competent actions → positive_mass > 0.70
- `test_non_targeted_weaker_than_targeted`: Non-targeted update delta < targeted update delta × 0.4
- `test_likelihood_values_in_range`: All P(action|type) in [0.10, 0.95]
- `test_distribution_normalizes`: BeliefDistribution sums to 1.0 ± 1e-6 after any update
- `test_damping_factor_applied`: Non-targeted update uses 0.3x damping numerically verified
- `test_positive_mass_increases_on_competent_action`: send_document(DPA) to Legal increases Legal's positive_mass
- `test_negative_mass_increases_on_deceptive_action`: exec_escalation before round 5 increases targeted stakeholder's deceptive mass

### Phase 2 Tests: Stakeholder Models

**test_cvar_preferences.py:**
- `test_cvar_veto_fires_despite_positive_expected_utility`: Core test — Legal with deal without DPA, E[u] > 0 but veto_triggered == True
- `test_cvar_does_not_fire_with_full_documentation`: After providing DPA + security_cert + adequate liability_cap, veto_triggered == False for Legal
- `test_cvar_decreases_monotonically_with_documentation`: Adding documentation reduces CVaR, never increases it
- `test_risk_profile_ordering`: CVaR ordering Legal > Procurement > Finance > Operations > TechLead > ExecSponsor for same deal terms
- `test_lambda_risk_weight`: High λ stakeholder (Legal, 0.70) more influenced by CVaR than low λ stakeholder (TechLead, 0.30)
- `test_veto_status_never_exposes_tau`: VetoStatus.reason does not contain numeric τ value
- `test_outcome_distribution_sums_to_one`: Probabilities sum to 1.0

### Phase 3 Tests: Reward System

**test_utterance_scorer.py:**
- `test_causal_score_deterministic`: Same action + same graph returns identical r^causal
- `test_cache_returns_same_score`: Identical (message, action, state) returns cache hit
- `test_cache_hit_rate_after_repeated_calls`: After 10 calls with 3 unique inputs, cache hit rate > 0.0
- `test_high_centrality_target_scores_higher`: Targeting hub node scores r^causal > leaf node + 0.20 in star graph
- `test_lookahead_cost_subtracted`: Action with lookahead has r^goal exactly 0.07 lower
- `test_lookahead_cost_not_below_zero`: r^goal never negative even if base score < 0.07
- `test_all_dimensions_in_range`: All 5 dimensions in [0.0, 1.0] for 20 random inputs
- `test_prediction_accuracy_not_in_reward`: weighted_sum() does not include prediction_accuracy
- `test_parse_failure_returns_neutral`: Malformed MiniMax JSON returns neutral scores, not crash
- `test_risk_score_for_dpa_send_to_legal`: DPA to Legal with elevated CVaR → r^risk > 0.60
- `test_risk_score_near_zero_for_wrong_stakeholder`: DPA to TechLead → r^risk < 0.25

### Phase 4 Tests: Environment

**test_observation_mechanism.py:**
- `test_g_never_in_observation`: DealRoomObservation has no 'graph' or 'causal_graph' attribute
- `test_engagement_not_cancellable`: |inferred_delta - true_delta| > 0.02 (noise not cancelled by subtraction)
- `test_engagement_history_length`: Exactly 5 values per stakeholder after step 1
- `test_weak_signal_probabilistic_near_threshold`: Belief delta in [0.08, 0.12] fires ~70% over 200 trials (55% < rate < 85%)
- `test_weak_signal_never_fires_below_threshold`: Belief delta < 0.08 → weak_signals empty always
- `test_weak_signal_always_fires_above_threshold`: Belief delta > 0.12 → weak_signals always fires
- `test_echo_recall_rate`: 60-80% detection rate over 200 trials, autocorrelation < 0.05
- `test_veto_precursor_fires_for_non_targeted`: When agent targets Finance with risky terms, Legal's veto_precursors fires if CVaR crosses 70% of τ_Legal
- `test_reset_clears_all_state`: After reset(), accumulators and history re-initialized, two resets produce independent observations
- `test_episode_seed_reproducibility`: Same seed → identical observation sequences

**test_end_to_end.py:**
- `test_full_episode_runs_without_crash`: reset + 5 steps + terminal state, all 5 reward dimensions non-zero
- `test_reward_vector_all_five_dimensions`: reward dict has goal, trust, info, risk, causal keys all in [0.0, 1.0]
- `test_veto_terminates_episode`: Veto → done=True, terminal_reward=-1.0
- `test_lookahead_does_not_advance_state`: lookahead action → SimulationResult, round_number unchanged
- `test_lookahead_reduces_goal_score`: Same action with/without lookahead differs by exactly 0.07 on r^goal

### Phase 5 Tests: Curriculum

**test_curriculum.py:**
- `test_failure_detection_f1`: F1 detected when veto fires AND r^risk consistently low
- `test_failure_detection_f3`: F3 detected when r^causal does not improve by 0.10 over episode
- `test_generates_exact_n_scenarios`: generate_adaptive_scenario called n times returns exactly n scenarios
- `test_difficulty_distribution_maintained`: After 100 generations, 15-25% easy, 55-65% frontier, 15-25% hard
- `test_f5_hacking_detected`: One dimension > 0.75 and another < 0.25 → F5 flagged

### Integration Tests

**test_integration.py:**
- `test_openenv_validate_passes`: openenv validate passes with zero errors
- `test_smoke_test_passes`: container_route_smoke.py exits 0
- `test_grpo_training_20_episodes`: 20 GRPO episodes runs without OOM or crash, r^causal in episode 20 differs from episode 1 by > 0.05
- `test_v2_baseline_not_regressed`: v3 observation is a superset of v2 (same top-level keys present)
- `test_all_five_reward_dimensions_logged`: After one full episode, metrics contain r_goal_mean, r_trust_mean, r_info_mean, r_risk_mean, r_causal_mean separately
- `test_g_never_observable`: 100 steps across 5 episodes, no observation field named 'graph', 'causal_graph', 'G', or 'edge_weights'

---

## Out of Scope

- **HuggingFace Space visualization** — three-panel visualization (causal graph, reward sparklines, deliberation heat) is deferred until all tests pass
- **GRPO training on GPU** — actual QLoRA training on Qwen2.5-3B is deferred; GRPOTrainer runs self-play with random policy for testing
- **GRPO Colab notebook** — full training notebook with 50 episodes is deferred; basic trainer structure exists for testing
- **V2 compatibility shim** — backwards compatibility layer in server/deal_room_environment.py is deferred
- **Web UI updates** — Gradio custom components for v3 visualization are deferred
- **Inference pipeline updates** — system prompt updates for v3 capabilities in inference.py are deferred
- **OpenEnv validation tooling** — openenv.yaml task parameter updates are deferred
- **Container deployment** — Dockerfile updates for v3 are deferred

---

## Further Notes

### Critical Research Claim

The central research contribution is `test_cvar_veto_fires_despite_positive_expected_utility`: if this test passes, it demonstrates that CVaR-aware stakeholders can veto deals with positive expected utility, which is the core Dec-POMDP dynamic that makes this environment novel. This is the first RL negotiation environment where "making the deal better for everyone" is insufficient — the agent must also manage tail-risk distributions.

### Identifiability Theorem

The environment is learnable because of Theorem 3.1: if the agent targets stakeholder i and stakeholder j ≠ i shows a correlated response in the next round, then with probability ≥ 1 - δ there exists a directed path from i to j in G. This is testable via `test_graph_identifiability` — if 20 graphs can't be distinguished by their behavioral signatures, the environment cannot be learned.

### Non-Hackability Guarantee

The five-dimensional reward is proven non-hackable because:
- Maximizing r^goal via concessions → r^trust and r^info penalize capitulation without reciprocal information
- Maximizing r^trust via agreeableness → r^goal and r^causal penalize generic messages
- Maximizing r^info via diagnostic questions → r^risk penalizes over-questioning as uncertainty introduction
- Scripting a fixed sequence → r^causal fails because G changes each episode
- Maximum achievable R_episode via single-dimension optimization ≈ 0.40, well below 0.80 for genuine multi-dimensional strategy

### Absolute Invariants Summary

These 10 invariants must pass all tests at all times. Any test failure indicates a violation:

1. G never in DealRoomObservation
2. B_i(t) raw distributions never in DealRoomObservation  
3. τ_i never in DealRoomObservation
4. Episode reset regenerates everything from scratch
5. Utterance scoring caches on exact hash
6. All 5 reward dimensions logged separately
7. Lookahead costs exactly 0.07 from r^goal
8. Deliberation Layer 1 is pure Python (no LLM calls)
9. Deliberation Layer 2 fails silently (returns "")
10. Engagement levels are accumulated noisy deltas, not fresh computations

### Implementation Status

**COMPLETE:**
- committee/causal_graph.py — full implementation
- committee/belief_tracker.py — full implementation
- committee/deliberation_engine.py — full implementation
- stakeholders/archetypes.py — full implementation (6 profiles)
- stakeholders/cvar_preferences.py — full implementation
- rewards/utterance_scorer.py — full implementation with caching
- rewards/pareto_efficiency.py — full implementation
- environment/dealroom_v3.py — full environment with all 5 signals
- environment/lookahead.py — full implementation with K=2 hypotheses
- curriculum/adaptive_generator.py — full implementation
- training/grpo_trainer.py — full implementation

**MISSING (Priority):**
- All test files in deal_room/tests/ — ZERO tests exist
- grpo_colab.ipynb — training notebook not created
- server/app.py update — not pointed to dealroom_v3
- OpenEnv validation — not verified
- inference.py update — system prompt not updated for v3

### Estimated Build Completion

- **Day 1**: Write all 7 test files (test_causal_graph, test_belief_tracker, test_cvar_preferences, test_utterance_scorer, test_observation_mechanism, test_end_to_end, test_curriculum, test_integration)
- **Day 2**: Fix all failing tests, achieve 100% pass rate
- **Day 3**: Create grpo_colab.ipynb, verify openenv validate passes
- **Day 4**: Update server/app.py, inference.py, deploy to HF Space
- **Day 5**: Run 50-episode training demo, generate learning curves
- **Day 6**: Write HuggingFace blog, final polish
