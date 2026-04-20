# DELIBERATION_AND_CURRICULUM.md
# DealRoom v3 — Deliberation Protocol & Self-Improving Curriculum

> **Scope:** Committee deliberation engine and adaptive curriculum specification.
> Covers the two-layer deliberation architecture, curriculum failure taxonomy,
> CurriculumGenerator algorithm, four-phase training loop, and self-play arms race.
> The developer implementing `committee/deliberation_engine.py` and
> `curriculum/adaptive_generator.py` reads this document.

---

## 1. Deliberation Architecture: Two Layers

The deliberation engine has two completely separate layers. They must be built
and tested independently. Layer 2 failing must never break Layer 1.

```
Layer 1 — Computation (pure Python)
  Input:  G, beliefs_before_action, beliefs_after_vendor_action
  Output: updated_beliefs (B_i(t+Δ) for all stakeholders)
  Method: propagate_beliefs() linear equation
  Speed:  microseconds
  Tests:  test_propagation_direction, test_damping_prevents_runaway
  Status: REQUIRED — training breaks if this is wrong

Layer 2 — Rendering (one MiniMax call per vendor turn)
  Input:  beliefs_before, beliefs_after (from Layer 1), G summary
  Output: 2-3 turn summary dialogue string
  Method: minimax_call()
  Speed:  ~150ms
  Status: DISPLAY ONLY — training is unaffected if this returns empty string
  Use:    Demo visualization, HF blog, judge Q&A panel
```

**Rule:** Every function in Layer 1 has a test. Layer 2 has no tests — it fails gracefully with empty string.

---

## 2. Layer 1 — Computation Protocol (Step by Step)

### Per-Round Deliberation Sequence

```
ROUND t:
  1. Vendor sends action a_t targeting stakeholder k

  2. Bayesian update (belief_tracker.py):
     B_k(t+1) ∝ P(a_t | type) · B_k(t)      ← ONLY stakeholder k updated
     B_j(t)   unchanged for all j ≠ k

  3. Deliberation engine runs (this document):
     for step in range(N_STEPS[scenario]):
       for i in stakeholders:
         B_i += Σ_j w_ji · ΔB_j             ← propagate_beliefs()
     → Produces B_i(t+Δ) for all i

  4. build_observation() runs (observation_mechanism.md):
     Uses B_i(t+Δ) to populate all five observation signals
     Vendor agent sees round t+1 observation

  5. Reward scorer runs (utterance_reward_spec.md):
     Scores message a_t against state_before / state_after / G_true
```

### N_STEPS by Scenario

```python
N_DELIBERATION_STEPS = {
    "aligned":             3,   # sparse graph, few propagation paths
    "conflicted":          3,   # two-cluster, limited cross-cluster propagation
    "hostile_acquisition": 4,   # dense graph, longer propagation chains needed
}
```

**Why 3–4 and not more:** With N=3, paths of length up to 3 hops are observable (j→i→k→l). In a 7-node graph, almost all relevant paths are ≤ 3 hops. N=5+ produces only marginal additional propagation but increases computation and makes the observation harder to learn from (too many cascades happening simultaneously).

### Core Implementation

```python
def propagate_beliefs(
    graph: CausalGraph,
    beliefs_before_action: Dict[str, BeliefDistribution],
    beliefs_after_action: Dict[str, BeliefDistribution],
    n_steps: int = 3
) -> Dict[str, BeliefDistribution]:
    """
    See CAUSAL_COMMITTEE_DYNAMICS.md Section 3 for full implementation.
    Reproduced here for reference.

    beliefs_before_action: B_i(t) for all i (snapshot before any update)
    beliefs_after_action:  B_i(t) with ONLY targeted stakeholder updated

    Returns: B_i(t+Δ) for all i after N propagation steps.
    """
    current = {sid: b.copy() for sid, b in beliefs_after_action.items()}

    for step in range(n_steps):
        next_beliefs = {sid: b.copy() for sid, b in current.items()}

        for dest in graph.nodes:
            influencers = graph.get_influencers(dest)
            if not influencers:
                continue

            total_delta = sum(
                w * (current[src].positive_mass() - beliefs_before_action[src].positive_mass())
                for src, w in influencers.items()
            )

            if abs(total_delta) > 0.005:
                next_beliefs[dest] = _apply_belief_delta(
                    belief=current[dest],
                    delta=total_delta,
                    damping=0.85 ** step
                )

        current = next_beliefs

    return current
```

---

## 3. Layer 2 — Rendering Protocol

### The One MiniMax Call

```python
class CommitteeDeliberationEngine:

    def generate_summary(
        self,
        beliefs_before: Dict[str, BeliefDistribution],
        beliefs_after: Dict[str, BeliefDistribution],
        targeted_stakeholder: str,
        episode_rng: np.random.Generator
    ) -> str:
        """
        Generate 2-3 turn plausible committee dialogue for demo/logging.
        Called AFTER propagate_beliefs() has already updated beliefs.
        Output does NOT affect any state. Fails silently.

        Selects the two non-targeted stakeholders whose beliefs changed most.
        Prompts MiniMax to write a brief realistic dialogue between them.
        """
        deltas = {
            sid: abs(
                beliefs_after[sid].positive_mass() -
                beliefs_before[sid].positive_mass()
            )
            for sid in beliefs_after
            if sid != targeted_stakeholder
        }

        if not deltas or max(deltas.values()) < 0.03:
            return ""  # No meaningful deliberation occurred — skip rendering

        # Pick top two stakeholders by delta magnitude
        top_two = sorted(deltas, key=lambda x: deltas[x], reverse=True)[:2]
        if len(top_two) < 2:
            return ""

        s1, s2 = top_two[0], top_two[1]
        d1 = beliefs_after[s1].positive_mass() - beliefs_before[s1].positive_mass()
        d2 = beliefs_after[s2].positive_mass() - beliefs_before[s2].positive_mass()

        sentiment1 = "cautiously positive" if d1 > 0.05 else ("concerned" if d1 < -0.05 else "neutral")
        sentiment2 = "cautiously positive" if d2 > 0.05 else ("concerned" if d2 < -0.05 else "neutral")

        prompt = f"""Two committee members briefly discuss the vendor after their latest communication.
{s1} ({ARCHETYPES[s1].role}) is currently {sentiment1} about the vendor.
{s2} ({ARCHETYPES[s2].role}) is currently {sentiment2} about the vendor.
Write 2-3 turns of realistic internal discussion (no vendor present).
Under 80 words total. Do not invent facts not stated."""

        try:
            return minimax_call(prompt, max_tokens=100, temperature=0.8)
        except Exception:
            return ""  # Always fail silently
```

### Demo Visualization Integration

The summary string populates the "Committee Deliberation Heat" panel in the live demo. The panel shows:
1. The two stakeholders in dialogue (nodes in the graph visualization)
2. A brief arrow animation between them showing information flow
3. The 2-3 turn summary dialogue text

This panel is **visible to judges only** — the vendor agent never sees it. This is the design choice that makes the demo compelling: judges observe the hidden committee dynamics that the agent must infer.

---

## 4. Self-Improving Curriculum

### Why Adaptive Curriculum

A static scenario set (`aligned/conflicted/hostile_acquisition`) produces agents that overfit to specific configurations. An adaptive curriculum generates episodes at the **frontier of agent capability** — hard enough to push improvement, easy enough to provide learning signal.

The committee's deliberation engine doubles as an automatic adversary. When the agent discovers that targeting Finance first cascades to Procurement (because `w_Finance→Procurement` is high in many sampled graphs), the curriculum response is to generate future scenarios where this pattern is disrupted — lower `w_Finance→Procurement`, higher `w_Legal→Procurement` — forcing the agent to discover a new approach.

### Failure Mode Taxonomy

| ID | Failure mode | Diagnostic signal | Root cause |
|----|-------------|-------------------|-----------|
| F1 | CVaR veto despite positive expected outcome | Terminal outcome = 'veto'; `r^risk` consistently < 0.30 in final 5 rounds | Agent treats deal as value-maximization; ignores tail-risk signals |
| F2 | Trust collapse mid-episode | `r^trust` drops sharply (>0.20) in rounds 7–10 | Agent uses aggressive tactics after initial progress; relationship not sustained |
| F3 | Failed graph inference | `r^causal` stays in [0.15, 0.30] throughout episode | Engagement_delta correlations not being integrated; agent not using history |
| F4 | Timeout without coalition formation | Low `r^info` (< 0.25) throughout; terminal = 'timeout' | Agent not asking probing questions; stuck on known constraints |
| F5 | Single-dimension reward hacking | One dimension > 0.75, others < 0.30 | Agent found a partial shortcut (e.g., document spam, question flooding) |
| F6 | Authority shift blindness | `r^causal` drops sharply after round 8 in hostile_acquisition | Agent not detecting that G changed; using stale inferred graph |

### CurriculumGenerator Algorithm

```python
@dataclass
class FailureAnalysis:
    failure_modes: Dict[str, float]  # failure_id -> frequency in recent batch
    worst_graph_configs: List[GraphConfig]  # G configs that caused most failures
    worst_cvar_configs: List[CVaRConfig]    # τ_i configs that caused most failures
    agent_capability_estimate: float        # 0-1 estimate of current agent level

class AdaptiveCurriculumGenerator:

    def __init__(self, config: CurriculumConfig = CurriculumConfig()):
        self.config = config
        self._scenario_pool: List[ScenarioConfig] = []
        self._difficulty_distribution = [0.20, 0.60, 0.20]  # easy, frontier, hard

    def analyze_failures(
        self,
        trajectories: List[EpisodeTrajectory]
    ) -> FailureAnalysis:
        """
        Analyze last B episodes to identify systematic failure patterns.
        B = config.analysis_batch_size (default 10 episodes)
        """
        failure_counts = defaultdict(int)
        for traj in trajectories:
            failure_counts.update(self._detect_failures(traj))

        # Normalize to frequencies
        total = len(trajectories)
        failure_modes = {k: v/total for k, v in failure_counts.items()}

        # Find which graph configs caused F1 and F3 failures
        worst_graphs = self._identify_worst_configs(
            trajectories, target_failures=['F1', 'F3']
        )

        # Find which CVaR configs caused F1 and F2 failures
        worst_cvars = self._identify_worst_cvar_configs(
            trajectories, target_failures=['F1', 'F2']
        )

        # Estimate agent capability from recent reward distribution
        recent_rewards = [t.total_reward for t in trajectories[-5:]]
        capability = float(np.mean(recent_rewards))

        return FailureAnalysis(
            failure_modes=failure_modes,
            worst_graph_configs=worst_graphs,
            worst_cvar_configs=worst_cvars,
            agent_capability_estimate=capability
        )

    def _detect_failures(self, traj: EpisodeTrajectory) -> Dict[str, int]:
        """Detect which failure modes occurred in a single episode."""
        failures = {}

        # F1: CVaR veto
        if traj.terminal_outcome == 'veto':
            failures['F1'] = 1

        # F2: Trust collapse — look for r^trust drop of > 0.20 in rounds 7-10
        trust_scores = traj.reward_by_dimension['trust']
        if len(trust_scores) >= 10:
            early_trust = np.mean(trust_scores[:5])
            mid_trust   = np.mean(trust_scores[6:10])
            if early_trust - mid_trust > 0.20:
                failures['F2'] = 1

        # F3: Failed graph inference — r^causal stays flat
        causal_scores = traj.reward_by_dimension['causal']
        if len(causal_scores) >= 10:
            early_causal = np.mean(causal_scores[:5])
            late_causal  = np.mean(causal_scores[-5:])
            if late_causal - early_causal < 0.10:  # Less than 0.10 improvement
                failures['F3'] = 1

        # F4: Timeout
        if traj.terminal_outcome == 'timeout':
            info_scores = traj.reward_by_dimension['info']
            if np.mean(info_scores) < 0.25:
                failures['F4'] = 1

        # F5: Single-dimension hacking
        means = {dim: np.mean(scores) for dim, scores in traj.reward_by_dimension.items()}
        max_dim = max(means.values())
        min_dim = min(means.values())
        if max_dim > 0.75 and min_dim < 0.25:
            failures['F5'] = 1

        return failures

    def generate_harder_scenarios(
        self,
        analysis: FailureAnalysis,
        n_scenarios: int = 50
    ) -> List[ScenarioConfig]:
        """
        Generate n_scenarios new scenarios targeting the identified failure modes.
        Maintains difficulty distribution: 20% easy, 60% frontier, 20% very hard.
        """
        new_scenarios = []

        # F1 response: tighten CVaR thresholds for legal/compliance
        if analysis.failure_modes.get('F1', 0) > 0.30:
            for _ in range(int(n_scenarios * 0.25)):
                new_scenarios.append(self._generate_tight_cvar_scenario(
                    tighten_roles=['Legal', 'Compliance'],
                    tau_reduction=0.03  # reduce threshold by 0.03
                ))

        # F3 response: sample denser G with less predictable centrality
        if analysis.failure_modes.get('F3', 0) > 0.30:
            for _ in range(int(n_scenarios * 0.25)):
                new_scenarios.append(self._generate_ambiguous_graph_scenario())

        # F2 response: increase stakeholder baseline skepticism
        if analysis.failure_modes.get('F2', 0) > 0.30:
            for _ in range(int(n_scenarios * 0.20)):
                new_scenarios.append(self._generate_skeptical_prior_scenario())

        # F5 response: add adversarial stakeholder that detects single-strategy exploits
        if analysis.failure_modes.get('F5', 0) > 0.20:
            for _ in range(int(n_scenarios * 0.15)):
                new_scenarios.append(self._generate_adversarial_stakeholder_scenario())

        # Fill remainder with standard scenarios at frontier difficulty
        while len(new_scenarios) < n_scenarios:
            new_scenarios.append(self._generate_frontier_scenario(
                capability=analysis.agent_capability_estimate
            ))

        return new_scenarios[:n_scenarios]

    def maintain_difficulty_distribution(
        self,
        new_scenarios: List[ScenarioConfig]
    ) -> List[ScenarioConfig]:
        """
        Ensure 20% easy, 60% frontier, 20% hard in the scenario pool.
        Prevents catastrophic forgetting of easy scenarios.
        If agent learned to handle aligned scenarios, it must keep handling them.
        """
        easy     = [s for s in new_scenarios if s.difficulty == 'easy']
        frontier = [s for s in new_scenarios if s.difficulty == 'frontier']
        hard     = [s for s in new_scenarios if s.difficulty == 'hard']

        target_total = len(new_scenarios)
        balanced = (
            easy[:int(target_total * 0.20)] +
            frontier[:int(target_total * 0.60)] +
            hard[:int(target_total * 0.20)]
        )

        return balanced
```

---

## 5. Four-Phase Self-Play Training Loop

```python
def run_four_phase_training(
    model: PreTrainedModel,
    env: DealRoomV3,
    curriculum_gen: AdaptiveCurriculumGenerator,
    n_phases: int = 4,
    episodes_per_phase: int = 50,
    group_size: int = 4              # GRPO group size
) -> TrainingResult:
    """
    Four-phase loop from arXiv 2603.10476 (Learning to Negotiate).

    Phase 1: GRPO training against current scenario pool
    Phase 2: Committee adaptation to agent's discovered strategies
    Phase 3: Curriculum hardening based on Phase 1 failures
    Phase 4: Repeat with harder committee configs

    This creates an arms race between agent capability and environment difficulty.
    """
    all_metrics = []

    for phase in range(n_phases):
        print(f"\n=== Phase {phase+1}/{n_phases} ===")

        # PHASE 1: GRPO Training
        print(f"Phase {phase+1}.1: GRPO training ({episodes_per_phase} episodes)...")
        phase_trajectories = []
        for episode in range(episodes_per_phase):
            trajectory = run_grpo_episode(model, env, group_size)
            phase_trajectories.append(trajectory)
            metrics = compute_episode_metrics(trajectory)
            all_metrics.append(metrics)

            if episode % 10 == 0:
                print(f"  Episode {episode}: r^causal={metrics.r_causal_mean:.3f}, "
                      f"r^goal={metrics.r_goal_mean:.3f}, "
                      f"terminal={metrics.terminal_outcome}")

        # PHASE 2: Committee Adaptation
        print(f"Phase {phase+1}.2: Committee adapting to agent strategies...")
        env.committee_engine.adapt_to_agent_strategies(
            trajectories=phase_trajectories,
            adaptation_strength=0.1  # subtle adaptation, not adversarial jump
        )

        # PHASE 3: Curriculum Analysis and Hardening
        print(f"Phase {phase+1}.3: Analyzing failures and generating harder scenarios...")
        failure_analysis = curriculum_gen.analyze_failures(phase_trajectories)
        print(f"  Failure frequencies: {failure_analysis.failure_modes}")

        new_scenarios = curriculum_gen.generate_harder_scenarios(
            analysis=failure_analysis,
            n_scenarios=50
        )
        balanced_scenarios = curriculum_gen.maintain_difficulty_distribution(new_scenarios)
        env.update_scenario_pool(balanced_scenarios)

        print(f"  Generated {len(balanced_scenarios)} new scenarios for phase {phase+2}")
        print(f"  Agent capability estimate: {failure_analysis.agent_capability_estimate:.3f}")

    return TrainingResult(
        all_metrics=all_metrics,
        final_pareto_efficiency=compute_pareto_efficiency(all_metrics[-20:]),
        r_causal_curve=[m.r_causal_mean for m in all_metrics],
        r_goal_curve=[m.r_goal_mean for m in all_metrics],
    )
```

### Committee Adaptation to Agent Strategies

```python
def adapt_to_agent_strategies(
    self,
    trajectories: List[EpisodeTrajectory],
    adaptation_strength: float = 0.10
) -> None:
    """
    Subtle committee adaptation: when agent discovers a pattern that reliably
    produces high r^causal (e.g., Finance is always the hub), strengthen
    Finance↔Procurement edge in future G samples to make that pattern harder.

    This is NOT adversarial training — it's a natural arms race.
    The committee becomes "more sophisticated" in response to vendor patterns,
    just as real committees adapt when they recognize vendor strategies.

    adaptation_strength=0.10: only small adjustments per phase.
    Prevents the committee from overadapting in a single phase.
    """
    # Find the agent's dominant targeting strategy
    target_frequencies = defaultdict(int)
    for traj in trajectories:
        for action in traj.actions:
            if action.target_ids:
                target_frequencies[action.target_ids[0]] += 1

    total_actions = sum(target_frequencies.values())
    if total_actions == 0:
        return

    dominant_target = max(target_frequencies, key=target_frequencies.get)
    dominance = target_frequencies[dominant_target] / total_actions

    if dominance > 0.50:  # Agent targeting one stakeholder > 50% of the time
        # Increase the weight of edges AWAY from dominant target
        # This makes the committee more resilient to single-node strategies
        self._strengthen_edges_from(dominant_target, delta=adaptation_strength)
        print(f"  Committee adapting: strengthening edges from {dominant_target} "
              f"(targeted {dominance:.0%} of turns)")
```

---

## 6. Lookahead Tool

The agent can optionally call `simulate()` before acting to get predicted belief updates and response previews.

```python
@dataclass
class LookaheadRequest:
    depth: int = 2           # turns to simulate forward
    n_hypotheses: int = 2    # K mental state hypotheses (from ToMAgent)

@dataclass
class SimulationResult:
    predicted_responses: Dict[str, str]      # stakeholder_id -> predicted text
    predicted_belief_deltas: Dict[str, float] # predicted engagement changes
    cvar_impact: Dict[str, float]             # predicted CVaR change per stakeholder
    graph_information_gain: float             # predicted entropy reduction about G
    cost: float = 0.07                        # subtracted from r^goal for this turn

class LookaheadSimulator:

    def simulate(
        self,
        action_draft: DealRoomAction,
        current_state: DealRoomState,
        n_hypotheses: int = 2,
        depth: int = 2
    ) -> SimulationResult:
        """
        K=2 mental state hypotheses (from ToMAgent arXiv 2509.22887).
        Minimax robustness: pick action that performs best across both hypotheses.
        Cost: 0.07 subtracted from r^goal for using lookahead.
        """
        # Generate K=2 competing belief hypotheses for targeted stakeholders
        hypotheses = self._generate_hypotheses(action_draft.target_ids[0], current_state)

        # Simulate under each hypothesis
        simulation_results = []
        for hypothesis in hypotheses:
            sim_state = current_state.with_belief_override(
                action_draft.target_ids[0], hypothesis
            )
            predicted = self._simulate_one_hypothesis(action_draft, sim_state, depth)
            simulation_results.append(predicted)

        # Minimax: take the result from the worst-case hypothesis
        # This is conservative — the agent picks the action that is
        # most robust to uncertainty about the stakeholder's true belief
        worst_case = min(simulation_results, key=lambda x: x.predicted_goal_delta)

        return SimulationResult(
            predicted_responses=worst_case.responses,
            predicted_belief_deltas=worst_case.belief_deltas,
            cvar_impact=worst_case.cvar_impact,
            graph_information_gain=worst_case.graph_info_gain,
            cost=0.07
        )

    def _generate_hypotheses(
        self,
        target_stakeholder: str,
        state: DealRoomState
    ) -> List[BeliefDistribution]:
        """
        Generate K=2 competing hypotheses about target's current belief.
        Based on the observable behavioral history and current weak signals.
        """
        current_belief = state.agent_belief_estimates[target_stakeholder]

        # Hypothesis 1: Optimistic interpretation (vendor appears competent)
        h1 = current_belief.with_positive_shift(delta=0.15)

        # Hypothesis 2: Pessimistic interpretation (vendor appears uncertain)
        h2 = current_belief.with_negative_shift(delta=0.15)

        return [h1, h2]
```

---

## 7. Required Tests

```python
# tests/test_deliberation_engine.py

def test_layer1_independent_of_layer2():
    """Deliberation Layer 1 must work when Layer 2 is unavailable."""
    engine = CommitteeDeliberationEngine(graph=test_graph, n_steps=3)
    beliefs_before = create_neutral_beliefs(test_stakeholders)
    beliefs_after_action = dict(beliefs_before)
    beliefs_after_action["Finance"] = apply_positive_delta(beliefs_before["Finance"], 0.3)

    # Simulate Layer 2 failure
    with mock.patch.object(engine, '_generate_summary', return_value=""):
        result = engine.run(
            vendor_action=create_action("send_document", ["roi_model"], target="Finance"),
            beliefs_before_action=beliefs_before,
            beliefs_after_vendor_action=beliefs_after_action,
            render_summary=True
        )

    assert result.updated_beliefs is not None
    assert result.summary_dialogue == ""
    # Layer 1 output should be valid
    for sid, b in result.updated_beliefs.items():
        assert abs(sum(b.distribution.values()) - 1.0) < 1e-6, f"{sid} belief not normalized"


def test_four_phase_loop_runs_without_crash():
    """Four-phase training loop must complete 20 episodes without crashing."""
    model = load_tiny_test_model()
    env = DealRoomV3(config=TestConfig(max_rounds=5))  # short episodes for testing
    curriculum = AdaptiveCurriculumGenerator()

    result = run_four_phase_training(
        model=model,
        env=env,
        curriculum_gen=curriculum,
        n_phases=2,
        episodes_per_phase=5
    )

    assert len(result.all_metrics) == 10
    assert len(result.r_causal_curve) == 10


def test_curriculum_generates_target_count():
    """Curriculum generator must produce exactly n_scenarios scenarios."""
    gen = AdaptiveCurriculumGenerator()
    analysis = create_test_failure_analysis(failure_modes={"F1": 0.50, "F3": 0.30})
    scenarios = gen.generate_harder_scenarios(analysis, n_scenarios=50)
    assert len(scenarios) == 50


def test_difficulty_distribution_maintained():
    """After balancing, should be approximately 20/60/20 easy/frontier/hard."""
    gen = AdaptiveCurriculumGenerator()
    scenarios = [create_random_scenario() for _ in range(100)]
    balanced = gen.maintain_difficulty_distribution(scenarios)

    easy_count     = sum(1 for s in balanced if s.difficulty == 'easy')
    frontier_count = sum(1 for s in balanced if s.difficulty == 'frontier')
    hard_count     = sum(1 for s in balanced if s.difficulty == 'hard')

    assert 15 <= easy_count <= 25, f"Easy count {easy_count} not near 20%"
    assert 55 <= frontier_count <= 65, f"Frontier count {frontier_count} not near 60%"
    assert 15 <= hard_count <= 25, f"Hard count {hard_count} not near 20%"
```

---

## 8. Colab Notebook Structure

```python
# training/grpo_colab.ipynb

# Cell 1: Setup
# Install openenv, trl, transformers, peft, bitsandbytes, minimax_sdk

# Cell 2: Load policy model
# Qwen2.5-3B-Instruct, 4-bit QLoRA, rank-16

# Cell 3: Initialize environment
# DealRoomV3 + CommitteeDeliberationEngine + UtteranceScorer

# Cell 4: GRPO config
# G=4, max_new_tokens=256, lr=1e-5, temperature=0.9

# Cell 5: Run four-phase training (50 total episodes per phase = 200 total)
# Logs: episode reward, terminal outcome, r^causal, r^goal each episode

# Cell 6: Plot learning curves
# Five separate curves: r^goal, r^trust, r^info, r^risk, r^causal
# One curve: Pareto efficiency score
# All plotted against episode number
# KEY VISUALIZATION: r^causal from ~0.18 (episode 1) to ~0.65 (episode 50)

# Cell 7: Baseline comparison
# Untrained agent: run 10 episodes on hostile_acquisition
# Trained agent: run 10 episodes on hostile_acquisition
# Show: Pareto efficiency improvement, terminal outcome distribution

# Cell 8: Export trained model
# Save LoRA adapters to HF Hub
```

---

*Implementation order: `committee/deliberation_engine.py` → `curriculum/adaptive_generator.py`*
*Layer 1 must be tested before Layer 2 is built*
*Four-phase training loop lives in `training/grpo_trainer.py`*
*Colab notebook lives in `training/grpo_colab.ipynb`*
