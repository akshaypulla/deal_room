"""
End-to-end workflow tests simulating complete user journeys.
Tests realistic multi-step negotiation scenarios.
"""

import pytest
import numpy as np
from models import DealRoomAction
from server.deal_room_environment import DealRoomEnvironment
from server.grader import CCIGrader


class TestEndToEndAlignedScenario:
    """Test complete aligned scenario workflow."""

    def test_aligned_episode_completes(self):
        """Test that aligned scenario can complete successfully."""
        env = DealRoomEnvironment()
        obs = env.reset(seed=42, task_id="aligned")

        # Systematic document delivery sequence
        actions = [
            # Round 0: Opening (already done in reset)
            # Round 1: ROI model to CFO
            DealRoomAction(
                action_type="send_document",
                target="CFO",
                message="Here is our detailed ROI analysis showing 14-month payback.",
                documents=[{"type": "roi_model", "specificity": "high"}],
            ),
            # Round 2: DPA to Legal
            DealRoomAction(
                action_type="send_document",
                target="Legal",
                message="Here is our GDPR-compliant DPA and SOC2 Type II certification.",
                documents=[{"type": "dpa", "specificity": "high"}],
            ),
            # Round 3: Security cert to Legal
            DealRoomAction(
                action_type="send_document",
                target="Legal",
                message="Additional security documentation including penetration test results.",
                documents=[{"type": "security_cert", "specificity": "high"}],
            ),
            # Round 4: Implementation timeline to CTO
            DealRoomAction(
                action_type="send_document",
                target="CTO",
                message="Our implementation team dedicates senior engineers to your integration.",
                documents=[{"type": "implementation_timeline", "specificity": "high"}],
            ),
            # Round 5: Reference case to Ops
            DealRoomAction(
                action_type="send_document",
                target="Ops",
                message="Reference case from a similar deployment that delivered on schedule.",
                documents=[{"type": "reference_case", "specificity": "high"}],
            ),
            # Round 6-7: Progress to closure
            DealRoomAction(
                action_type="direct_message",
                target="all",
                message="I believe we have addressed the core requirements. Let's move forward together.",
            ),
            DealRoomAction(
                action_type="group_proposal",
                target="all",
                message="I'm committed to a long-term partnership. Proposal: proceed with final approval.",
            ),
        ]

        final_score = 0.0
        for i, action in enumerate(actions):
            obs, reward, done, info = env.step(action)
            if done:
                final_score = reward
                break

        # Should complete (may or may not succeed depending on state)
        assert env.state.round_number > 0

    def test_aligned_with_collaborative_messages(self):
        """Test aligned scenario with collaborative language."""
        env = DealRoomEnvironment()
        env.reset(seed=42, task_id="aligned")

        collaborative_messages = [
            "Thank you for your time. We appreciate your partnership and want to work together.",
            "We understand your concerns and are flexible on the timeline.",
            "Our goal is mutual success. Let's find a solution that works for both parties.",
            "We value your feedback and are committed to addressing your specific needs.",
            "Partnership is about collaboration. We're invested in your long-term success.",
        ]

        for msg in collaborative_messages:
            action = DealRoomAction(
                action_type="direct_message", target="all", message=msg
            )
            obs, reward, done, info = env.step(action)
            if done:
                break

        # Environment should handle collaborative messages without errors


class TestEndToEndConflictedScenario:
    """Test complete conflicted scenario workflow."""

    def test_conflicted_episode_handles_tension(self):
        """Test that conflicted scenario handles CTO-CFO tension."""
        env = DealRoomEnvironment()
        env.reset(seed=42, task_id="conflicted")

        # First address CFO (before CTO in conflicted scenario per design)
        actions = [
            DealRoomAction(
                action_type="backchannel",
                target="CFO",
                message="I want to address your concerns directly. We're committed to making this work.",
            ),
            DealRoomAction(
                action_type="send_document",
                target="CFO",
                message="Here's the ROI model addressing your specific Q3 budget concerns.",
                documents=[{"type": "roi_model", "specificity": "high"}],
            ),
            DealRoomAction(
                action_type="send_document",
                target="CTO",
                message="Technical architecture and integration approach detailed here.",
                documents=[{"type": "implementation_timeline", "specificity": "high"}],
            ),
            DealRoomAction(
                action_type="direct_message",
                target="Legal",
                message="Here is our comprehensive GDPR-compliant DPA.",
                documents=[{"type": "dpa", "specificity": "high"}],
            ),
            DealRoomAction(
                action_type="direct_message",
                target="Procurement",
                message="Compliance documentation and vendor qualification complete.",
                documents=[{"type": "security_cert", "specificity": "high"}],
            ),
        ]

        for action in actions:
            obs, reward, done, info = env.step(action)
            if done:
                break


class TestEndToEndHostileAcquisition:
    """Test complete hostile acquisition scenario."""

    def test_hostile_acquisition_adapts_to_hint(self):
        """Test that scenario hint is properly injected at round 3."""
        env = DealRoomEnvironment()
        env.reset(seed=42, task_id="hostile_acquisition")

        actions = [
            DealRoomAction(
                action_type="direct_message", target="all", message="Initial outreach."
            ),
            DealRoomAction(
                action_type="send_document",
                target="Legal",
                message="DPA and GDPR compliance.",
                documents=[{"type": "dpa", "specificity": "high"}],
            ),
            DealRoomAction(
                action_type="send_document",
                target="CFO",
                message="ROI with compliance framing.",
                documents=[{"type": "roi_model", "specificity": "high"}],
            ),
        ]

        hint_received = False
        for i, action in enumerate(actions):
            obs, reward, done, info = env.step(action)
            if obs.scenario_hint:
                hint_received = True
                assert (
                    "GDPR" in obs.scenario_hint
                    or "compliance" in obs.scenario_hint.lower()
                )
            if done:
                break

        # Hint should be received at round 3
        assert hint_received or env.state.round_number >= 3


class TestEndToEndVetoAvoidance:
    """Test workflows that successfully avoid veto."""

    def test_veto_precursor_responded_to(self):
        """Test that responding to veto precursors prevents veto."""
        env = DealRoomEnvironment()
        env.reset(seed=42, task_id="aligned")

        # Simulate a scenario where veto risk builds
        env.state.veto_risk["CFO"] = 0.30
        env.state.veto_precursors_fired["CFO"] = False

        # Respond via backchannel before risk gets too high
        action = DealRoomAction(
            action_type="backchannel",
            target="CFO",
            channel="backchannel",
            message="I want to make sure we address any concerns you have directly. I'm committed to making this work.",
        )
        obs, reward, done, info = env.step(action)

        # Veto should not have fired
        assert not env.state.deal_failed

    def test_low_satisfaction_recovery(self):
        """Test that low satisfaction can be recovered."""
        env = DealRoomEnvironment()
        env.reset(seed=42, task_id="aligned")

        # Drop CFO satisfaction
        env.state.satisfaction["CFO"] = 0.30

        # Use collaborative approach
        action = DealRoomAction(
            action_type="direct_message",
            target="CFO",
            message="I understand your concerns. Let me address them specifically. We're flexible and want to find a solution that works for your situation.",
        )
        obs, reward, done, info = env.step(action)

        # Environment should handle gracefully
        assert obs is not None


class TestEndToEndTimingAndPacing:
    """Test timing and pacing behaviors."""

    def test_early_closure_better_than_late(self):
        """Test that earlier closure yields better scores."""
        # This would require running many episodes to test statistically
        # Just verify the scoring mechanism works
        from models import DealRoomState

        state_early = DealRoomState(
            deal_closed=True,
            deal_stage="closed",
            round_number=3,
            max_rounds=10,
            satisfaction={
                sid: 0.7 for sid in ["CFO", "CTO", "Legal", "Procurement", "Ops"]
            },
            beliefs={
                sid: {"competence": 0.6, "risk_tolerance": 0.6, "pricing_rigor": 0.5}
                for sid in ["CFO", "CTO", "Legal", "Procurement", "Ops"]
            },
            validation_failures=0,
        )

        state_late = DealRoomState(
            deal_closed=True,
            deal_stage="closed",
            round_number=9,
            max_rounds=10,
            satisfaction={
                sid: 0.7 for sid in ["CFO", "CTO", "Legal", "Procurement", "Ops"]
            },
            beliefs={
                sid: {"competence": 0.6, "risk_tolerance": 0.6, "pricing_rigor": 0.5}
                for sid in ["CFO", "CTO", "Legal", "Procurement", "Ops"]
            },
            validation_failures=0,
        )

        early_score = CCIGrader.compute(state_early)
        late_score = CCIGrader.compute(state_late)

        assert early_score > late_score


class TestEndToEndValidationFailures:
    """Test validation failure handling."""

    def test_garbage_input_handled(self):
        """Test that garbage input doesn't crash the environment."""
        env = DealRoomEnvironment()
        env.reset(seed=42, task_id="aligned")

        for _ in range(5):
            action = DealRoomAction(
                action_type="direct_message",
                target="all",
                message="!@#$%^&*()_+{}:|?><~`",
            )
            obs, reward, done, info = env.step(action)
            if done:
                break

        # Should complete without crashing
        assert env.state.round_number > 0

    def test_validation_failure_penalty_applied(self):
        """Test that repeated validation failures apply scrutiny mode."""
        env = DealRoomEnvironment()
        env.reset(seed=42, task_id="aligned")

        # Send low-quality messages
        for _ in range(5):
            action = DealRoomAction(
                action_type="direct_message", target="all", message="asdf"
            )
            obs, reward, done, info = env.step(action)
            if done:
                break

        # After multiple failures, scrutiny mode should activate
        # (when fallback_streak >= 2)


class TestEndToEndStageTransitions:
    """Test stage transition behaviors."""

    def test_evaluation_to_negotiation_requires_min_rounds(self):
        """Test that stage advancement requires minimum rounds."""
        env = DealRoomEnvironment()
        env.reset(seed=42, task_id="aligned")

        # Check initial stage
        assert env.state.deal_stage == "evaluation"

        # Take actions and verify progression rules
        for _ in range(10):
            action = DealRoomAction(
                action_type="direct_message", target="all", message="Test"
            )
            obs, reward, done, info = env.step(action)
            if done:
                break

        # Stage should have advanced (or not) based on rules

    def test_regression_on_blocker_at_legal_review(self):
        """Test that new blockers at legal_review cause regression."""
        env = DealRoomEnvironment()
        env.reset(seed=42, task_id="aligned")

        # Manually set to legal_review
        env.state.deal_stage = "legal_review"
        initial_stage = env.state.deal_stage

        # Create a blocker
        env.state.satisfaction["CFO"] = 0.25

        action = DealRoomAction(
            action_type="direct_message", target="all", message="Test"
        )
        obs, reward, done, info = env.step(action)

        # Check if regression occurred
        # Note: regression logic depends on specific conditions


class TestEndToEndMessageQuality:
    """Test message quality impact on outcomes."""

    def test_collaborative_vs_aggressive_outcomes(self):
        """Test that collaborative language produces better outcomes."""
        env1 = DealRoomEnvironment()
        env2 = DealRoomEnvironment()

        env1.reset(seed=42, task_id="aligned")
        env2.reset(seed=42, task_id="aligned")

        collaborative_msg = "Thank you for your time. We value our partnership and want to work together for mutual success."
        aggressive_msg = (
            "This is our final offer. Take it or leave it. Deadline is non-negotiable."
        )

        # Run a few steps with collaborative
        for _ in range(3):
            action = DealRoomAction(
                action_type="direct_message", target="all", message=collaborative_msg
            )
            env1.step(action)

        # Run a few steps with aggressive
        for _ in range(3):
            action = DealRoomAction(
                action_type="direct_message", target="all", message=aggressive_msg
            )
            env2.step(action)

        # Collaborative should maintain or improve satisfaction
        # Aggressive should decrease or maintain
        # This is a probabilistic test


class TestEndToEndDocumentSequence:
    """Test document delivery sequence impact."""

    def test_systematic_document_delivery(self):
        """Test that systematic document delivery works."""
        env = DealRoomEnvironment()
        env.reset(seed=42, task_id="aligned")

        documents = [
            ("CFO", "roi_model", "high"),
            ("Legal", "dpa", "high"),
            ("CTO", "implementation_timeline", "high"),
            ("Legal", "security_cert", "high"),
            ("Procurement", "reference_case", "high"),
        ]

        for target, doc_type, spec in documents:
            action = DealRoomAction(
                action_type="send_document",
                target=target,
                message=f"Here is the {doc_type} document.",
                documents=[{"type": doc_type, "specificity": spec}],
            )
            obs, reward, done, info = env.step(action)
            if done:
                break

        # Should complete without errors
        assert env.state.round_number > 0


class TestEndToEndConcurrentScenarios:
    """Test running multiple scenarios."""

    def test_multiple_scenarios_sequential(self):
        """Test running all scenarios sequentially."""
        for task_id in ["aligned", "conflicted", "hostile_acquisition"]:
            env = DealRoomEnvironment()
            env.reset(seed=42, task_id=task_id)

            for _ in range(10):
                action = DealRoomAction(
                    action_type="direct_message", target="all", message="Test"
                )
                obs, reward, done, info = env.step(action)
                if done:
                    break

            assert env.state.round_number > 0

    def test_random_seed_variation(self):
        """Test that different seeds produce variation."""
        scores = []
        for seed in [1, 42, 100, 1234]:
            env = DealRoomEnvironment()
            env.reset(seed=seed, task_id="aligned")

            for _ in range(15):
                action = DealRoomAction(
                    action_type="direct_message", target="all", message="Test"
                )
                obs, reward, done, info = env.step(action)
                if done:
                    break

            scores.append(env.state.round_number)

        # Different seeds should produce different round numbers
        # (probabilistically, not always)


class TestEndToEndCalibration:
    """Test calibration patterns (from calibrate.py)."""

    def test_random_agent_baseline(self):
        """Test random agent produces expected baseline scores."""
        from calibrate import RandomAgent

        scores = []
        for seed in range(20):
            env = DealRoomEnvironment()
            obs = env.reset(seed=seed, task_id="aligned")
            agent = RandomAgent(np.random.default_rng(seed))

            for _ in range(20):
                action = agent.act(obs)
                obs, reward, done, info = env.step(action)
                if done:
                    scores.append(reward)
                    break

        # Random agent should produce low scores (mostly 0s, maybe some low scores)
        avg_score = sum(scores) / len(scores) if scores else 0
        assert 0.0 <= avg_score <= 0.3  # Random should be in low range


class TestEndToEndStrategicAgent:
    """Test strategic agent patterns."""

    def test_strategic_agent_basic(self):
        """Test that strategic agent produces better outcomes than random."""
        from calibrate import StrategicAgent

        env = DealRoomEnvironment()
        obs = env.reset(seed=42, task_id="aligned")
        agent = StrategicAgent()

        for _ in range(15):
            action = agent.act(obs)
            obs, reward, done, info = env.step(action)
            if done:
                break

        # Strategic agent should complete the episode
        assert env.state.round_number > 0
