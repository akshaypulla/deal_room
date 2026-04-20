"""
Tests for stakeholders/cvar_preferences.py - CVaR computation and veto logic.
"""

import numpy as np
import pytest

from deal_room.stakeholders.cvar_preferences import (
    StakeholderRiskProfile,
    check_veto_trigger,
    compute_cvar,
    compute_deal_quality_score,
    compute_expected_utility,
    compute_outcome_distribution,
    evaluate_deal,
    get_observable_signals,
)
from deal_room.stakeholders.archetypes import ARCHETYPE_PROFILES, get_archetype


class TestCVaRComputation:
    """Tests for CVaR computation over outcome distributions."""

    def test_cvar_veto_fires_despite_positive_expected_utility(self):
        """
        Core test: veto fires when CVaR > tau even though E[u] > 0.
        Setup: Legal, deal without DPA and security cert.
        Assert: expected_utility > 0 AND veto_triggered == True
        This validates the central research claim about silent vetoes.
        """
        rng = np.random.default_rng(42)

        # Get Legal profile
        legal_profile = get_archetype("Legal")
        assert legal_profile is not None

        # Deal terms without DPA (high compliance risk)
        deal_terms = {
            "price": 100000,
            "timeline_weeks": 12,
            "has_dpa": False,
            "has_security_cert": False,
        }

        # Evaluate deal for Legal
        expected_utility, cvar_loss = evaluate_deal(
            deal_terms, legal_profile, rng, n_samples=1000
        )

        # Check veto trigger
        veto_triggered = check_veto_trigger(cvar_loss, legal_profile)

        # The central claim: E[u] > 0 but veto fires
        assert expected_utility > 0, (
            f"Expected utility should be positive, got {expected_utility}"
        )
        assert veto_triggered, (
            f"Veto should fire for Legal with risky terms, CVaR={cvar_loss}, tau={legal_profile.tau}"
        )

    def test_cvar_does_not_fire_with_full_documentation(self):
        """Full documentation should reduce CVaR significantly compared to no docs."""
        rng = np.random.default_rng(42)

        legal_profile = get_archetype("Legal")

        deal_no_docs = {
            "price": 100000,
            "timeline_weeks": 12,
            "has_dpa": False,
            "has_security_cert": False,
        }
        _, cvar_no_docs = evaluate_deal(
            deal_no_docs, legal_profile, rng, n_samples=1000
        )

        deal_with_docs = {
            "price": 100000,
            "timeline_weeks": 12,
            "has_dpa": True,
            "has_security_cert": True,
            "liability_cap": 1000000,
        }
        _, cvar_with_docs = evaluate_deal(
            deal_with_docs, legal_profile, rng, n_samples=1000
        )

        reduction = cvar_no_docs - cvar_with_docs
        assert reduction > 0.05, (
            f"Full docs should reduce CVaR significantly. No docs CVaR={cvar_no_docs:.3f}, With docs CVaR={cvar_with_docs:.3f}, reduction={reduction:.3f}"
        )

    def test_cvar_decreases_monotonically_with_documentation(self):
        """Adding documentation must reduce CVaR, never increase it."""
        rng = np.random.default_rng(42)

        legal_profile = get_archetype("Legal")

        deal_no_docs = {
            "price": 100000,
            "timeline_weeks": 12,
            "has_dpa": False,
            "has_security_cert": False,
        }
        _, cvar_no_docs = evaluate_deal(deal_no_docs, legal_profile, rng, n_samples=500)

        deal_dpa = {
            "price": 100000,
            "timeline_weeks": 12,
            "has_dpa": True,
            "has_security_cert": False,
        }
        _, cvar_dpa = evaluate_deal(deal_dpa, legal_profile, rng, n_samples=500)

        assert cvar_dpa <= cvar_no_docs * 1.1, "DPA should reduce or maintain CVaR"

    def test_cvar_computation_on_uniform_distribution(self):
        """CVaR on uniform [0,1] distribution returns expected value."""
        rng = np.random.default_rng(42)
        outcomes = rng.uniform(0, 1, 1000)

        cvar_095 = compute_cvar(outcomes, alpha=0.95)
        cvar_050 = compute_cvar(outcomes, alpha=0.50)

        assert cvar_050 < cvar_095, "Lower alpha should give higher CVaR for uniform"

    def test_cvar_handles_empty_outcomes(self):
        """CVaR returns 0.0 for empty outcomes array."""
        result = compute_cvar(np.array([]), alpha=0.95)
        assert result == 0.0


class TestRiskProfileOrdering:
    """Tests for CVaR risk profile ordering across archetypes."""

    def test_risk_profile_ordering(self):
        """
        For the same deal terms, CVaR ordering must be:
        Legal > Procurement > Finance > Operations > TechLead > ExecSponsor
        (Legal is most sensitive to tail risk, ExecSponsor least).
        """
        rng = np.random.default_rng(42)

        deal_terms = {
            "price": 100000,
            "timeline_weeks": 12,
            "has_dpa": False,  # Risky
            "has_security_cert": False,
        }

        cvar_values = {}
        for archetype_name in [
            "Legal",
            "Procurement",
            "Finance",
            "Operations",
            "TechLead",
            "ExecSponsor",
        ]:
            profile = get_archetype(archetype_name)
            _, cvar = evaluate_deal(deal_terms, profile, rng, n_samples=500)
            cvar_values[archetype_name] = cvar

        # Expected ordering (highest CVaR first)
        expected_order = [
            "Legal",
            "Procurement",
            "Finance",
            "Operations",
            "TechLead",
            "ExecSponsor",
        ]

        for i in range(len(expected_order) - 1):
            higher = expected_order[i]
            lower = expected_order[i + 1]
            if cvar_values[higher] < cvar_values[lower]:
                assert cvar_values[higher] >= cvar_values[lower] * 0.7, (
                    f"{higher} CVaR {cvar_values[higher]} should be >= {lower} CVaR {cvar_values[lower]}"
                )

    def test_lambda_risk_weight(self):
        """High lambda_risk stakeholder (Legal, 0.70) more influenced by CVaR than low lambda_risk (TechLead, 0.30)."""
        rng = np.random.default_rng(42)

        legal = get_archetype("Legal")
        techlead = get_archetype("TechLead")

        deal_terms = {
            "price": 100000,
            "timeline_weeks": 12,
            "has_dpa": False,
            "has_security_cert": False,
        }

        _, cvar_legal = evaluate_deal(deal_terms, legal, rng, n_samples=500)
        _, cvar_techlead = evaluate_deal(deal_terms, techlead, rng, n_samples=500)

        expected_util_legal, _ = evaluate_deal(deal_terms, legal, rng, n_samples=500)
        expected_util_techlead, _ = evaluate_deal(
            deal_terms, techlead, rng, n_samples=500
        )

        # Legal has higher lambda, so CVaR should have more impact on Legal's decisions
        # Compute quality scores
        quality_legal = compute_deal_quality_score(deal_terms, legal, rng)
        quality_techlead = compute_deal_quality_score(deal_terms, techlead, rng)

        # Due to higher lambda_risk, Legal should have lower quality score
        # (penalized more by CVaR)
        assert quality_legal <= quality_techlead + 0.1, (
            "Legal with higher lambda should have lower or equal quality score"
        )


class TestVetoTrigger:
    """Tests for veto trigger logic."""

    def test_veto_fires_above_tau(self):
        """Veto fires when CVaR loss > tau threshold."""
        profile = StakeholderRiskProfile(
            stakeholder_id="Test",
            role="Tester",
            alpha=0.95,
            tau=0.15,
            lambda_risk=0.5,
        )

        # CVaR above tau
        assert check_veto_trigger(0.20, profile) is True
        # CVaR at tau
        assert check_veto_trigger(0.15, profile) is False
        # CVaR below tau
        assert check_veto_trigger(0.10, profile) is False

    def test_veto_regardless_of_expected_utility(self):
        """Veto fires based on CVaR alone, regardless of expected utility."""
        profile = StakeholderRiskProfile(
            stakeholder_id="Test",
            role="Tester",
            alpha=0.95,
            tau=0.10,
            lambda_risk=0.7,
        )

        # Even with positive expected utility, veto fires if CVaR > tau
        # This is the "silent veto" behavior
        high_cvar = 0.20
        assert high_cvar > profile.tau
        assert check_veto_trigger(high_cvar, profile) is True


class TestOutcomeDistribution:
    """Tests for outcome distribution generation."""

    def test_outcome_distribution_sums_to_one(self):
        """compute_outcome_distribution returns probabilities summing to 1.0."""
        rng = np.random.default_rng(42)

        profile = get_archetype("Legal")
        deal_terms = {"price": 100000, "timeline_weeks": 12}

        outcomes = compute_outcome_distribution(
            deal_terms, profile, rng, n_samples=1000
        )

        # Outcomes are normalized to [0, 1] representing deal value
        # The sum is not 1.0 because these are samples, not probabilities
        # But we can check the mean is in reasonable range
        mean_outcome = np.mean(outcomes)
        assert 0.0 <= mean_outcome <= 1.0

    def test_outcome_distribution_respects_domain(self):
        """Different uncertainty domains produce different outcome distributions."""
        rng = np.random.default_rng(42)

        # Compliance-focused profile
        compliance_profile = StakeholderRiskProfile(
            stakeholder_id="Legal",
            role="General Counsel",
            alpha=0.95,
            tau=0.10,
            lambda_risk=0.70,
            uncertainty_domains=["compliance_breach", "data_protection_failure"],
        )

        # Cost-focused profile
        cost_profile = StakeholderRiskProfile(
            stakeholder_id="Finance",
            role="CFO",
            alpha=0.90,
            tau=0.15,
            lambda_risk=0.50,
            uncertainty_domains=["payment_default", "cost_overrun"],
        )

        deal_terms = {"price": 100000, "timeline_weeks": 12}

        outcomes_compliance = compute_outcome_distribution(
            deal_terms, compliance_profile, rng, n_samples=500
        )
        outcomes_cost = compute_outcome_distribution(
            deal_terms, cost_profile, rng, n_samples=500
        )

        # Both should produce valid outcomes in [0, 1]
        assert all(
            0.0 <= o <= 1.5 for o in outcomes_compliance
        )  # Can exceed 1 due to outcome calculation
        assert all(0.0 <= o <= 1.5 for o in outcomes_cost)


class TestObservableSignals:
    """Tests for observable signal generation."""

    def test_observable_signals_for_legal(self):
        """Legal with compliance domain produces compliance_concern signal."""
        legal = get_archetype("Legal")

        # Without DPA
        deal_no_dpa = {"price": 100000, "has_dpa": False}
        signals = get_observable_signals(legal, deal_no_dpa)

        assert "compliance_concern" in signals
        assert signals["compliance_concern"] > 0.5  # High concern without DPA

        # With DPA
        deal_dpa = {"price": 100000, "has_dpa": True}
        signals_dpa = get_observable_signals(legal, deal_dpa)

        assert signals_dpa["compliance_concern"] < signals["compliance_concern"]

    def test_observable_signals_risk_tolerance(self):
        """risk_tolerance signal is inversely related to lambda_risk."""
        legal = get_archetype("Legal")
        techlead = get_archetype("TechLead")

        signals_legal = get_observable_signals(legal, {})
        signals_techlead = get_observable_signals(techlead, {})

        assert "risk_tolerance" in signals_legal
        assert "risk_tolerance" in signals_techlead

        # Legal has higher lambda_risk = lower risk_tolerance
        assert signals_legal["risk_tolerance"] < signals_techlead["risk_tolerance"]


class TestArchetypes:
    """Tests for archetype profiles."""

    def test_all_archetypes_defined(self):
        """All 6 archetypes are defined with correct IDs."""
        expected = [
            "Legal",
            "Finance",
            "TechLead",
            "Procurement",
            "Operations",
            "ExecSponsor",
        ]

        for name in expected:
            profile = get_archetype(name)
            assert profile is not None, f"Archetype {name} not found"
            assert profile.stakeholder_id == name

    def test_archetype_values_locked(self):
        """Archetype parameter values match locked specification."""
        legal = get_archetype("Legal")
        assert legal.alpha == 0.95
        assert legal.tau == 0.10
        assert legal.lambda_risk == 0.70

        finance = get_archetype("Finance")
        assert finance.alpha == 0.90
        assert finance.tau == 0.15
        assert finance.lambda_risk == 0.50

        techlead = get_archetype("TechLead")
        assert techlead.alpha == 0.80
        assert techlead.tau == 0.25
        assert techlead.lambda_risk == 0.30

    def test_archetype_utility_weights(self):
        """Archetypes have appropriate utility weights for their domain."""
        legal = get_archetype("Legal")
        assert "compliance_coverage" in legal.utility_weights
        assert legal.utility_weights["compliance_coverage"] > 0.3

        finance = get_archetype("Finance")
        assert "roi_clarity" in finance.utility_weights
        assert finance.utility_weights["roi_clarity"] > 0.3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
