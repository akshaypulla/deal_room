"""
Unit tests for ClaimsTracker - regex-only numerical contradiction detection.
Tests claim extraction, deviation detection, and target expansion.
"""

import pytest
from server.claims import ClaimsTracker, expand_targets, VALID_SUBGROUPS, CLAIM_PATTERNS


class TestExpandTargets:
    """Test target string expansion to individual stakeholder IDs."""

    def test_all_expands_to_all_ids(self, stakeholder_ids):
        result = expand_targets("all")
        assert len(result) == 5
        assert set(result) == set(stakeholder_ids)

    def test_single_stakeholder(self):
        assert expand_targets("CFO") == ["CFO"]
        assert expand_targets("cto") == ["CTO"]
        assert expand_targets("CTO") == ["CTO"]
        assert expand_targets("Legal") == ["Legal"]
        assert expand_targets("Procurement") == ["Procurement"]
        assert expand_targets("Ops") == ["Ops"]

    def test_subgroup_cto_cfo(self):
        result = expand_targets("cto_cfo")
        assert result == ["CTO", "CFO"]

    def test_subgroup_legal_procurement(self):
        result = expand_targets("legal_procurement")
        assert result == ["Legal", "Procurement"]

    def test_invalid_target_returns_empty(self):
        assert expand_targets("invalid") == []
        assert expand_targets("") == []
        assert expand_targets("xyz") == []

    def test_case_insensitive_single_target(self):
        assert expand_targets("cfo") == ["CFO"]
        assert expand_targets("CfO") == ["CFO"]  # case insensitive match
        assert expand_targets("cto") == ["CTO"]
        assert expand_targets("LEGAL") == ["Legal"]

    def test_subgroups_case_insensitive(self):
        result = expand_targets("CTO_CFO")
        assert set(result) == {"CTO", "CFO"}

    def test_valid_subgroups_registry(self):
        assert "cto_cfo" in VALID_SUBGROUPS
        assert "legal_procurement" in VALID_SUBGROUPS
        assert VALID_SUBGROUPS["cto_cfo"] == ["CTO", "CFO"]
        assert VALID_SUBGROUPS["legal_procurement"] == ["Legal", "Procurement"]


class TestClaimsTrackerBasic:
    """Test ClaimsTracker basic functionality."""

    def test_initialization(self):
        tracker = ClaimsTracker()
        assert tracker.claims == {}

    def test_reset(self):
        tracker = ClaimsTracker()
        tracker.claims = {"CFO:implementation_weeks": [4.0]}
        tracker.reset()
        assert tracker.claims == {}

    def test_extract_no_message(self):
        tracker = ClaimsTracker()
        result = tracker.extract_and_track("CFO", "")
        assert result is False

    def test_extract_no_target(self):
        tracker = ClaimsTracker()
        result = tracker.extract_and_track("", "Some message")
        assert result is False


class TestClaimsTrackerImplementationWeeks:
    """Test implementation_weeks claim pattern extraction."""

    def test_go_live_weeks(self):
        tracker = ClaimsTracker()
        result = tracker.extract_and_track("CFO", "We can go live in 12 weeks.")
        assert result is False  # First extraction, no prior
        assert "CFO:implementation_weeks" in tracker.claims
        assert tracker.claims["CFO:implementation_weeks"] == [12.0]

    def test_delivery_weeks(self):
        tracker = ClaimsTracker()
        result = tracker.extract_and_track("CTO", "We will deliver in 8 weeks.")
        assert tracker.claims["CTO:implementation_weeks"] == [8.0]

    def test_deploy_weeks(self):
        tracker = ClaimsTracker()
        result = tracker.extract_and_track("Ops", "We can deploy within 6 weeks.")
        assert tracker.claims["Ops:implementation_weeks"] == [6.0]

    def test_development_weeks_wrong_pattern(self):
        """Test that "Development" does NOT match the deliver pattern."""
        tracker = ClaimsTracker()
        result = tracker.extract_and_track(
            "CFO", "Development will take approximately 12 weeks."
        )
        # "Development" doesn't match "deliver" pattern - only exact words match
        assert "CFO:implementation_weeks" not in tracker.claims


class TestClaimsTrackerPriceCommit:
    """Test price_commit claim pattern extraction."""

    def test_simple_price(self):
        tracker = ClaimsTracker()
        result = tracker.extract_and_track("CFO", "Our price is $450,000.")
        assert tracker.claims["CFO:price_commit"] == [450000.0]

    def test_price_with_comma(self):
        tracker = ClaimsTracker()
        result = tracker.extract_and_track("CFO", "The contract is worth $1,500,000.")
        assert tracker.claims["CFO:price_commit"] == [1500000.0]

    def test_price_four_digits(self):
        tracker = ClaimsTracker()
        result = tracker.extract_and_track("Procurement", "Cost is $15,000 per user.")
        assert tracker.claims["Procurement:price_commit"] == [15000.0]


class TestClaimsTrackerTeamSize:
    """Test team_size claim pattern extraction."""

    def test_team_of_format(self):
        tracker = ClaimsTracker()
        result = tracker.extract_and_track("CTO", "We will have a team of 5 engineers.")
        assert tracker.claims["CTO:team_size"] == [5.0]

    def test_dedicated_engineers_format(self):
        tracker = ClaimsTracker()
        result = tracker.extract_and_track(
            "CTO", "We will provide 3 dedicated engineers."
        )
        assert tracker.claims["CTO:team_size"] == [3.0]

    def test_multi_engineers_format(self):
        tracker = ClaimsTracker()
        result = tracker.extract_and_track(
            "CTO", "Our team consists of 8 members for this project."
        )
        assert tracker.claims["CTO:team_size"] == [8.0]


class TestClaimsTrackerDeviation:
    """Test contradiction detection when values deviate beyond tolerance."""

    def test_no_contradiction_same_value(self):
        tracker = ClaimsTracker()
        tracker.extract_and_track("CFO", "Go live in 12 weeks.")
        result = tracker.extract_and_track("CFO", "We can deliver in 12 weeks.")
        assert result is False  # No deviation > 15%

    def test_contradiction_within_tolerance(self):
        tracker = ClaimsTracker()
        tracker.extract_and_track("CFO", "Go live in 12 weeks.")
        result = tracker.extract_and_track("CFO", "We estimate 13 weeks.")
        assert result is False  # 8.3% deviation < 15%

    def test_contradiction_outside_tolerance(self):
        tracker = ClaimsTracker()
        tracker.extract_and_track("CFO", "Go live in 10 weeks.")
        # Use text that matches the pattern - "We can deploy in 16 weeks" should match
        result = tracker.extract_and_track(
            "CFO", "We can deploy in 16 weeks now."
        )  # 60% deviation > 15%
        assert result is True

    def test_contradiction_price_increase(self):
        tracker = ClaimsTracker()
        tracker.extract_and_track("CFO", "Price is $100000.")
        result = tracker.extract_and_track("CFO", "Price is now $130000.")
        assert result is True  # 30% deviation > 15%

    def test_contradiction_price_decrease(self):
        tracker = ClaimsTracker()
        tracker.extract_and_track("CFO", "Price is $100000.")
        result = tracker.extract_and_track("CFO", "Price is now $80000.")
        assert result is True  # 20% deviation > 15%

    def test_multiple_claims_accumulated(self):
        tracker = ClaimsTracker()
        tracker.extract_and_track("CTO", "Team of 5 engineers.")
        tracker.extract_and_track("CTO", "We can do 6 engineers.")
        tracker.extract_and_track("CTO", "Actually 7 engineers.")
        assert len(tracker.claims["CTO:team_size"]) == 3

    def test_different_claim_types_independent(self):
        tracker = ClaimsTracker()
        tracker.extract_and_track("CFO", "Go live in 12 weeks, price is $100000.")
        assert "CFO:implementation_weeks" in tracker.claims
        assert "CFO:price_commit" in tracker.claims


class TestClaimsTrackerEdgeCases:
    """Test edge cases and boundary conditions for ClaimsTracker."""

    def test_zero_first_value(self):
        tracker = ClaimsTracker()
        tracker.extract_and_track("CFO", "Go live in 0 weeks.")  # Edge case
        # Next extraction with non-zero should not trigger deviation from zero
        result = tracker.extract_and_track("CFO", "Actually 4 weeks.")
        assert result is False  # Can't compute deviation from zero

    def test_very_large_numbers(self):
        tracker = ClaimsTracker()
        tracker.extract_and_track("CFO", "Contract worth $10,000,000.")
        assert tracker.claims["CFO:price_commit"] == [10000000.0]

    def test_decimal_values(self):
        tracker = ClaimsTracker()
        tracker.extract_and_track(
            "CTO", "Implementation in 6 weeks."
        )  # Only integers captured
        assert tracker.claims["CTO:implementation_weeks"] == [6.0]

    def test_unicode_numbers(self):
        tracker = ClaimsTracker()
        result = tracker.extract_and_track("CFO", "Price is $٥٠,٠٠٠")  # Arabic numerals
        # May or may not extract, but shouldn't crash

    def test_message_with_no_claims(self):
        tracker = ClaimsTracker()
        result = tracker.extract_and_track("CFO", "Thank you for your time today.")
        assert result is False
        assert len(tracker.claims) == 0

    def test_multiple_numbers_extracted(self):
        tracker = ClaimsTracker()
        # Implementation weeks pattern should capture the first number in the message
        result = tracker.extract_and_track("CFO", "Go live in 12 weeks, team of 5.")
        # Only implementation_weeks pattern fires, team_size needs "team of" or "of X engineers"
        assert "CFO:implementation_weeks" in tracker.claims

    def test_claim_patterns_all_valid(self):
        """Verify all claim patterns are valid regex."""
        for name, pattern in CLAIM_PATTERNS.items():
            import re

            re.compile(pattern)  # Should not raise


class TestClaimsTrackerMultipleStakeholders:
    """Test tracking claims for multiple stakeholders independently."""

    def test_independent_tracking(self):
        tracker = ClaimsTracker()
        tracker.extract_and_track("CFO", "Go live in 10 weeks.")
        tracker.extract_and_track("CTO", "Team of 5 engineers.")
        assert "CFO:implementation_weeks" in tracker.claims
        assert "CTO:team_size" in tracker.claims
        assert "CFO:team_size" not in tracker.claims

    def test_same_stakeholder_different_claim_types(self):
        tracker = ClaimsTracker()
        tracker.extract_and_track("CFO", "Go live in 10 weeks.")
        result = tracker.extract_and_track("CFO", "Price is $500000.")
        assert result is False  # Different claim types don't contradict
        assert len(tracker.claims) == 2
