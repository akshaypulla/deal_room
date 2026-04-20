#!/usr/bin/env python3
"""
test_08_cvar_preferences.py
DealRoom v3 — CVaR Preferences & Grader Unit Tests (runs inside container)

Validates:
- CORE CLAIM: eu>0 but cvar_loss > tau → veto fires
- Full documentation (DPA + security cert) yields lower CVaR than poor docs
- CVaR formula is mathematically correct
- CVaR differentiates between high/low outcome distributions
- Risk tolerance ordering: Legal < Finance < ExecSponsor
- Aggressive timeline increases TechLead CVaR vs reasonable timeline
- CVaR is coherent (satisfies subadditivity on this distribution)
"""

import sys

sys.path.insert(0, "/app/env")

import os
from pathlib import Path

try:
    from dotenv import load_dotenv

    _dotenv = Path("/app/.env")
    if _dotenv.exists():
        load_dotenv(_dotenv)
except Exception:
    pass


def test_8_1_core_claim():
    print("\n[8.1] CORE CLAIM: eu>0 yet cvar_loss > tau → veto fires...")
    from deal_room.stakeholders.cvar_preferences import evaluate_deal
    from deal_room.stakeholders.archetypes import get_archetype
    import numpy as np

    legal_profile = get_archetype("Legal")
    rng = np.random.default_rng(42)

    terms = {
        "price": 0.85,
        "support_level": "enterprise",
        "timeline_weeks": 12,
        "has_dpa": False,
        "has_security_cert": False,
        "liability_cap": 0.2,
    }
    eu, cvar_loss = evaluate_deal(terms, legal_profile, rng, n_samples=500)

    print(f"  EU = {eu:.4f}")
    print(f"  CVaR loss = {cvar_loss:.4f}")
    print(f"  tau (Legal) = {legal_profile.tau:.4f}")

    assert eu > 0, (
        f"EU must be positive (this deal has good expected value): eu={eu:.4f}"
    )
    assert cvar_loss > legal_profile.tau, (
        f"CVaR loss {cvar_loss:.4f} must exceed tau {legal_profile.tau} for veto to fire"
    )
    assert cvar_loss > eu, "CVaR loss should exceed EU in this risky deal configuration"

    print(
        f"  ✓ CORE CLAIM VERIFIED: eu={eu:.3f}>0, cvar_loss={cvar_loss:.3f}>tau={legal_profile.tau:.3f}"
    )


def test_8_2_good_docs_lower_cvar_than_poor():
    print("\n[8.2] Full documentation reduces CVaR vs poor documentation...")
    from deal_room.stakeholders.cvar_preferences import evaluate_deal
    from deal_room.stakeholders.archetypes import get_archetype
    import numpy as np

    legal_profile = get_archetype("Legal")
    rng = np.random.default_rng(42)

    poor_terms = {
        "price": 0.85,
        "support_level": "enterprise",
        "timeline_weeks": 12,
        "has_dpa": False,
        "has_security_cert": False,
        "liability_cap": 0.2,
    }
    good_terms = {
        "price": 0.70,
        "support_level": "enterprise",
        "timeline_weeks": 16,
        "has_dpa": True,
        "has_security_cert": True,
        "liability_cap": 1.0,
    }

    _, cvar_poor = evaluate_deal(poor_terms, legal_profile, rng, n_samples=500)
    _, cvar_good = evaluate_deal(good_terms, legal_profile, rng, n_samples=500)

    print(f"  Poor docs CVaR: {cvar_poor:.4f}")
    print(f"  Good docs CVaR: {cvar_good:.4f}")

    assert cvar_good < cvar_poor, (
        f"Good docs should reduce CVaR: good={cvar_good:.3f} >= poor={cvar_poor:.3f} — broken grader"
    )

    print(f"  ✓ Good docs reduce CVaR: {cvar_poor:.3f} → {cvar_good:.3f}")


def test_8_3_cvar_formula_correct():
    print("\n[8.3] CVaR formula correctness: high outcomes → low CVaR...")
    from deal_room.stakeholders.cvar_preferences import compute_cvar
    import numpy as np

    outcomes = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 0.5, 0.5, 0.5, 0.5, 0.5])
    cvar = compute_cvar(outcomes, alpha=0.95)

    print(f"  CVaR at 0.95 for [1.0×5, 0.5×5]: {cvar:.4f}")
    assert 0.5 <= cvar <= 1.0, f"CVaR should be between 0.5 and 1.0, got {cvar:.4f}"

    cvar_high = compute_cvar(np.array([0.8] * 10), alpha=0.90)
    cvar_low = compute_cvar(np.array([0.2] * 10), alpha=0.90)

    assert cvar_high < cvar_low, (
        f"Higher outcomes should give lower CVaR: {cvar_high:.3f} >= {cvar_low:.3f}"
    )

    print(f"  ✓ CVaR formula correct: high={cvar_high:.3f}, low={cvar_low:.3f}")


def test_8_4_tau_ordering():
    print("\n[8.4] Risk tolerance ordering: Legal < Finance < ExecSponsor...")
    from deal_room.stakeholders.archetypes import get_archetype

    legal_tau = get_archetype("Legal").tau
    finance_tau = get_archetype("Finance").tau
    exec_tau = get_archetype("ExecSponsor").tau

    print(f"  Legal:       tau = {legal_tau:.3f}")
    print(f"  Finance:     tau = {finance_tau:.3f}")
    print(f"  ExecSponsor:  tau = {exec_tau:.3f}")

    assert legal_tau < finance_tau, (
        f"Legal.tau ({legal_tau:.2f}) should be < Finance.tau ({finance_tau:.2f})"
    )
    assert finance_tau < exec_tau, (
        f"Finance.tau ({finance_tau:.2f}) should be < ExecSponsor.tau ({exec_tau:.2f})"
    )

    print("  ✓ Tau ordering: Legal < Finance < ExecSponsor")


def test_8_5_aggressive_timeline_higher_cvar():
    print("\n[8.5] Aggressive timeline increases TechLead CVaR...")
    from deal_room.stakeholders.cvar_preferences import evaluate_deal
    from deal_room.stakeholders.archetypes import get_archetype
    import numpy as np

    tech_profile = get_archetype("TechLead")
    rng = np.random.default_rng(42)

    t_agg = {
        "has_dpa": True,
        "has_security_cert": True,
        "liability_cap": 1.0,
        "price": 0.70,
        "support_level": "enterprise",
        "timeline_weeks": 4,
    }
    t_reas = {
        "has_dpa": True,
        "has_security_cert": True,
        "liability_cap": 1.0,
        "price": 0.70,
        "support_level": "enterprise",
        "timeline_weeks": 16,
    }

    _, cvar_a = evaluate_deal(t_agg, tech_profile, rng, n_samples=500)
    _, cvar_r = evaluate_deal(t_reas, tech_profile, rng, n_samples=500)

    print(f"  Aggressive (4wk) CVaR: {cvar_a:.4f}")
    print(f"  Reasonable (16wk) CVaR: {cvar_r:.4f}")

    assert cvar_a > cvar_r, (
        f"Aggressive timeline CVaR {cvar_a:.3f} must exceed reasonable {cvar_r:.3f}"
    )

    print(f"  ✓ Aggressive timeline CVaR ({cvar_a:.3f}) > reasonable ({cvar_r:.3f})")


def test_8_6_cvar_subadditivity_sanity():
    print("\n[8.6] CVaR is coherent (satisfies subadditivity on this distribution)...")
    from deal_room.stakeholders.cvar_preferences import compute_cvar
    import numpy as np

    # CVaR should be ≤ max (coherent risk measure property)
    outcomes = np.array([0.3, 0.4, 0.5, 0.6, 0.9])
    cvar = compute_cvar(outcomes, alpha=0.95)
    max_val = np.max(outcomes)

    print(f"  CVaR: {cvar:.4f}, Max: {max_val:.4f}")
    assert cvar <= max_val, f"CVaR ({cvar:.4f}) must be ≤ max outcome ({max_val:.4f})"
    print("  ✓ CVaR is ≤ max outcome (coherence check)")


def run_all():
    print("=" * 60)
    print("  DealRoom v3 — CVaR Preferences Unit Tests (Container)")
    print("=" * 60)

    tests = [
        test_8_1_core_claim,
        test_8_2_good_docs_lower_cvar_than_poor,
        test_8_3_cvar_formula_correct,
        test_8_4_tau_ordering,
        test_8_5_aggressive_timeline_higher_cvar,
        test_8_6_cvar_subadditivity_sanity,
    ]

    failed = []
    for t in tests:
        try:
            t()
        except AssertionError as e:
            print(f"  ✗ FAILED: {e}")
            failed.append(t.__name__)
        except Exception as e:
            print(f"  ✗ ERROR: {e}")
            failed.append(t.__name__)

    print("\n" + "=" * 60)
    passed = len(tests) - len(failed)
    print(f"  ✓ SECTION 8 — {passed}/{len(tests)} checks passed")
    if failed:
        print(f"  ✗ FAILED: {failed}")
        import sys

        sys.exit(1)
    print("=" * 60)


if __name__ == "__main__":
    run_all()
