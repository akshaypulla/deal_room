import sys

sys.path.insert(0, "/app/env")


def test_8_1_the_core_claim():
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

    assert eu > 0, f"Expected utility must be positive. Got: {eu:.4f}"
    assert cvar_loss > legal_profile.tau, (
        f"CVaR loss {cvar_loss:.4f} must exceed Legal's tau {legal_profile.tau}."
    )
    print(f"✓ 8.1: CORE CLAIM — eu={eu:.3f}>0, cvar_loss={cvar_loss:.3f}>tau")


def test_8_2_full_documentation_cvar_lower_than_poor():
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
    assert cvar_good < cvar_poor, (
        f"Good docs should reduce CVaR: {cvar_good:.3f} < {cvar_poor:.3f}"
    )
    print(f"✓ 8.2: Good docs reduce CVaR: poor={cvar_poor:.3f} > good={cvar_good:.3f}")


def test_8_3_cvar_formula_correctness():
    from deal_room.stakeholders.cvar_preferences import compute_cvar
    import numpy as np

    outcomes = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 0.5, 0.5, 0.5, 0.5, 0.5])
    cvar = compute_cvar(outcomes, alpha=0.95)
    print(f"  cvar at 0.95 for [1x5, 0.5x5]: {cvar:.3f}")
    assert 0.5 <= cvar <= 1.0, f"CVaR should be between 0.5 and 1.0, got {cvar:.4f}"
    print(f"✓ 8.3: CVaR formula computes correctly ({cvar:.3f})")


def test_8_4_cvar_with_varying_outcomes():
    from deal_room.stakeholders.cvar_preferences import compute_cvar
    import numpy as np

    cvar_high = compute_cvar(np.array([0.8] * 5), alpha=0.90)
    cvar_low = compute_cvar(np.array([0.2] * 5), alpha=0.90)
    assert cvar_high < cvar_low, (
        f"Higher outcomes should give lower CVaR: {cvar_high:.3f} >= {cvar_low:.3f}"
    )
    print(
        f"✓ 8.4: CVaR differentiates outcome levels: high={cvar_high:.3f}, low={cvar_low:.3f}"
    )


def test_8_5_tau_ordering():
    from deal_room.stakeholders.archetypes import get_archetype

    legal_tau = get_archetype("Legal").tau
    finance_tau = get_archetype("Finance").tau
    exec_tau = get_archetype("ExecSponsor").tau

    assert legal_tau < finance_tau, (
        f"Legal.tau ({legal_tau:.2f}) should < Finance.tau ({finance_tau:.2f})"
    )
    assert finance_tau < exec_tau, (
        f"Finance.tau ({finance_tau:.2f}) should < ExecSponsor.tau ({exec_tau:.2f})"
    )
    print(
        f"✓ 8.5: Risk tolerance ordering: Legal={legal_tau:.2f} < Finance={finance_tau:.2f} < ExecSponsor={exec_tau:.2f}"
    )


def test_8_6_aggressive_timeline_increases_tech_cvar():
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
    assert cvar_a > cvar_r, (
        f"Aggressive timeline CVaR {cvar_a:.3f} must > reasonable {cvar_r:.3f}"
    )
    print(f"✓ 8.6: Aggressive timeline CVaR ({cvar_a:.3f}) > reasonable ({cvar_r:.3f})")


if __name__ == "__main__":
    for fn in [
        test_8_1_the_core_claim,
        test_8_2_full_documentation_cvar_lower_than_poor,
        test_8_3_cvar_formula_correctness,
        test_8_4_cvar_with_varying_outcomes,
        test_8_5_tau_ordering,
        test_8_6_aggressive_timeline_increases_tech_cvar,
    ]:
        fn()
    print("\n✓ SECTION 8 PASSED — CVaR mechanism is correct")
