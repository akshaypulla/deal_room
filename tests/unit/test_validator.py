from server.validator import OutputValidator


def test_validator_normalizes_dynamic_targets():
    validator = OutputValidator()
    payload, confidence = validator.validate(
        '{"action_type":"direct_message","target":"finance","message":"hello"}',
        available_targets=["finance", "technical"],
    )
    assert confidence == 1.0
    assert payload["target_ids"] == ["finance"]


def test_validator_soft_rejects_unknown_target():
    validator = OutputValidator()
    payload, _ = validator.validate(
        '{"action_type":"direct_message","target":"unknown","message":"hello"}',
        available_targets=["finance"],
    )
    assert payload["malformed_action"] is True
    assert payload["error"] == "unknown_target:unknown"


def test_validator_filters_proposed_terms():
    validator = OutputValidator()
    payload, _ = validator.validate(
        '{"action_type":"group_proposal","target":"all","message":"go","proposed_terms":{"price":10,"junk":1}}',
        available_targets=["finance"],
    )
    assert payload["proposed_terms"] == {"price": 10}
