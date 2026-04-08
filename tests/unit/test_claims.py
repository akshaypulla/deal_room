from server.claims import CommitmentLedger


def test_commitment_ledger_flags_numeric_contradiction():
    ledger = CommitmentLedger()
    first = ledger.ingest(["finance"], [{"slot": "price", "value": 180000, "text": "180000"}], {})
    second = ledger.ingest(["finance"], [{"slot": "price", "value": 230000, "text": "230000"}], {})
    assert first["contradictions"] == []
    assert len(second["contradictions"]) == 1


def test_commitment_ledger_trims_history():
    ledger = CommitmentLedger(max_claims=3)
    for idx in range(5):
        ledger.ingest(["finance"], [{"slot": "timeline_weeks", "value": 10 + idx, "text": str(idx)}], {})
    assert len(ledger.claims) == 3
