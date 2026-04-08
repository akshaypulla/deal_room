from server.semantics import SemanticAnalyzer


def test_semantic_analyzer_extracts_claims_and_artifacts():
    analyzer = SemanticAnalyzer()
    result = analyzer.analyze(
        "Here is our ROI model. The price is 180000 and the rollout is 14 weeks with GDPR controls.",
        {"documents": [{"type": "roi_model"}], "requested_artifacts": {}},
        {"finance": "finance"},
    )
    slots = {item["slot"] for item in result["claim_candidates"]}
    assert "price" in slots
    assert "timeline_weeks" in slots
    assert "roi_model" in result["artifact_matches"]


def test_semantic_analyzer_returns_known_backend():
    analyzer = SemanticAnalyzer()
    result = analyzer.analyze(
        "We can work through this together with a concrete plan.",
        {"documents": [], "requested_artifacts": {}},
        {"technical": "technical"},
    )
    assert result["backend"] in {"embedding", "tfidf", "lexical"}
