from inference import run_task


def test_baseline_runs_aligned():
    result = run_task("aligned", seed=7)
    assert result["steps"] > 0
    assert 0.0 <= result["score"] <= 1.0
