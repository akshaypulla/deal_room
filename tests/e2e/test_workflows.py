import pytest

from inference import run_task


def test_baseline_runs_aligned():
    result = run_task("aligned", seed=7)
    assert result["steps"] > 0
    assert 0.0 < result["score"] < 1.0


@pytest.mark.parametrize("seed", [3, 4, 6, 8])
def test_hostile_baseline_survives_historically_failing_seeds(seed: int):
    result = run_task("hostile_acquisition", seed=seed)
    assert result["success"] is True
    assert 0.0 < result["score"] < 1.0


def test_inference_logs_match_required_markers(capsys):
    run_task("aligned", seed=7)
    output = capsys.readouterr().out.strip().splitlines()
    assert output[0].startswith("[START] task=aligned env=deal-room model=")
    assert any(line.startswith("[STEP] step=") for line in output)
    assert output[-1].startswith("[END] success=")
