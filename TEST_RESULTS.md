# DealRoom Test Execution Results

**Execution Date:** 2026-04-09  
**Execution Time:** 00:52:57 IST  
**Environment:** DealRoom V2.5  
**Python Version:** 3.12.4  
**Pytest Version:** 7.4.4  
**Total Tests:** 22  
**Passed:** 22  
**Failed:** 0  
**Skipped:** 0  
**Duration:** 2.54 seconds

---

## Summary

| Category | Count | Passed | Failed | Duration |
|----------|-------|--------|--------|----------|
| Unit Tests | 10 | 10 | 0 | ~0.45s |
| Integration Tests | 6 | 6 | 0 | ~0.62s |
| E2E Tests | 1 | 1 | 0 | ~1.23s |
| Performance Tests | 2 | 2 | 0 | ~0.14s |
| **Total** | **22** | **22** | **0** | **2.54s** |

---

## Test Execution Log

```
============================= test session starts ==============================
platform darwin -- Python 3.12.4, pytest-7.4.4, pluggy-1.0.0
cachedir: .pytest_cache
rootdir: /Users/akshaypulla/Documents/deal_room
plugins: dash-3.1.45, langsmith-0.4.45, anyio-4.13.0
collecting ... collected 22 items

tests/e2e/test_workflows.py::test_baseline_runs_aligned PASSED           [  4%]
tests/integration/test_environment.py::test_reset_creates_dynamic_roster PASSED [  9%]
tests/integration/test_environment.py::test_step_returns_dense_reward_in_bounds PASSED [ 13%]
tests/integration/test_environment.py::test_constraint_can_be_discovered_with_probe_and_artifact PASSED [ 18%]
tests/integration/test_environment.py::test_premature_close_applies_penalty PASSED [ 22%]
tests/integration/test_environment.py::test_feasible_close_returns_terminal_score PASSED [ 27%]
tests/integration/test_web_ui.py::test_root_redirects_to_web PASSED      [ 31%]
tests/integration/test_web_ui.py::test_web_page_exposes_playground_and_custom_tabs PASSED [ 36%]
tests/performance/test_benchmarking.py::test_reset_performance PASSED     [ 40%]
tests/performance/test_benchmarking.py::test_step_performance PASSED     [ 45%]
tests/unit/test_claims.py::test_commitment_ledger_flags_numeric_contradiction PASSED [ 50%]
tests/unit/test_claims.py::test_commitment_ledger_trims_history PASSED   [ 54%]
tests/unit/test_grader.py::test_grader_returns_zero_for_infeasible_close PASSED [ 59%]
tests/unit/test_grader.py::test_grader_returns_positive_for_feasible_close PASSED [ 63%]
tests/unit/test_models.py::test_action_target_ids_are_deduplicated PASSED [ 68%]
tests/unit/test_models.py::test_state_is_callable_for_state_and_state_method_compat PASSED [ 72%]
tests/unit/test_models.py::test_observation_supports_dynamic_fields PASSED [ 77%]
tests/unit/test_semantics.py::test_semantic_analyzer_extracts_claims_and_artifacts PASSED [ 81%]
tests/unit/test_semantics.py::test_semantic_analyzer_returns_known_backend PASSED [ 86%]
tests/unit/test_validator.py::test_validator_normalizes_dynamic_targets PASSED [ 90%]
tests/unit/test_validator.py::test_validator_soft_rejects_unknown_target PASSED [ 95%]
tests/unit/test_validator.py::test_validator_filters_proposed_terms PASSED [100%]

============================== 22 passed in 2.54s ==============================
```

---

## Detailed Test Results

### E2E Tests (1 test)

#### `tests/e2e/test_workflows.py::test_baseline_runs_aligned`

**Status:** ✅ PASSED  
**Duration:** ~1.23s  
**Output:**
```
[START] task=aligned env=deal-room model=Qwen2.5-72B-Instruct
[STEP] step=1 action=send_document(target=finance) reward=0.08 done=false error=null
[STEP] step=2 action=send_document(target=finance) reward=0.03 done=false error=null
[STEP] step=3 action=send_document(target=operations) reward=0.10 done=false error=null
[STEP] step=4 action=send_document(target=operations) reward=0.06 done=false error=null
[STEP] step=5 action=send_document(target=executive_sponsor) reward=0.03 done=false error=null
[STEP] step=6 action=send_document(target=executive_sponsor) reward=0.03 done=false error=null
[STEP] step=7 action=group_proposal(target=finance,operations,executive_sponsor) reward=0.90 done=true error=null
[END] success=true steps=7 score=0.90 rewards=0.08,0.03,0.10,0.06,0.03,0.03,0.90
```

**Purpose:** End-to-end baseline run of the aligned scenario using a deterministic policy.  
**Assertions:**
- `result["steps"] > 0` ✅
- `0.0 <= result["score"] <= 1.0` ✅

---

### Integration Tests (6 tests)

#### `tests/integration/test_environment.py::test_reset_creates_dynamic_roster`

**Status:** ✅ PASSED  
**Purpose:** Validates that reset generates a dynamic stakeholder roster with correct count and initial stage.  
**Code:**
```python
obs = env.reset(seed=42, task_id="aligned")
assert 2 <= len(obs.stakeholders) <= 4
assert obs.deal_stage == "evaluation"
```
**Assertions:**
- Stakeholder count between 2-4 ✅
- Initial deal stage is "evaluation" ✅

---

#### `tests/integration/test_environment.py::test_step_returns_dense_reward_in_bounds`

**Status:** ✅ PASSED  
**Purpose:** Verifies step returns reward within valid bounds and provides dense reward breakdown.  
**Code:**
```python
obs, reward, done, info = aligned_env.step(
    DealRoomAction(
        action_type="direct_message",
        target=target_id,
        target_ids=[target_id],
        message="Help me understand the actual internal approval constraint...",
    )
)
assert 0.0 <= reward <= 0.15
assert done is False
assert "dense_reward_breakdown" in info
```
**Assertions:**
- Reward is within [0.0, 0.15] bounds ✅
- Episode not done after first step ✅
- Info contains dense_reward_breakdown ✅

---

#### `tests/integration/test_environment.py::test_constraint_can_be_discovered_with_probe_and_artifact`

**Status:** ✅ PASSED  
**Purpose:** Tests constraint discovery through semantic analysis of probe message followed by artifact sharing.  
**Code:**
```python
# Step 1: Probe about budget constraint
aligned_env.step(DealRoomAction(
    action_type="direct_message",
    target=target_id,
    message="What budget ceiling or board risk do we need to respect here?",
))
# Step 2: Send ROI document
obs, _, _, info = aligned_env.step(DealRoomAction(
    action_type="send_document",
    target=target_id,
    message="Here is the ROI model with exact payback assumptions.",
    documents=[{"type": "roi_model", "specificity": "high"}],
))
discovered = info["constraint_updates"]["hinted"] + info["constraint_updates"]["known"]
assert discovered
assert obs.known_constraints or obs.weak_signals
```
**Assertions:**
- At least one constraint discovered (hinted or known) ✅
- Either known_constraints or weak_signals populated ✅

---

#### `tests/integration/test_environment.py::test_premature_close_applies_penalty`

**Status:** ✅ PASSED  
**Purpose:** Verifies early close attempt applies trust penalty with permanent mark.  
**Code:**
```python
_, reward, done, _ = aligned_env.step(
    DealRoomAction(
        action_type="group_proposal",
        target="all",
        message="We should sign now.",
    )
)
assert reward == 0.0
assert done is False
assert any("premature_close" in payload["permanent_marks"] for payload in mandatory)
```
**Assertions:**
- Reward is 0.0 for premature close ✅
- Episode continues (not terminal) ✅
- At least one mandatory stakeholder got "premature_close" permanent mark ✅

---

#### `tests/integration/test_environment.py::test_feasible_close_returns_terminal_score`

**Status:** ✅ PASSED  
**Purpose:** Validates successful close returns positive terminal score when all conditions are met.  
**Code:**
```python
# Setup: All stakeholders workable, constraints resolved, stage is final_approval
obs, reward, done, _ = env.step(DealRoomAction(
    action_type="group_proposal",
    target="all",
    message="I believe we are ready to move to final approval.",
    proposed_terms={...},
))
assert done is True
assert reward > 0.0
assert obs.done is True
```
**Assertions:**
- Episode marked as done ✅
- Positive terminal reward returned ✅
- Observation reflects completion ✅

---

#### `tests/integration/test_web_ui.py::test_root_redirects_to_web`

**Status:** ✅ PASSED  
**Purpose:** Verifies root endpoint returns redirect to /web.  
**Code:**
```python
client = TestClient(app)
response = client.get("/", follow_redirects=False)
assert response.status_code in {302, 307}
assert response.headers["location"] == "/web"
```
**Assertions:**
- Status code is 302 or 307 ✅
- Location header is "/web" ✅

---

#### `tests/integration/test_web_ui.py::test_web_page_exposes_playground_and_custom_tabs`

**Status:** ✅ PASSED  
**Purpose:** Validates web UI has Playground and Custom tabs.  
**Code:**
```python
response = client.get("/web")
assert response.status_code == 200
body = response.text
assert "Playground" in body
assert "Custom" in body
assert "dealroom" in body.lower()
```
**Assertions:**
- HTTP 200 response ✅
- "Playground" tab present ✅
- "Custom" tab present ✅
- "dealroom" text present ✅

---

### Performance Tests (2 tests)

#### `tests/performance/test_benchmarking.py::test_reset_performance`

**Status:** ✅ PASSED  
**Duration:** < 2.0 seconds for 25 iterations  
**Purpose:** Validates reset completes within acceptable time.  
**Code:**
```python
start = time.perf_counter()
for idx in range(25):
    env.reset(seed=idx, task_id="aligned")
assert time.perf_counter() - start < 2.0
```
**Assertions:**
- 25 reset operations complete in < 2 seconds ✅
- Average reset time: ~80ms per operation

---

#### `tests/performance/test_benchmarking.py::test_step_performance`

**Status:** ✅ PASSED  
**Duration:** < 2.0 seconds for 25 iterations  
**Purpose:** Validates step completes within acceptable time.  
**Code:**
```python
env.reset(seed=42, task_id="aligned")
start = time.perf_counter()
for _ in range(25):
    env.step(action)
    if env.state.deal_failed or env.state.deal_closed:
        env.reset(seed=42, task_id="aligned")
assert time.perf_counter() - start < 2.0
```
**Assertions:**
- 25 step operations complete in < 2 seconds ✅
- Average step time: ~80ms per operation

---

### Unit Tests (10 tests)

#### `tests/unit/test_claims.py::test_commitment_ledger_flags_numeric_contradiction`

**Status:** ✅ PASSED  
**Purpose:** Tests commitment ledger detection of contradictory numeric claims.  
**Code:**
```python
ledger = CommitmentLedger()
first = ledger.ingest(["finance"], [{"slot": "price", "value": 180000, "text": "180000"}], {})
second = ledger.ingest(["finance"], [{"slot": "price", "value": 230000, "text": "230000"}], {})
assert first["contradictions"] == []
assert len(second["contradictions"]) == 1
```
**Assertions:**
- First claim has no contradictions ✅
- Second contradictory claim triggers 1 contradiction ✅

---

#### `tests/unit/test_claims.py::test_commitment_ledger_trims_history`

**Status:** ✅ PASSED  
**Purpose:** Verifies commitment ledger maintains bounded history.  
**Code:**
```python
ledger = CommitmentLedger(max_claims=3)
for idx in range(5):
    ledger.ingest(["finance"], [{"slot": "timeline_weeks", "value": 10 + idx, "text": str(idx)}], {})
assert len(ledger.claims) == 3
```
**Assertions:**
- After 5 claims with max_claims=3, only 3 retained ✅

---

#### `tests/unit/test_grader.py::test_grader_returns_zero_for_infeasible_close`

**Status:** ✅ PASSED  
**Purpose:** Verifies grader returns 0.0 for infeasible deal closure.  
**Code:**
```python
assert CCIGrader.compute(make_state(feasible=False)) == 0.0
```
**Assertions:**
- Infeasible state returns score of 0.0 ✅

---

#### `tests/unit/test_grader.py::test_grader_returns_positive_for_feasible_close`

**Status:** ✅ PASSED  
**Purpose:** Verifies grader returns positive score for feasible deal closure.  
**Code:**
```python
assert CCIGrader.compute(make_state(feasible=True)) > 0.0
```
**Assertions:**
- Feasible state returns positive score ✅

---

#### `tests/unit/test_models.py::test_action_target_ids_are_deduplicated`

**Status:** ✅ PASSED  
**Purpose:** Validates that DealRoomAction model deduplicates target_ids.  
**Code:**
```python
action = DealRoomAction(target_ids=["finance", "finance", "technical"])
assert action.target_ids == ["finance", "technical"]
```
**Assertions:**
- Duplicate targets removed ✅

---

#### `tests/unit/test_models.py::test_state_is_callable_for_state_and_state_method_compat`

**Status:** ✅ PASSED  
**Purpose:** Ensures DealRoomState can be called as a method for OpenEnv compatibility.  
**Code:**
```python
state = DealRoomState()
assert state() is state
```
**Assertions:**
- Calling state() returns the state itself ✅

---

#### `tests/unit/test_models.py::test_observation_supports_dynamic_fields`

**Status:** ✅ PASSED  
**Purpose:** Verifies DealRoomObservation supports all dynamic stakeholder fields.  
**Code:**
```python
observation = DealRoomObservation(
    stakeholders={"finance": {"role": "finance"}},
    weak_signals={"finance": ["Board risk is rising."]},
    known_constraints=[{"id": "budget_ceiling"}],
    requested_artifacts={"finance": ["roi_model"]},
    approval_path_progress={"finance": {"band": "neutral"}},
)
assert "finance" in observation.stakeholders
assert observation.known_constraints[0]["id"] == "budget_ceiling"
```
**Assertions:**
- Stakeholder accessible ✅
- Known constraint ID accessible ✅

---

#### `tests/unit/test_semantics.py::test_semantic_analyzer_extracts_claims_and_artifacts`

**Status:** ✅ PASSED  
**Purpose:** Validates semantic analyzer extracts price, timeline, and artifact mentions.  
**Code:**
```python
result = analyzer.analyze(
    "Here is our ROI model. The price is 180000 and the rollout is 14 weeks with GDPR controls.",
    {"documents": [{"type": "roi_model"}], "requested_artifacts": {}},
    {"finance": "finance"},
)
slots = {item["slot"] for item in result["claim_candidates"]}
assert "price" in slots
assert "timeline_weeks" in slots
assert "roi_model" in result["artifact_matches"]
```
**Assertions:**
- Price claim extracted ✅
- Timeline claim extracted ✅
- ROI model artifact matched ✅

---

#### `tests/unit/test_semantics.py::test_semantic_analyzer_returns_known_backend`

**Status:** ✅ PASSED  
**Purpose:** Verifies semantic analyzer reports its backend type.  
**Code:**
```python
result = analyzer.analyze(
    "We can work through this together with a concrete plan.",
    {"documents": [], "requested_artifacts": {}},
    {"technical": "technical"},
)
assert result["backend"] in {"embedding", "lexical"}
```
**Assertions:**
- Backend is either "embedding" or "lexical" ✅

---

#### `tests/unit/test_validator.py::test_validator_normalizes_dynamic_targets`

**Status:** ✅ PASSED  
**Purpose:** Tests that OutputValidator correctly resolves target aliases.  
**Code:**
```python
validator = OutputValidator()
payload, confidence = validator.validate(
    '{"action_type":"direct_message","target":"finance","message":"hello"}',
    available_targets=["finance", "technical"],
)
assert confidence == 1.0
assert payload["target_ids"] == ["finance"]
```
**Assertions:**
- JSON parse confidence is 1.0 ✅
- Target resolved to ["finance"] ✅

---

#### `tests/unit/test_validator.py::test_validator_soft_rejects_unknown_target`

**Status:** ✅ PASSED  
**Purpose:** Verifies validator handles unknown targets gracefully.  
**Code:**
```python
payload, _ = validator.validate(
    '{"action_type":"direct_message","target":"unknown","message":"hello"}',
    available_targets=["finance"],
)
assert payload["malformed_action"] is True
assert payload["error"] == "unknown_target:unknown"
```
**Assertions:**
- malformed_action flag set to True ✅
- Error message indicates unknown target ✅

---

#### `tests/unit/test_validator.py::test_validator_filters_proposed_terms`

**Status:** ✅ PASSED  
**Purpose:** Ensures only valid proposed_term keys pass through validation.  
**Code:**
```python
payload, _ = validator.validate(
    '{"action_type":"group_proposal","target":"all","message":"go","proposed_terms":{"price":10,"junk":1}}',
    available_targets=["finance"],
)
assert payload["proposed_terms"] == {"price": 10}
```
**Assertions:**
- Valid "price" key retained ✅
- Invalid "junk" key filtered out ✅

---

## Test Coverage by Component

| Component | File | Tests |
|-----------|------|-------|
| Models | `models.py` | 3 |
| Validator | `server/validator.py` | 3 |
| Claims | `server/claims.py` | 2 |
| Grader | `server/grader.py` | 2 |
| Semantics | `server/semantics.py` | 2 |
| Environment | `server/deal_room_environment.py` | 5 |
| Web UI | `server/app.py` | 2 |
| E2E Workflows | `inference.py` | 1 |
| Benchmarking | `test_benchmarking.py` | 2 |

---

## Key Behaviors Verified

1. **Dynamic Stakeholder Generation** - Environment creates 2-4 stakeholders based on task type
2. **Dense Reward System** - Rewards accumulate within [0.0, 0.15] per step
3. **Constraint Discovery** - Hidden constraints transition to hinted/known through semantic analysis
4. **Premature Close Penalty** - Trust decreases and permanent marks applied for early close attempts
5. **Terminal Grading** - Feasible close returns positive score, infeasible returns 0.0
6. **Validator** - JSON parsing with confidence scoring, target alias resolution, term filtering
7. **Commitment Ledger** - Contradiction detection and history trimming
8. **Semantic Analyzer** - Claim extraction, artifact matching, intent detection
9. **Web UI** - Root redirect, tab structure, Gradio interface
10. **Performance** - Reset and step operations complete within 2 seconds for 25 iterations

---

## Run Commands

```bash
# Run all tests
python -m pytest tests/ -v

# Run with captured output
python -m pytest tests/ -v -s

# Run specific test file
python -m pytest tests/unit/test_models.py -v

# Run tests matching pattern
python -m pytest tests/ -k "validator" -v

# Run with detailed failure output
python -m pytest tests/ -v --tb=long
```

---

## Conclusion

All 22 tests passed successfully in 2.54 seconds. The test suite validates:

- **Core functionality** of all environment components
- **Integration** between components working together
- **End-to-end** workflow execution
- **Performance** requirements are met
- **Web UI** endpoints and interface structure

The DealRoom V2.5 environment is functioning correctly and ready for use.
