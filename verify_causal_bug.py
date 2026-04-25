#!/usr/bin/env python3
"""
Verify the causal reward bug - StateSnapshot._snapshot() doesn't include
cross_stakeholder_echoes, so score_causal_observable always returns 0.
"""

import sys

sys.path.insert(0, "/Users/akshaypulla/Documents/deal_room")

from deal_room_v4_clean import (
    DealRoomV4,
    StateSnapshot,
    score_causal_observable,
    STANDARD_STAKEHOLDERS,
)
from models import DealRoomAction

print("=" * 70)
print("CONFIRMED BUG: r^causal always returns 0.0")
print("=" * 70)

env = DealRoomV4()
obs = env.reset(seed=999, task_id="aligned")

action = DealRoomAction(
    action_type="send_document(roi_model)_to_finance",
    target="Finance",
    target_ids=["Finance"],
    documents=[{"name": "roi_model", "content": "ROI"}],
)

state_before = env._snapshot()

obs, reward, done, info = env.step(action)

state_after = env._snapshot()

print(f"\nobs.cross_stakeholder_echoes: {obs.cross_stakeholder_echoes}")
print(f"state_after type: {type(state_after)}")
print(f"state_after fields: {[f for f in dir(state_after) if not f.startswith('_')]}")
print(f"\nscore_causal_observable(action, state_before, state_after):")
print(f"  → Returns: {score_causal_observable(action, state_before, state_after)}")

print(f"\nRoot cause:")
print(f"  StateSnapshot class does NOT have cross_stakeholder_echoes field")
print(f"  _snapshot() returns StateSnapshot, which is missing this attribute")
print(f"  score_causal_observable accesses state_after.cross_stakeholder_echoes")
print(f"  → AttributeError caught, returns 0.0")

print(f"\nImpact:")
print(f"  r^causal is PERMANENTLY 0.0 for all actions")
print(f"  The observable echo propagation signal is never rewarded")
print(f"  GRPO advantage computation is affected (15% of reward weight)")
print(f"  This is a BUG, not a guardrail - causal reward is broken")
print(f"\nFix: Add cross_stakeholder_echoes to StateSnapshot._snapshot()")
