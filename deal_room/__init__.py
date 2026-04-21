"""
DealRoom v3 - Research-grade OpenEnv RL environment for enterprise B2B negotiation.
"""

try:
    from deal_room.environment.dealroom_v3 import DealRoomV3
except ImportError:
    from environment.dealroom_v3 import DealRoomV3

DealRoomEnvironment = DealRoomV3

__all__ = ["DealRoomV3", "DealRoomEnvironment"]
