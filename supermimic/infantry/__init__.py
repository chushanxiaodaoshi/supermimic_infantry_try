"""Fudan balance-infantry robot integration."""

from .control import policy_action_to_control
from .spec import INFANTRY_SPEC

__all__ = ["INFANTRY_SPEC", "policy_action_to_control"]
