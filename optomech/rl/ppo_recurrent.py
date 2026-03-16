"""Backward-compatibility shim. Canonical location: train/ppo/ppo_recurrent.py"""
import os
import sys

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from train.ppo.ppo_recurrent import normalize_obs, compute_gae, recurrent_generator  # noqa: F401
