"""Helpers for loading and saving optional runtime-profile defaults."""

import copy
import json
from pathlib import Path

RUNTIME_PROFILE_SCHEMA_VERSION = 1

DEFAULT_RUNTIME_PROFILE = {
    "schema_version": RUNTIME_PROFILE_SCHEMA_VERSION,
    "training": {
        "mode": "single_pass",
        "scheduled_training_enabled": False,
        "token_duration": 0.02,
        "token_duration_source": "default",
        "calibrated": False,
    },
    "evaluation": {
        "mode": "streaming",
        "min_context_tokens": 1,
        "stride": 1,
    },
    "runtime": {
        "default_step_time": 0.02,
    },
    "opencl": {
        "platform_index": None,
        "device_index": None,
    },
}


def default_runtime_profile():
    """Return a deep copy of the runtime-profile template."""
    return copy.deepcopy(DEFAULT_RUNTIME_PROFILE)


def runtime_profile_path(base_dir):
    """Path for the local machine-specific runtime profile."""
    return Path(base_dir) / "config" / "runtime_profile.json"


def runtime_profile_example_path(base_dir):
    """Path for the committed example runtime profile."""
    return Path(base_dir) / "config" / "runtime_profile.example.json"


def load_runtime_profile(base_dir):
    """Load the local runtime profile if one exists."""
    profile_path = runtime_profile_path(base_dir)
    if not profile_path.is_file():
        return None

    with profile_path.open("r", encoding="utf-8") as profile_file:
        return json.load(profile_file)


def save_runtime_profile(base_dir, document):
    """Persist a machine-local runtime profile document."""
    profile_path = runtime_profile_path(base_dir)
    profile_path.parent.mkdir(parents=True, exist_ok=True)

    with profile_path.open("w", encoding="utf-8") as profile_file:
        json.dump(document, profile_file, indent=2)

    return profile_path