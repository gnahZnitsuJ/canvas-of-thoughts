"""Committed runtime defaults used when no explicit local override is active."""

# This is the fallback simulator advancement interval when the optional local
# runtime profile does not provide one. It also supplies the uncalibrated token
# duration and is recorded in checkpoint compatibility metadata. Keep calibrated
# machine-local values in runtime_profile.json instead of editing this default.
DEFAULT_STEP_TIME_SECONDS = 0.02
