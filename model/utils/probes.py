"""Probe-policy helpers for build-time instrumentation control."""

import nengo


VALID_PROBE_MODES = ("minimal", "debug")


class ProbeRegistry:
    """Create required probes always and debug probes only when requested.

    The registry keeps a small audit trail of which labeled probes were created
    and which were intentionally skipped so telemetry can explain the active
    instrumentation surface for a run.
    """

    def __init__(self, mode="debug"):
        if mode not in VALID_PROBE_MODES:
            raise ValueError(f"Unknown probe mode: {mode}")

        self.mode = mode
        self.created_probe_labels = []
        self.skipped_probe_labels = []

    def _record_created(self, label):
        self.created_probe_labels.append(label)

    def _record_skipped(self, label):
        self.skipped_probe_labels.append(label)

    def required(self, target, *, label, **probe_kwargs):
        """Create a probe that is required for normal model behavior."""
        probe = nengo.Probe(target, label=label, **probe_kwargs)
        self._record_created(label)
        return probe

    def debug(self, target, *, label, **probe_kwargs):
        """Create a probe only when the build is in debug instrumentation mode."""
        if self.mode != "debug":
            self._record_skipped(label)
            return None

        probe = nengo.Probe(target, label=label, **probe_kwargs)
        self._record_created(label)
        return probe
