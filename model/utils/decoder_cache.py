"""Persistent Nengo decoder-cache policy, instrumentation, and inspection.

Nengo remains responsible for serializing exact decoder arrays and deriving
keys from solver inputs. This module gives that cache a durable Canvas-owned
location, separates incompatible framework versions, and reports whether the
builder reused or solved the learned decoder entries.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import re
import sys
from pathlib import Path
from time import perf_counter

import nengo
import nengo_ocl
import nengo_spa
import numpy as np
import scipy
from nengo.builder import Model as NengoBuildModel
from nengo.cache import DecoderCache, Fingerprint, NoDecoderCache
from nengo_ocl.builder import Builder as OclBuilder

from config import cache_defaults

CACHE_INTEGRITY_INFO_KEY = "canvas_decoder_integrity_v1"


def _safe_version(value):
    """Return a path-safe package version without hiding compatibility data."""
    return re.sub(r"[^A-Za-z0-9._-]+", "-", str(value))


def framework_versions():
    """Return framework versions that bound cache binary compatibility."""
    return {
        "nengo": nengo.__version__,
        "nengo_ocl": nengo_ocl.__version__,
        "nengo_spa": nengo_spa.__version__,
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "python": f"{sys.version_info.major}.{sys.version_info.minor}",
        "platform": f"{platform.system()}-{platform.machine()}",
    }


def framework_namespace():
    """Return the version-scoped directory name for decoder artifacts."""
    versions = framework_versions()
    return "_".join(
        f"{name}-{_safe_version(version)}"
        for name, version in sorted(versions.items())
    )


def resolve_decoder_cache_root(environ=None):
    """Resolve the Canvas cache root without creating it.

    The environment override is useful for CI and removable storage. On
    Windows the default lives under LocalAppData; other systems follow the
    conventional per-user ``.cache`` location.
    """
    environ = os.environ if environ is None else environ
    override = environ.get(cache_defaults.DECODER_CACHE_ENV_VAR)
    if override:
        override_path = Path(override).expanduser()
        if not override_path.is_absolute():
            raise ValueError(
                f"{cache_defaults.DECODER_CACHE_ENV_VAR} must be an absolute path"
            )
        return override_path

    local_app_data = environ.get("LOCALAPPDATA")
    if local_app_data:
        return Path(local_app_data) / "canvas-of-thoughts" / "decoder-cache"
    return Path.home() / ".cache" / "canvas-of-thoughts" / "decoder-cache"


def resolve_decoder_cache_path(environ=None):
    """Resolve the active framework-version cache directory without writing."""
    return (
        resolve_decoder_cache_root(environ)
        / cache_defaults.DECODER_CACHE_SCHEMA_VERSION
        / framework_namespace()
    )


def _directory_size(path):
    if not path.exists():
        return 0
    return sum(
        item.stat().st_size
        for item in path.rglob("*")
        if item.is_file()
    )


def inspect_decoder_cache(environ=None):
    """Describe the persistent cache without creating or opening its index."""
    path = resolve_decoder_cache_path(environ)
    files = list(path.rglob("*")) if path.exists() else []
    return {
        "mode": "inspection",
        "path": str(path),
        "exists": path.is_dir(),
        "size_bytes": _directory_size(path),
        "file_count": sum(item.is_file() for item in files),
        "framework_versions": framework_versions(),
        "max_size_bytes": cache_defaults.DECODER_CACHE_MAX_SIZE_BYTES,
        "environment_override": cache_defaults.DECODER_CACHE_ENV_VAR,
    }


def _connection_name(connection, learning_connection_details):
    learned_details = learning_connection_details.get(id(connection))
    if learned_details is not None:
        return learned_details, True
    label = getattr(connection, "label", None)
    return {"connection": label or type(connection).__name__}, False


def _array_digest(value):
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(str(array.shape).encode("ascii"))
    digest.update(array.data)
    return digest.hexdigest()


def _rng_digest(rng):
    state = rng.get_state()
    digest = hashlib.sha256()
    digest.update(state[0].encode("ascii"))
    digest.update(np.ascontiguousarray(state[1]).data)
    digest.update(str(state[2:]).encode("ascii"))
    return digest.hexdigest()


class _TrackingMixin:
    """Collect cache outcomes while preserving Nengo's solver contract."""

    def _start_tracking(
        self,
        *,
        mode,
        path,
        learning_connections,
        learning_connection_metadata=(),
    ):
        self.mode = mode
        self.reported_path = path
        metadata = list(learning_connection_metadata)
        self._learning_connection_details = {
            id(connection): {
                **(metadata[index] if index < len(metadata) else {}),
                "connection": (
                    metadata[index].get("stable_id")
                    if index < len(metadata) and metadata[index].get("stable_id")
                    else f"learning_connection_{index}:"
                    f"{getattr(connection, 'label', None) or 'unlabeled'}"
                ),
            }
            for index, connection in enumerate(learning_connections)
        }
        self._reused = 0
        self._solved = 0
        self._failed = 0
        self._corrupt_reads = 0
        self._write_failures = 0
        self._invalidated_after_corruption = False
        self._repair_failed = False
        self._degraded_reason = None
        self._reuse_seconds = 0.0
        self._solve_seconds = 0.0
        self._learned_events = []

    def _track_solver(self, solver_fn, wrapped_solver):
        state = {"solved": False, "completed": False, "result": None}

        def observed_solver(*solver_args, **solver_kwargs):
            state["solved"] = True
            result = solver_fn(*solver_args, **solver_kwargs)
            result = self.attach_integrity(result)
            state["completed"] = True
            state["result"] = result
            return result

        # Disk caches need to be wrapped around the observing callable so a
        # cache hit can skip it. No-cache implementations pass it through.
        cached_solver = wrapped_solver(observed_solver)

        def tracked(
            conn,
            gain,
            bias,
            x,
            targets,
            rng=np.random,
            **call_kwargs,
        ):
            state["solved"] = False
            state["completed"] = False
            state["result"] = None
            outcome = None
            start = perf_counter()
            details, is_learning = _connection_name(
                conn,
                self._learning_connection_details,
            )
            if is_learning:
                try:
                    details = {
                        **details,
                        "nengo_cache_key": self.cache_key_for_call(
                            conn,
                            gain,
                            bias,
                            x,
                            targets,
                            rng,
                        ),
                        "nengo_key_inputs": {
                            "solver": str(Fingerprint(conn.solver)),
                            "neuron_type": str(
                                Fingerprint(conn.pre_obj.neuron_type)
                            ),
                            "gain": _array_digest(gain),
                            "bias": _array_digest(bias),
                            "evaluation": _array_digest(x),
                            "targets": _array_digest(targets),
                            "rng": _rng_digest(rng),
                        },
                    }
                except Exception:
                    # Key reporting is diagnostic only. Nengo's wrapped solver
                    # remains the authority for cacheability and fallback.
                    details = {**details, "nengo_cache_key": None}
            try:
                result = cached_solver(
                    conn,
                    gain,
                    bias,
                    x,
                    targets,
                    rng=rng,
                    **call_kwargs,
                )
            except Exception as exc:
                if state["completed"]:
                    # The scientific solve succeeded; only cache persistence
                    # failed. Return the complete decoder and quarantine the
                    # namespace after Nengo releases its index context.
                    self._write_failures += 1
                    self._degraded_reason = f"cache write failed: {type(exc).__name__}"
                    result = state["result"]
                    outcome = "solved-write-failed"
                elif state["solved"]:
                    # The numerical solver itself failed; hiding that would
                    # turn a scientific failure into a false cache recovery.
                    self._failed += 1
                    if is_learning:
                        self._learned_events.append(
                            {**details, "outcome": "failed"}
                        )
                    raise

                else:
                    # Nengo 3.2 handles missing entries and some structural
                    # errors, but malformed NCO payloads can raise other read
                    # exceptions. Recompute the complete decoder rather than
                    # applying partial state.
                    self._corrupt_reads += 1
                    result = observed_solver(
                        conn,
                        gain,
                        bias,
                        x,
                        targets,
                        rng=rng,
                        **call_kwargs,
                    )
                    outcome = "corrupt-recomputed"

            if not state["solved"] and not self.integrity_is_valid(result):
                # A parseable NCO/NPY payload can still contain bit-flipped
                # values. Treat missing or mismatched Canvas integrity metadata
                # as untrusted and regenerate the entire decoder.
                self._corrupt_reads += 1
                result = observed_solver(
                    conn,
                    gain,
                    bias,
                    x,
                    targets,
                    rng=rng,
                    **call_kwargs,
                )
                outcome = "unverified-recomputed"

            elapsed = perf_counter() - start
            if state["solved"]:
                self._solved += 1
                self._solve_seconds += elapsed
                outcome = outcome or "solved"
            else:
                self._reused += 1
                self._reuse_seconds += elapsed
                outcome = "reused"
            if is_learning:
                self._learned_events.append(
                    {
                        **details,
                        "outcome": outcome,
                        "seconds": elapsed,
                    }
                )
            return result

        return tracked

    def telemetry(self):
        """Return a JSON-safe build report for console and run telemetry."""
        readonly = bool(getattr(self, "readonly", False))
        cache_available = (
            self.mode == "off" or getattr(self, "_index", None) is not None
        )
        try:
            size_bytes = self.get_size_in_bytes()
        except OSError:
            size_bytes = None
        return {
            "mode": self.mode,
            "path": str(self.reported_path) if self.reported_path else None,
            "persistent": self.mode != "off",
            "cache_available": cache_available,
            "readonly": readonly,
            "write_enabled": (
                self.mode != "off" and cache_available and not readonly
            ),
            "reused_solver_calls": self._reused,
            "solved_solver_calls": self._solved,
            "failed_solver_calls": self._failed,
            "corrupt_cache_reads": self._corrupt_reads,
            "cache_write_failures": self._write_failures,
            "invalidated_after_corruption": self._invalidated_after_corruption,
            "repair_failed": self._repair_failed,
            "degraded_reason": self._degraded_reason,
            "reuse_seconds": self._reuse_seconds,
            "solve_seconds": self._solve_seconds,
            "learned_connections": list(self._learned_events),
            "size_bytes": size_bytes,
            "max_size_bytes": cache_defaults.DECODER_CACHE_MAX_SIZE_BYTES,
            "framework_versions": framework_versions(),
        }

    def finalize_after_build(self):
        """Quarantine a namespace that produced an unreadable cache record."""
        unavailable = self.mode != "off" and getattr(self, "_index", None) is None
        if not (self._corrupt_reads or self._write_failures or unavailable):
            return
        try:
            if unavailable:
                repair_cache = DecoderCache(cache_dir=self.reported_path)
                repair_cache.invalidate()
            else:
                self.invalidate()
        except Exception as exc:
            self._repair_failed = True
            self._degraded_reason = (
                f"cache quarantine failed: {type(exc).__name__}"
            )
        else:
            self._invalidated_after_corruption = True

    def attach_integrity(self, result):
        return result

    def integrity_is_valid(self, result):
        return True

    def cache_key_for_call(self, conn, gain, bias, x, targets, rng):
        return None


class TrackedDecoderCache(_TrackingMixin, DecoderCache):
    """Nengo's exact disk cache with bounded storage and outcome telemetry."""

    def __init__(
        self,
        cache_dir,
        *,
        mode="auto",
        learning_connections=(),
        learning_connection_metadata=(),
    ):
        super().__init__(readonly=False, cache_dir=cache_dir)
        self._start_tracking(
            mode=mode,
            path=Path(cache_dir),
            learning_connections=learning_connections,
            learning_connection_metadata=learning_connection_metadata,
        )

    def wrap_solver(self, solver_fn):
        return self._track_solver(solver_fn, super().wrap_solver)

    def cache_key_for_call(self, conn, gain, bias, x, targets, rng):
        """Expose Nengo's exact opaque key for learned-entry diagnostics."""
        return self._get_cache_key(
            conn.solver,
            conn.pre_obj.neuron_type,
            gain,
            bias,
            x,
            targets,
            rng,
        )

    def attach_integrity(self, result):
        decoders, info = result
        if not isinstance(info, dict):
            return result
        info = dict(info)
        array = np.ascontiguousarray(decoders)
        info[CACHE_INTEGRITY_INFO_KEY] = {
            "digest": _array_digest(array),
            "dtype": str(array.dtype),
            "shape": list(array.shape),
        }
        return decoders, info

    def integrity_is_valid(self, result):
        decoders, info = result
        if not isinstance(info, dict):
            return False
        expected = info.get(CACHE_INTEGRITY_INFO_KEY)
        if not isinstance(expected, dict):
            return False
        array = np.ascontiguousarray(decoders)
        return expected == {
            "digest": _array_digest(array),
            "dtype": str(array.dtype),
            "shape": list(array.shape),
        }

    def shrink(self, limit=None):
        # Nengo calls shrink() after each top-level network build. Override its
        # user-wide 512 MB default with this project's committed cache policy.
        return super().shrink(
            cache_defaults.DECODER_CACHE_MAX_SIZE_BYTES if limit is None else limit
        )


class TrackedNoDecoderCache(_TrackingMixin, NoDecoderCache):
    """No-cache mode that still reports which solver work was performed."""

    def __init__(
        self,
        *,
        learning_connections=(),
        learning_connection_metadata=(),
        reported_path=None,
        mode="off",
        degraded_reason=None,
    ):
        self._start_tracking(
            mode=mode,
            path=reported_path,
            learning_connections=learning_connections,
            learning_connection_metadata=learning_connection_metadata,
        )
        self._degraded_reason = degraded_reason

    def wrap_solver(self, solver_fn):
        return self._track_solver(solver_fn, lambda observed: observed)


def _write_cache_metadata(cache_path):
    """Write stable provenance beside Nengo's opaque exact-array artifacts."""
    metadata = {
        "schema_version": cache_defaults.DECODER_CACHE_SCHEMA_VERSION,
        "framework_versions": framework_versions(),
        "max_size_bytes": cache_defaults.DECODER_CACHE_MAX_SIZE_BYTES,
        "storage": "nengo.DecoderCache exact decoder arrays",
    }
    target = cache_path / "cache-info.json"
    encoded = json.dumps(metadata, indent=2, sort_keys=True) + "\n"
    if target.exists() and target.read_text(encoding="utf-8") == encoded:
        return
    temporary = target.with_name(f"{target.name}.{os.getpid()}.tmp")
    temporary.write_text(encoded, encoding="utf-8")
    temporary.replace(target)


def create_decoder_cache(
    mode,
    learning_connections=(),
    learning_connection_metadata=(),
    environ=None,
):
    """Create the configured cache without changing Nengo's global settings."""
    if mode not in cache_defaults.DECODER_CACHE_MODES:
        raise ValueError(f"Unknown decoder cache mode: {mode}")

    path = resolve_decoder_cache_path(environ)
    if mode == "off":
        return TrackedNoDecoderCache(
            learning_connections=learning_connections,
            learning_connection_metadata=learning_connection_metadata,
            reported_path=path,
        )

    try:
        cache = TrackedDecoderCache(
            path,
            mode=mode,
            learning_connections=learning_connections,
            learning_connection_metadata=learning_connection_metadata,
        )
        if mode == "refresh":
            cache.invalidate()
        _write_cache_metadata(path)
        return cache
    except Exception as exc:
        return TrackedNoDecoderCache(
            mode=mode,
            degraded_reason=f"cache initialization failed: {type(exc).__name__}",
            learning_connections=learning_connections,
            learning_connection_metadata=learning_connection_metadata,
            reported_path=path,
        )


def make_backend_build_model(backend, decoder_cache):
    """Create a Nengo builder model using the selected persistent cache."""
    if backend == "nengo_ocl":
        return NengoBuildModel(decoder_cache=decoder_cache, builder=OclBuilder())
    if backend == "nengo":
        return NengoBuildModel(decoder_cache=decoder_cache)
    raise ValueError(f"Unknown simulator backend: {backend}")


def format_decoder_cache_summary(summary):
    """Render a concise cache status suitable for CLI output."""
    if summary["mode"] == "inspection":
        state = "present" if summary["exists"] else "not created"
        return (
            f"Decoder cache: {state}; {summary['size_bytes']} bytes in "
            f"{summary['file_count']} files\nPath: {summary['path']}"
        )
    degraded = (
        " [read-only/degraded]"
        if summary["persistent"] and not summary.get("write_enabled", True)
        else ""
    )
    return (
        f"Decoder cache ({summary['mode']}): "
        f"{summary['reused_solver_calls']} reused, "
        f"{summary['solved_solver_calls']} solved\n"
        f"Path: {summary['path'] or 'disabled'}{degraded}"
    )
