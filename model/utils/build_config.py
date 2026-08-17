"""Shared model-build configuration helpers for compile workflow experiments."""

import hashlib
from contextlib import contextmanager

import nengo
import numpy as np
from nengo.utils.least_squares_solvers import Cholesky

from config import model_defaults


COMPILE_PROFILE_SETTINGS = {
    "full": {
        "ensemble_n_eval_points": None,
    },
    "fast-solver": {
        # Lower eval-point counts reduce solver setup work without changing the
        # architecture shape, which makes this a useful compile-only profile.
        "ensemble_n_eval_points": 100,
    },
}
DEFAULT_COMPILE_PROFILE_NAME = "full"
LEARNED_INIT_MODES = (
    "random-function",
    "zero-nosolver",
    "seeded-nosolver",
)
DEFAULT_LEARNED_INIT_MODE = "random-function"
DEFAULT_LEARNED_INIT_SEED = model_defaults.LEARNED_INIT_SEED


def available_compile_profiles():
    """Return sorted build-profile names accepted by the profile resolver."""
    return tuple(sorted(COMPILE_PROFILE_SETTINGS))


def validate_learned_init_configuration(
    init_mode,
    init_seed,
    *,
    reserve_component_offsets=True,
):
    """Validate a base or already-derived learned-connection seed."""
    if init_mode not in LEARNED_INIT_MODES:
        raise ValueError(f"Unknown learned init mode: {init_mode}")
    if init_mode == "seeded-nosolver" and init_seed is None:
        raise ValueError("seeded-nosolver requires an explicit learned-init seed")
    reserved_offset = (
        model_defaults.LEARNED_INIT_MAX_SEED_OFFSET
        if reserve_component_offsets
        else 0
    )
    max_base_seed = np.iinfo(np.uint32).max - reserved_offset
    if init_seed is not None and not 0 <= init_seed <= max_base_seed:
        raise ValueError(
            "learned-init seed must leave room for configured component offsets"
        )


def resolve_compile_profile(name):
    """Return the concrete settings behind a named compile profile."""
    if name not in COMPILE_PROFILE_SETTINGS:
        raise ValueError(f"Unknown compile profile: {name}")

    return {
        "name": name,
        "settings": dict(COMPILE_PROFILE_SETTINGS[name]),
    }


@contextmanager
def compile_profile_scope(profile):
    """Apply deterministic Nengo defaults and one compile profile per build."""
    settings = profile["settings"]
    ensemble_n_eval_points = settings.get("ensemble_n_eval_points")

    # Nengo 3.2 shares module-level FrozenObject defaults whose fingerprints
    # change after their lazy repr cache is populated. Fresh, canonical objects
    # retain the exact default numerical behavior and make cache keys stable.
    neuron_type = nengo.LIF()
    repr(neuron_type)
    solver = _stable_default_solver()
    with nengo.Config(nengo.Ensemble, nengo.Connection) as config:
        config[nengo.Ensemble].neuron_type = neuron_type
        config[nengo.Connection].solver = solver
        if ensemble_n_eval_points is not None:
            config[nengo.Ensemble].n_eval_points = ensemble_n_eval_points
        yield


def _random_function_initializer(dimensions, init_seed):
    """Return a random target function that is stable for each input vector.

    Deriving a local seed from the configured seed and input avoids mutable RNG
    state. Rebuilding the same Nengo network therefore produces identical
    targets and the same authoritative Nengo decoder-cache key.
    """
    if init_seed is None:
        return lambda x: np.random.random(dimensions)

    seed_bytes = int(init_seed).to_bytes(4, byteorder="little", signed=False)

    def initialize(x):
        digest = hashlib.sha256(seed_bytes)
        digest.update(np.asarray(x, dtype=np.float64).tobytes(order="C"))
        local_seed = int.from_bytes(digest.digest()[:4], byteorder="little")
        return np.random.RandomState(local_seed).random_sample(dimensions)

    return initialize


def _seeded_decoder_values(pre_obj, dimensions, init_seed):
    """Build deterministic decoder values for seeded NoSolver experiments."""
    if init_seed is None:
        raise ValueError("seeded-nosolver requires an explicit learned-init seed")

    rng = np.random.RandomState(init_seed)
    scale = 1.0 / np.sqrt(max(getattr(pre_obj, "n_neurons", 1), 1))
    return rng.standard_normal((pre_obj.n_neurons, dimensions)) * scale


def _stable_default_solver():
    """Return Nengo's standard solver with a stable Nengo 3.2 fingerprint.

    Nengo's module-level default ``LstsqL2`` instance is shared, and its
    fingerprint changes after ``repr`` lazily caches constructor arguments.
    Giving each learned connection a fresh, initialized instance preserves the
    same numerical solver while keeping disk-cache keys stable in long-lived
    benchmark processes.
    """
    least_squares_solver = Cholesky()
    repr(least_squares_solver)
    solver = nengo.solvers.LstsqL2(solver=least_squares_solver)
    repr(solver)
    return solver


def make_learned_connection(
    pre_obj,
    post_obj,
    *,
    dimensions,
    learning_rate,
    init_mode=DEFAULT_LEARNED_INIT_MODE,
    init_seed=DEFAULT_LEARNED_INIT_SEED,
):
    """Create one PES connection using the requested initialization strategy.

    `random-function` preserves decoded-function initialization while using a
    deterministic default seed so Nengo can reuse its exact decoder artifact.
    `zero-nosolver` and `seeded-nosolver` keep the same learning rule but swap
    in explicit decoder initialization so compile experiments can separate
    solver cost from architecture cost.
    """
    validate_learned_init_configuration(
        init_mode,
        init_seed,
        reserve_component_offsets=False,
    )

    if init_mode == "random-function":
        return nengo.Connection(
            pre_obj,
            post_obj,
            function=_random_function_initializer(dimensions, init_seed),
            solver=_stable_default_solver(),
            learning_rule_type=nengo.PES(learning_rate),
        )

    if init_mode == "zero-nosolver":
        solver = nengo.solvers.NoSolver(values=None, weights=False)
    elif init_mode == "seeded-nosolver":
        solver = nengo.solvers.NoSolver(
            values=_seeded_decoder_values(pre_obj, dimensions, init_seed),
            weights=False,
        )
    else:
        raise AssertionError(f"Unhandled learned init mode: {init_mode}")

    return nengo.Connection(
        pre_obj,
        post_obj,
        solver=solver,
        learning_rule_type=nengo.PES(learning_rate),
    )
