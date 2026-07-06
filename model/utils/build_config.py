"""Shared model-build configuration helpers for compile workflow experiments."""

from contextlib import contextmanager

import nengo
import numpy as np


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
    """Apply temporary Nengo build settings for one model-construction pass."""
    settings = profile["settings"]
    ensemble_n_eval_points = settings.get("ensemble_n_eval_points")

    if ensemble_n_eval_points is None:
        yield
        return

    with nengo.Config(nengo.Ensemble) as config:
        config[nengo.Ensemble].n_eval_points = ensemble_n_eval_points
        yield


def _random_function_initializer(dimensions, init_seed):
    """Return the legacy random-function initializer, optionally seeded."""
    if init_seed is None:
        return lambda x: np.random.random(dimensions)

    rng = np.random.RandomState(init_seed)
    return lambda x: rng.random_sample(dimensions)


def _seeded_decoder_values(pre_obj, dimensions, init_seed):
    """Build deterministic decoder values for seeded NoSolver experiments."""
    if init_seed is None:
        raise ValueError("seeded-nosolver requires an explicit learned-init seed")

    rng = np.random.RandomState(init_seed)
    scale = 1.0 / np.sqrt(max(getattr(pre_obj, "n_neurons", 1), 1))
    return rng.standard_normal((pre_obj.n_neurons, dimensions)) * scale


def make_learned_connection(
    pre_obj,
    post_obj,
    *,
    dimensions,
    learning_rate,
    init_mode="random-function",
    init_seed=None,
):
    """Create one PES connection using the requested initialization strategy.

    `random-function` preserves the original decoded-connection behavior.
    `zero-nosolver` and `seeded-nosolver` keep the same learning rule but swap
    in explicit decoder initialization so compile experiments can separate
    solver cost from architecture cost.
    """
    if init_mode == "random-function":
        return nengo.Connection(
            pre_obj,
            post_obj,
            function=_random_function_initializer(dimensions, init_seed),
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
        raise ValueError(f"Unknown learned init mode: {init_mode}")

    return nengo.Connection(
        pre_obj,
        post_obj,
        solver=solver,
        learning_rule_type=nengo.PES(learning_rate),
    )
