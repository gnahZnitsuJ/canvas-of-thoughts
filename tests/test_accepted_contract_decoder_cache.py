"""Independent acceptance tests for persistent decoder-cache behavior."""

import contextlib
import io
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import nengo
import nengo_spa as spa
import numpy as np


MODEL_DIR = Path(__file__).resolve().parents[1] / "model"
if str(MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(MODEL_DIR))

from app.args import parse_args  # noqa: E402
import components.net_comp as net_comp  # noqa: E402
from config import cache_defaults, model_defaults  # noqa: E402
from utils.build_config import (  # noqa: E402
    compile_profile_scope,
    make_learned_connection,
    validate_learned_init_configuration,
)
from utils.decoder_cache import (  # noqa: E402
    TrackedDecoderCache,
    TrackedNoDecoderCache,
    create_decoder_cache,
    format_decoder_cache_summary,
    inspect_decoder_cache,
    make_backend_build_model,
    resolve_decoder_cache_path,
)


class CountingSolver:
    """Small deterministic stand-in for an expensive decoder solve."""

    def __init__(self):
        self.calls = 0

    def __call__(self, conn, gain, bias, x, targets, rng=np.random, **kwargs):
        self.calls += 1
        shape = (x.shape[1], targets.shape[1])
        return np.full(shape, self.calls, dtype=float), {"calls": self.calls}


def make_connection(label="learned-decoder"):
    return nengo.Connection(
        nengo.Ensemble(8, 2, add_to_container=False),
        nengo.Node(size_in=2, add_to_container=False),
        solver=nengo.solvers.LstsqL2(reg=0.1),
        label=label,
        add_to_container=False,
    )


def solver_arguments(connection, *, target_value=1.0, seed=17):
    return {
        "conn": connection,
        "gain": np.ones(8),
        "bias": np.zeros(8),
        "x": np.arange(24, dtype=float).reshape(12, 2),
        "targets": np.full((12, 2), target_value),
        "rng": np.random.RandomState(seed),
    }


class DecoderCacheConfigurationContractTests(unittest.TestCase):
    def parse(self, *arguments):
        with patch.object(sys, "argv", ["model/main.py", *arguments]):
            return parse_args()

    def test_committed_defaults_are_deterministic_and_modes_are_stable(self):
        self.assertEqual(model_defaults.LEARNED_INIT_SEED, model_defaults.MODEL_SEED)
        self.assertEqual(
            cache_defaults.DECODER_CACHE_MODES,
            ("auto", "refresh", "off"),
        )
        self.assertEqual(cache_defaults.DEFAULT_DECODER_CACHE_MODE, "auto")
        self.assertGreater(cache_defaults.DECODER_CACHE_MAX_SIZE_BYTES, 0)
        self.assertIsInstance(cache_defaults.DECODER_CACHE_ENV_VAR, str)
        self.assertTrue(cache_defaults.DECODER_CACHE_ENV_VAR)

    def test_cli_defaults_and_all_documented_cache_modes(self):
        defaults = self.parse()
        self.assertEqual(defaults.learned_init_seed, model_defaults.MODEL_SEED)
        self.assertEqual(defaults.decoder_cache_mode, "auto")
        self.assertFalse(defaults.inspect_decoder_cache)

        for mode in cache_defaults.DECODER_CACHE_MODES:
            with self.subTest(mode=mode):
                parsed = self.parse(
                    "--decoder-cache-mode",
                    mode,
                    "--inspect-decoder-cache",
                )
                self.assertEqual(parsed.decoder_cache_mode, mode)
                self.assertTrue(parsed.inspect_decoder_cache)

    def test_cli_rejects_an_unknown_cache_mode(self):
        with self.assertRaises(SystemExit) as raised:
            with contextlib.redirect_stderr(io.StringIO()):
                self.parse("--decoder-cache-mode", "sometimes")
        self.assertEqual(raised.exception.code, 2)

    def test_environment_override_is_version_scoped(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            override = Path(temporary_dir) / "shared-cache-root"
            environ = {cache_defaults.DECODER_CACHE_ENV_VAR: str(override)}

            resolved = resolve_decoder_cache_path(environ)

            self.assertTrue(resolved.is_relative_to(override))
            self.assertNotEqual(resolved, override)
            self.assertIn(f"nengo-{nengo.__version__}", resolved.name)
            self.assertFalse(resolved.exists())

    def test_relative_environment_override_is_rejected(self):
        environ = {
            cache_defaults.DECODER_CACHE_ENV_VAR: "relative/cache-directory"
        }
        with self.assertRaisesRegex(ValueError, "absolute path"):
            resolve_decoder_cache_path(environ)

    def test_inspection_of_missing_cache_does_not_create_it(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            override = Path(temporary_dir) / "not-created"
            environ = {cache_defaults.DECODER_CACHE_ENV_VAR: str(override)}

            summary = inspect_decoder_cache(environ)

            self.assertFalse(override.exists())
            self.assertFalse(summary["exists"])
            self.assertEqual(summary["size_bytes"], 0)
            self.assertEqual(summary["file_count"], 0)
            self.assertEqual(summary["path"], str(resolve_decoder_cache_path(environ)))
            rendered = format_decoder_cache_summary(summary)
            self.assertIn("not created", rendered)
            self.assertIn(summary["path"], rendered)


class LearnedInitializationContractTests(unittest.TestCase):
    def make_random_function_connection(self, seed):
        with nengo.Network():
            return make_learned_connection(
                nengo.Ensemble(8, 2),
                nengo.Node(size_in=6),
                dimensions=6,
                learning_rate=0.001,
                init_mode="random-function",
                init_seed=seed,
            )

    def test_seeded_random_function_is_referentially_deterministic(self):
        input_vector = np.array([0.25, -0.75])
        same_seed_first = self.make_random_function_connection(seed=1234)
        same_seed_second = self.make_random_function_connection(seed=1234)
        different_seed = self.make_random_function_connection(seed=1235)

        expected = same_seed_first.function(input_vector)

        np.testing.assert_array_equal(
            same_seed_first.function(input_vector.copy()),
            expected,
        )
        np.testing.assert_array_equal(
            same_seed_second.function(input_vector),
            expected,
        )
        self.assertFalse(
            np.array_equal(
                same_seed_first.function(np.array([0.25, -0.5])),
                expected,
            )
        )
        self.assertFalse(
            np.array_equal(different_seed.function(input_vector), expected)
        )

    def test_root_context_metadata_has_stable_ids_and_effective_seeds(self):
        max_effective_seed = np.iinfo(np.uint32).max
        base_seed = (
            max_effective_seed - model_defaults.LEARNED_INIT_MAX_SEED_OFFSET
        )
        validate_learned_init_configuration("random-function", base_seed)
        vocabulary = spa.Vocabulary(
            model_defaults.VOCAB_DIMENSIONS,
            strict=False,
        )
        vocabulary.populate("POS")

        assembled = net_comp.Model(
            sub_lengths=[1, model_defaults.CONTEXT_LENGTH],
            model_vocab=vocabulary,
            probe_mode="minimal",
            learned_init_mode="random-function",
            learned_init_seed=base_seed,
            compile_profile_name="fast-solver",
            compile_profile_settings={"ensemble_n_eval_points": 100},
            architecture_name="root-context-v1",
        )

        metadata_by_component = {
            item["component"]: item
            for item in assembled.learning_connection_metadata
        }
        self.assertEqual(set(metadata_by_component), {"predictor", "refiner"})
        self.assertEqual(
            metadata_by_component["refiner"]["stable_id"],
            "refiner.learning_connection_0",
        )
        self.assertEqual(
            metadata_by_component["predictor"]["stable_id"],
            "predictor.learning_connection_0",
        )
        self.assertEqual(
            metadata_by_component["refiner"]["effective_seed"],
            base_seed,
        )
        self.assertEqual(
            metadata_by_component["predictor"]["effective_seed"],
            base_seed + 1,
        )
        self.assertEqual(
            metadata_by_component["predictor"]["effective_seed"],
            max_effective_seed,
        )
        self.assertNotEqual(
            metadata_by_component["refiner"]["effective_seed"],
            metadata_by_component["predictor"]["effective_seed"],
        )
        with self.assertRaises(ValueError):
            validate_learned_init_configuration(
                "random-function",
                base_seed + 1,
            )


class DecoderCacheSolverContractTests(unittest.TestCase):
    def make_small_canvas_graph(self):
        profile = {
            "name": "cache-regression",
            "settings": {"ensemble_n_eval_points": 20},
        }
        connection_specs = (
            ("refiner.learning_connection_0", 700),
            ("predictor.learning_connection_0", 701),
        )
        with compile_profile_scope(profile):
            with nengo.Network(seed=91) as network:
                learning_connections = []
                metadata = []
                for stable_id, effective_seed in connection_specs:
                    pre = nengo.Ensemble(12, 2)
                    post = nengo.Ensemble(12, 2)
                    learning_connections.append(
                        make_learned_connection(
                            pre,
                            post,
                            dimensions=2,
                            learning_rate=0.001,
                            init_mode="random-function",
                            init_seed=effective_seed,
                        )
                    )
                    metadata.append(
                        {
                            "stable_id": stable_id,
                            "component": stable_id.split(".", maxsplit=1)[0],
                            "effective_seed": effective_seed,
                        }
                    )
        return network, learning_connections, metadata

    def test_fresh_native_build_reuses_both_exact_learned_decoder_keys(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            environ = {cache_defaults.DECODER_CACHE_ENV_VAR: temporary_dir}

            first_network, first_connections, first_metadata = (
                self.make_small_canvas_graph()
            )
            first_cache = create_decoder_cache(
                "auto",
                learning_connections=first_connections,
                learning_connection_metadata=first_metadata,
                environ=environ,
            )
            with nengo.Simulator(
                first_network,
                model=make_backend_build_model("nengo", first_cache),
                progress_bar=False,
                optimize=False,
            ) as first_simulator:
                first_weights = {
                    metadata["stable_id"]: np.array(
                        first_simulator.data[connection].weights,
                        copy=True,
                    )
                    for connection, metadata in zip(
                        first_connections,
                        first_metadata,
                    )
                }

            second_network, second_connections, second_metadata = (
                self.make_small_canvas_graph()
            )
            second_cache = create_decoder_cache(
                "auto",
                learning_connections=second_connections,
                learning_connection_metadata=second_metadata,
                environ=environ,
            )
            with nengo.Simulator(
                second_network,
                model=make_backend_build_model("nengo", second_cache),
                progress_bar=False,
                optimize=False,
            ) as second_simulator:
                second_weights = {
                    metadata["stable_id"]: np.array(
                        second_simulator.data[connection].weights,
                        copy=True,
                    )
                    for connection, metadata in zip(
                        second_connections,
                        second_metadata,
                    )
                }

            first_summary = first_cache.telemetry()
            second_summary = second_cache.telemetry()
            first_keys = {
                event["stable_id"]: event["nengo_cache_key"]
                for event in first_summary["learned_connections"]
            }
            second_keys = {
                event["stable_id"]: event["nengo_cache_key"]
                for event in second_summary["learned_connections"]
            }

            self.assertEqual(first_summary["solved_solver_calls"], 2)
            self.assertEqual(first_summary["reused_solver_calls"], 0)
            self.assertEqual(second_summary["solved_solver_calls"], 0)
            self.assertEqual(second_summary["reused_solver_calls"], 2)
            self.assertEqual(first_keys, second_keys)
            self.assertEqual(
                set(second_keys),
                {
                    "refiner.learning_connection_0",
                    "predictor.learning_connection_0",
                },
            )
            self.assertTrue(all(second_keys.values()))
            self.assertEqual(set(first_weights), set(second_weights))
            for stable_id in first_weights:
                np.testing.assert_array_equal(
                    second_weights[stable_id],
                    first_weights[stable_id],
                )
            self.assertEqual(
                {
                    event["outcome"]
                    for event in second_summary["learned_connections"]
                },
                {"reused"},
            )

    def test_successful_solve_survives_cache_write_failure(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            connection = make_connection()
            solver = CountingSolver()
            cache = TrackedDecoderCache(
                temporary_dir,
                learning_connections=(connection,),
            )

            def fail_after_success(observed_solver):
                def wrapped(*args, **kwargs):
                    observed_solver(*args, **kwargs)
                    raise OSError("simulated cache write failure")

                return wrapped

            wrapped_solver = cache._track_solver(solver, fail_after_success)
            decoders, info = wrapped_solver(**solver_arguments(connection))

            np.testing.assert_array_equal(decoders, np.ones((2, 2)))
            self.assertEqual(info["calls"], 1)
            self.assertTrue(cache.integrity_is_valid((decoders, info)))
            telemetry = cache.telemetry()
            self.assertEqual(telemetry["solved_solver_calls"], 1)
            self.assertEqual(telemetry["failed_solver_calls"], 0)
            self.assertEqual(telemetry["cache_write_failures"], 1)
            self.assertIn("write failed", telemetry["degraded_reason"])
            self.assertEqual(
                telemetry["learned_connections"][0]["outcome"],
                "solved-write-failed",
            )

    def test_stale_integrity_rejects_parseable_decoder_and_recomputes(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            connection = make_connection()
            solver = CountingSolver()
            cache = TrackedDecoderCache(
                temporary_dir,
                learning_connections=(connection,),
            )
            cached_result = cache.attach_integrity(
                (np.zeros((2, 2)), {"origin": "cached"})
            )
            mutated_decoders = cached_result[0].copy()
            mutated_decoders[0, 0] = 99.0
            stale_result = (mutated_decoders, cached_result[1])
            self.assertFalse(cache.integrity_is_valid(stale_result))

            def return_stale_cached_result(observed_solver):
                def wrapped(*args, **kwargs):
                    return stale_result

                return wrapped

            wrapped_solver = cache._track_solver(
                solver,
                return_stale_cached_result,
            )
            recomputed = wrapped_solver(**solver_arguments(connection))

            self.assertEqual(solver.calls, 1)
            np.testing.assert_array_equal(recomputed[0], np.ones((2, 2)))
            self.assertTrue(cache.integrity_is_valid(recomputed))
            telemetry = cache.telemetry()
            self.assertEqual(telemetry["corrupt_cache_reads"], 1)
            self.assertEqual(telemetry["solved_solver_calls"], 1)
            self.assertEqual(telemetry["failed_solver_calls"], 0)
            self.assertEqual(
                telemetry["learned_connections"][0]["outcome"],
                "unverified-recomputed",
            )

    def test_auto_cache_persists_across_independent_instances(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            environ = {cache_defaults.DECODER_CACHE_ENV_VAR: temporary_dir}
            connection = make_connection()
            arguments = solver_arguments(connection)

            first_solver = CountingSolver()
            first_cache = create_decoder_cache(
                "auto",
                learning_connections=(connection,),
                environ=environ,
            )
            self.assertIsInstance(first_cache, TrackedDecoderCache)
            with first_cache:
                first_result = first_cache.wrap_solver(first_solver)(**arguments)
            self.assertEqual(first_solver.calls, 1)
            first_telemetry = first_cache.telemetry()
            self.assertEqual(first_telemetry["solved_solver_calls"], 1)
            self.assertEqual(first_telemetry["reused_solver_calls"], 0)
            self.assertEqual(
                first_telemetry["learned_connections"][0]["outcome"],
                "solved",
            )

            second_solver = CountingSolver()
            second_cache = create_decoder_cache(
                "auto",
                learning_connections=(connection,),
                environ=environ,
            )
            with second_cache:
                second_result = second_cache.wrap_solver(second_solver)(**arguments)

            self.assertEqual(second_solver.calls, 0)
            np.testing.assert_array_equal(second_result[0], first_result[0])
            self.assertEqual(second_result[1], first_result[1])
            second_telemetry = second_cache.telemetry()
            self.assertEqual(second_telemetry["reused_solver_calls"], 1)
            self.assertEqual(second_telemetry["solved_solver_calls"], 0)
            learned_event = second_telemetry["learned_connections"][0]
            self.assertEqual(learned_event["outcome"], "reused")
            self.assertIn(connection.label, learned_event["connection"])
            self.assertIn("1 reused", format_decoder_cache_summary(second_telemetry))

    def test_changed_targets_and_rng_seed_force_new_solves(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            environ = {cache_defaults.DECODER_CACHE_ENV_VAR: temporary_dir}
            connection = make_connection()
            solver = CountingSolver()
            cache = create_decoder_cache("auto", environ=environ)

            with cache:
                wrapped = cache.wrap_solver(solver)
                wrapped(**solver_arguments(connection, target_value=1.0, seed=17))
                wrapped(**solver_arguments(connection, target_value=1.0, seed=17))
                wrapped(**solver_arguments(connection, target_value=2.0, seed=17))
                wrapped(**solver_arguments(connection, target_value=1.0, seed=18))

            self.assertEqual(solver.calls, 3)
            telemetry = cache.telemetry()
            self.assertEqual(telemetry["reused_solver_calls"], 1)
            self.assertEqual(telemetry["solved_solver_calls"], 3)

    def test_missing_or_corrupt_artifact_falls_back_to_solver(self):
        for damage in ("missing", "corrupt"):
            with self.subTest(damage=damage), tempfile.TemporaryDirectory() as temporary_dir:
                environ = {cache_defaults.DECODER_CACHE_ENV_VAR: temporary_dir}
                connection = make_connection()
                arguments = solver_arguments(connection)
                initial_solver = CountingSolver()
                initial_cache = create_decoder_cache("auto", environ=environ)
                with initial_cache:
                    initial_cache.wrap_solver(initial_solver)(**arguments)
                    artifacts = [
                        Path(path)
                        for path in initial_cache.get_files()
                        if Path(path).suffix == ".nco"
                    ]
                self.assertEqual(len(artifacts), 1)

                if damage == "missing":
                    artifacts[0].unlink()
                else:
                    artifacts[0].write_bytes(b"not a valid Nengo cache entry")

                recovery_solver = CountingSolver()
                recovered_cache = create_decoder_cache("auto", environ=environ)
                with recovered_cache:
                    recovered_cache.wrap_solver(recovery_solver)(**arguments)

                self.assertEqual(recovery_solver.calls, 1)
                self.assertEqual(
                    recovered_cache.telemetry()["solved_solver_calls"],
                    1,
                )

    def test_refresh_invalidates_only_the_version_scoped_cache(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            override = Path(temporary_dir) / "cache-root"
            environ = {cache_defaults.DECODER_CACHE_ENV_VAR: str(override)}
            connection = make_connection()
            arguments = solver_arguments(connection)
            original_solver = CountingSolver()
            original_cache = create_decoder_cache("auto", environ=environ)
            with original_cache:
                original_cache.wrap_solver(original_solver)(**arguments)

            sibling_marker = override / "belongs-to-another-namespace.txt"
            sibling_marker.write_text("preserve", encoding="utf-8")
            refreshed_solver = CountingSolver()
            refreshed_cache = create_decoder_cache("refresh", environ=environ)
            with refreshed_cache:
                refreshed_cache.wrap_solver(refreshed_solver)(**arguments)

            self.assertEqual(refreshed_solver.calls, 1)
            self.assertEqual(refreshed_cache.telemetry()["mode"], "refresh")
            self.assertEqual(sibling_marker.read_text(encoding="utf-8"), "preserve")

    def test_off_mode_solves_every_time_without_writing(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            override = Path(temporary_dir) / "cache-must-not-exist"
            environ = {cache_defaults.DECODER_CACHE_ENV_VAR: str(override)}
            connection = make_connection()
            solver = CountingSolver()
            cache = create_decoder_cache(
                "off",
                learning_connections=(connection,),
                environ=environ,
            )

            self.assertIsInstance(cache, TrackedNoDecoderCache)
            with cache:
                wrapped = cache.wrap_solver(solver)
                wrapped(**solver_arguments(connection))
                wrapped(**solver_arguments(connection))

            self.assertEqual(solver.calls, 2)
            self.assertFalse(override.exists())
            telemetry = cache.telemetry()
            self.assertFalse(telemetry["persistent"])
            self.assertEqual(telemetry["solved_solver_calls"], 2)
            self.assertEqual(telemetry["reused_solver_calls"], 0)
            self.assertEqual(
                [event["outcome"] for event in telemetry["learned_connections"]],
                ["solved", "solved"],
            )

    def test_native_backend_receives_cache_and_unknown_backend_is_rejected(self):
        cache = create_decoder_cache("off", environ={})
        build_model = make_backend_build_model("nengo", cache)
        self.assertIs(build_model.decoder_cache, cache)

        with self.assertRaises(ValueError):
            make_backend_build_model("unsupported", cache)


if __name__ == "__main__":
    unittest.main()
