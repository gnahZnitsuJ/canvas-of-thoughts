"""Independent contract tests for buffered and scheduled model input."""

import sys
import unittest
from pathlib import Path

import numpy as np


MODEL_DIR = Path(__file__).resolve().parents[1] / "model"
if str(MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(MODEL_DIR))

from utils.input import InputModule  # noqa: E402


class InputModuleContractTests(unittest.TestCase):
    def test_schedule_uses_exact_half_open_windows_and_zero_elsewhere(self):
        module = InputModule(dim=2)
        first = np.array([1.0, 10.0])
        second = np.array([2.0, 20.0])
        module.set_schedule([first, second], start_time=1.0, token_duration=0.25)

        observations = {
            "before": module._output(np.nextafter(1.0, -np.inf)),
            "first_start": module._output(1.0),
            "first_end_inside": module._output(np.nextafter(1.25, -np.inf)),
            "second_start": module._output(1.25),
            "second_end_inside": module._output(np.nextafter(1.5, -np.inf)),
            "schedule_end": module._output(1.5),
        }

        np.testing.assert_array_equal(observations["before"], [0.0, 0.0])
        np.testing.assert_array_equal(observations["first_start"], first)
        np.testing.assert_array_equal(observations["first_end_inside"], first)
        np.testing.assert_array_equal(observations["second_start"], second)
        np.testing.assert_array_equal(observations["second_end_inside"], second)
        np.testing.assert_array_equal(observations["schedule_end"], [0.0, 0.0])

    def test_schedule_validates_duration_and_vector_shape(self):
        module = InputModule(dim=3)

        for invalid_duration in (0.0, -0.1):
            with self.subTest(token_duration=invalid_duration):
                with self.assertRaises(ValueError):
                    module.set_schedule(
                        [np.zeros(3)],
                        start_time=0.0,
                        token_duration=invalid_duration,
                    )

        for invalid_vector in ([1.0, 2.0], [[1.0, 2.0, 3.0]]):
            with self.subTest(vector=invalid_vector):
                with self.assertRaises(ValueError):
                    module.set_schedule(
                        [invalid_vector],
                        start_time=0.0,
                        token_duration=0.1,
                    )

    def test_schedule_copies_source_vectors_and_clear_restores_buffer(self):
        module = InputModule(dim=2)
        module.set([7.0, 8.0])
        source = np.array([3.0, 4.0])
        module.set_schedule([source], start_time=2.0, token_duration=0.5)

        source[:] = [-30.0, -40.0]
        np.testing.assert_array_equal(module._output(2.25), [3.0, 4.0])

        module.clear_schedule()
        np.testing.assert_array_equal(module._output(2.25), [7.0, 8.0])
        self.assertIsNone(module.schedule_vectors)
        self.assertIsNone(module.schedule_start_time)
        self.assertIsNone(module.schedule_token_duration)


if __name__ == "__main__":
    unittest.main()
