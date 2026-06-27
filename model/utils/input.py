# handling model input

import nengo
import numpy as np


class InputModule:
    """Feed either one buffered vector or a time-based vector schedule.

    Interactive prediction and generation keep using the simple buffer path.
    Scheduled training swaps in a token schedule so one long simulator run can
    present an entire training sequence without per-token `sim.run(...)` calls.
    """

    def __init__(self, dim):
        self.dim = dim
        self.buffer = np.zeros(dim)
        self.is_recall = True
        self.schedule_vectors = None
        self.schedule_start_time = None
        self.schedule_token_duration = None

    def node(self):
        """Return the Nengo node exposing this module's current output."""
        return nengo.Node(self._output, size_out=self.dim)

    def _output(self, t):
        """Emit either the buffered vector or the current scheduled token."""
        if self.schedule_vectors is not None:
            elapsed = t - self.schedule_start_time
            index = int(
                np.floor((elapsed + 1e-12) / self.schedule_token_duration)
            )

            if 0 <= index < len(self.schedule_vectors):
                return self.schedule_vectors[index]

            return np.zeros(self.dim)

        return self.buffer

    def set(self, vector):
        """Update the buffer used by stepwise training and interactive calls."""
        self.buffer[:] = np.asarray(vector, dtype=float)

    def set_schedule(self, vectors, start_time, token_duration):
        """Install a schedule of vectors presented in fixed-duration windows."""
        if token_duration <= 0:
            raise ValueError("token_duration must be positive")

        scheduled_vectors = []
        for vector in vectors:
            array = np.asarray(vector, dtype=float)
            if array.shape != (self.dim,):
                raise ValueError(
                    "Scheduled vectors must match the module dimension: "
                    f"expected {(self.dim,)}, found {array.shape}"
                )
            scheduled_vectors.append(array.copy())

        self.schedule_vectors = scheduled_vectors
        self.schedule_start_time = float(start_time)
        self.schedule_token_duration = float(token_duration)

    def clear_schedule(self):
        """Return the module to normal buffer-driven output."""
        self.schedule_vectors = None
        self.schedule_start_time = None
        self.schedule_token_duration = None


# creates a unitary vector for encoding position later
def make_unitary(dim, rng=np.random):
    v = rng.randn(dim)
    fft = np.fft.fft(v)
    fft /= np.abs(fft)
    return np.fft.ifft(fft).real