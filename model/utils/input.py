# handling model input

import nengo
import numpy as np
from config import model_parameters as mp
from utils.processing import WordsToSPAVocab

# Input Node for feeding data into the model. The buffer can be updated with new vectors as needed.
class InputModule:
    def __init__(self, dim):
        self.dim = dim
        self.buffer = np.zeros(dim)
        self.is_recall = True

    def node(self):
        return nengo.Node(self._output)

    def _output(self, t):
        return self.buffer

    def set(self, vector):
        self.buffer[:] = vector