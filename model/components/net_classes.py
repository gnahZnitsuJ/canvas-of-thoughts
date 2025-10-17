import nengo_spa as spa

class BaseComponent(spa.Network):
    def __init__(self, model_vocab):
        self.prediction = spa.state(model_vocab, subdimensions=model_vocab.dimensions, represent_cc_identity=False)