import numpy as np

class PriorityGridController:
    def __init__(self, weights=None):
        self.weights = weights if weights is not None else self.weights = np.random()