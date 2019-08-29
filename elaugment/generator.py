import numpy as np
from . import random


class TransformationsGenerator:
    def __init__(self, transforms, seed=0):
        self.transforms = transforms
        self.rs = np.random.RandomState(seed)

    def __iter__(self):
        return self

    def __next__(self):
        return self.next()

    def next(self):
        if isinstance(self.transforms, list):
            return random.RandomPipeline(self.transforms).draw(self.rs)

        else:
            return random.RandomPipeline([self.transforms]).draw(self.rs)
