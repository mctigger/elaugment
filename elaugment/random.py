from abc import ABC, abstractmethod

import numpy as np

from . import transformations


class Random(ABC):
    """Base class for all random transformations.

    This class let's the generator draw random parameters for the a given transformation.
    This enables reproducibility of random transformations,
    """
    @abstractmethod
    def draw(self, rs):
        """

        Args:
            rs: A numpy random state. This method is usually only called by the generator.

        Returns:
            transformations.Transformation: A deterministic transformation parameterized with randomly drawn parameters.
        """
        pass


class RandomPipeline(Random):
    """Creates a deterministic pipeline

    Transforms given RandomTransformations to deterministic transformations with a draw call and returns a Pipeline
    """
    def __init__(self, transforms):
        """
        Args:
            transforms: list of transformations.Transformation or RandomTransformation
        """
        self.transforms = transforms

    def draw(self, rs):
        applied_transforms = []
        for t in self.transforms:
            if isinstance(t, Random):
                applied_transforms.append(t.draw(rs))
            else:
                applied_transforms.append(t)

        return transformations.Pipeline(applied_transforms)


class RandomLambda(Random):
    def __init__(self, fn, transforms):
        self.fn = fn
        self.transforms = transforms

    def draw(self, rs):
        if isinstance(self.transforms, Random):
            transforms = self.transforms.draw(rs)
        else:
            transforms = self.transforms

        return transformations.Lambda(self.fn, transforms)


class RandomBinary(Random):
    """Wraps RandomTransformations and only applies them with probability p

    """
    def __init__(self, transform, p=0.5):
        """
        Args:
            transform (obj:RandomTransformation): A random transformation
            p (float): Probability between 0 and 1 that the transformation is applied.
        """
        self.t = transform
        self.p = p

    def draw(self, rs):
        if rs.random_sample() > self.p:
            return RandomPipeline([self.t]).draw(rs)

        return transformations.Identity()

