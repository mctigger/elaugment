from abc import ABC, abstractmethod


class Transformation(ABC):
    """
    Base class for all transformations.
    Child classes only have to implement transform() which transformes the input in a deterministic way.
    """
    @abstractmethod
    def transform(self, x):
        """
        Transforms the input in a deterministic way.

        :param x: Some input to transform.
        :type x: object

        :return object: The transformed input.
        """
        raise NotImplementedError

    def __call__(self, *args, **kwargs):
        return self.transform(*args, **kwargs)


class Identity(Transformation):
    """
    Returns the input unaltered.
    Useful placeholder transformation.
    """
    def transform(self, x):
        """

        :param x: Some input to transform.
        :type x: object
        :return: The unaltered input.
        """
        return x


class Pipeline(Transformation):
    """
    Runs given transforms in sequence.
    This transformation simply runs all the given transformations in the order given by transforms.
    """
    def __init__(self, transforms):
        """
        :param transforms: List of transformations.
        :type transforms: list of :class:`Transformation`
        """
        self.transforms = transforms

        for t in transforms:
            assert isinstance(t, Transformation)

    def transform(self, x):
        for t in self.transforms:
            x = t(x)

        return x


class Lambda(Transformation):
    def __init__(self, fn, transformations=None):
        self.fn = fn
        self.transformations = transformations

    def transform(self, x):
        return self.fn(self.transformations, x)
