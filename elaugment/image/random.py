import numpy as np

from elaugment.transformations import Pipeline
from elaugment.random import RandomBinary, Random
from elaugment.image import transformations


class RandomFlipLr(RandomBinary):
    """
    Randomly apply left-right flips.
    """
    def __init__(self, p=0.5):
        super(RandomFlipLr, self).__init__(transformations.FlipLr(), p)


class RandomFlipUd(RandomBinary):
    """
    Randomly apply up-down flips.
    """
    def __init__(self, p=0.5):
        super(RandomFlipUd, self).__init__(transformations.FlipUd(), p)


class RandomAffine(Random):
    """
    Random affine transformation using :class:`skimage.transform.AffineTransform`.
    """
    def __init__(
            self,
            image_size,
            rotation=0,
            translation=(0, 0),
            scale=(1, 1),
            shear=0,
            **kwargs
    ):
        """
        Takes functions of form: random_state -> parameters.

        :param image_size: Function returning the size of the input image.
        :param scale: Function returning the scale horizontal and vertical (sx, sy).
        :param rotation: Function returning the rotation angle in counter-clockwise direction as radians.
        :param shear: Function returning the shear angle in counter-clockwise direction as radians.
        :param translation: Function returning the translate horizontal and vertical (tx, ty).
        :type image_size: rs -> int
        :type scale: rs -> (int, int)
        :type rotation: rs -> int
        :type shear: rs -> int
        :type translation: rs -> (int, int)
        """
        self.image_size = image_size
        self.rotation = rotation
        self.translation = translation
        self.scale = scale
        self.shear = shear
        self.kwargs = kwargs

    def draw(self, rs):
        rotation = self.rotation(rs) if callable(self.rotation) else self.rotation
        translation = self.translation(rs) if callable(self.translation) else self.translation
        scale = self.scale(rs) if callable(self.scale) else self.scale
        shear = self.shear(rs) if callable(self.shear) else self.shear

        return transformations.Affine(
            self.image_size,
            rotation=rotation,
            translation=translation,
            scale=scale,
            shear=shear,
            **self.kwargs
        )


class RandomCrop(Random):
    """Randomly crops a quadratic patch of a given size from the input.

    """
    def __init__(self, crop_size):
        """
        :param crop_size: Edge size of the quadratic crop.
        :type crop_size: int
        """
        self.crop_size = crop_size

    def draw(self, rs):
        vertical = rs.rand()
        horizontal = rs.rand()
        
        return transformations.CropByFloat(self.crop_size, self.crop_size, vertical, horizontal)


class RandomResizedCrop(Random):
    """
    Randomly sized and scaled crop known from the inception networks training.
    """
    def __init__(self, crop_size, min_area=0.08, mode='reflect'):
        self.crop_size = crop_size
        self.min_area = min_area
        self.mode = mode

    def draw(self, rs):
        for attempt in range(10):
            target_area = rs.uniform(self.min_area, 1)
            aspect_ratio = rs.uniform(3. / 4, 4. / 3)

            w = np.sqrt(target_area * aspect_ratio)
            h = np.sqrt(target_area / aspect_ratio)

            if rs.random_sample() < 0.5:
                w, h = h, w

            if w <= 1 and h <= 1:
                i = rs.uniform(0, 1 - h)
                j = rs.uniform(0, 1 - w)

                transform = Pipeline([
                    transformations.CropRelaugmenttive(w, h, i, j),
                    transformations.Resize((self.crop_size, self.crop_size), mode=self.mode)
                ])

                return transform

        return RandomCrop(self.crop_size).draw(rs)


class RandomColorAdjustment(Random):
    """
    Randomly adjusts hue, saturation and brightness
    """
    def __init__(
            self,
            hue_shift_limit=(0.0, 1.0),
            sat_shift_limit=(0.0, 1.0),
            val_shift_limit=(0.0, 1.0)
    ):
        """
        Color adjustments are uniformly drawn from (lower_bound, upper_bound)

        :param hue_shift_limit: Interval to draw hue shift from. Must be between 0.0 and 1.0.
        :param sat_shift_limit: Interval to draw saturation shift from. Must be between 0.0 and 1.0.
        :param val_shift_limit: Interval to draw brightness shift from. Must be between 0.0 and 1.0.
        """
        self.hue_shift_limit = hue_shift_limit
        self.sat_shift_limit = sat_shift_limit
        self.val_shift_limit = val_shift_limit

    def draw(self, rs):
        hue_shift = rs.uniform(self.hue_shift_limit[0], self.hue_shift_limit[1])
        sat_shift = rs.uniform(self.sat_shift_limit[0], self.sat_shift_limit[1])
        val_shift = rs.uniform(self.val_shift_limit[0], self.val_shift_limit[1])

        return transformations.ColorAdjustment(hue_shift, sat_shift, val_shift)


class RandomColorPerturbation(Random):
    """
    Randomly changes the image colors based on eigenvectors and eigenvalues

    Implements color perturbation from:
        "ImageNet Classification with Deep Convolutional Neural Networks"
        Alex Krizhevsky and Sutskever, Ilya and Hinton, Geoffrey E,
        http://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf
    """
    def __init__(self, mean=0, std=0.1, num_channels=3, clip=True):
        """
        Values are drawn from a normal distribution.
        :param mean: Mean of the normal distribution.
        :param std: Standard deviation of the normal distribution.
        """
        self.mean = mean
        self.std = std
        self.num_channels = num_channels
        self.clip = clip

    def draw(self, rs):
        alphas = rs.normal(self.mean, self.std, size=self.num_channels)

        return transformations.ColorPerturbation(alphas, self.clip)


class RandomDistortion(Random):
    """
    Random elaugmentstic distortions.
    This implementation is pretty slow as scitkit-image PiecewiseAffine is Used.
    """
    def __init__(
            self,
            tiles_x,
            tiles_y,
            magnitude_x,
            magnitude_y,
            **kwargs
    ):
        """
        :param tiles_x: Number of points in x-direction.
        :param tiles_y: Number of points in y-direction.
        :param magnitude_x: Maximum magnitude relaugmenttive to the x-tile-size.
        :param magnitude_y: Maximum magnitude relaugmenttive to the y-tile-size.
        :type tiles_x: int
        :type tiles_y: int
        :type magnitude_x: float
        :type magnitude_y: float
        """
        self.tiles_x = tiles_x
        self.tiles_y = tiles_y
        self.magnitude_x = magnitude_x / tiles_x
        self.magnitude_y = magnitude_y / tiles_y

        self.kwargs = kwargs

    def draw(self, rs):
        src_cols = np.linspace(0, 1, self.tiles_x)
        src_rows = np.linspace(0, 1, self.tiles_y)
        src_rows, src_cols = np.meshgrid(src_rows, src_cols)
        src = np.dstack([src_cols.flat, src_rows.flat])[0]

        dst_rows = src[:, 1]
        dst_cols = src[:, 0]
        dst = np.vstack([dst_cols, dst_rows]).T

        dst[:, 0] += rs.uniform(-self.magnitude_x / 2, self.magnitude_x / 2, size=dst[:, 0].shape)
        dst[:, 1] += rs.uniform(-self.magnitude_y / 2, self.magnitude_y / 2, size=dst[:, 1].shape)

        src = src.reshape(self.tiles_x, self.tiles_y, 2)
        dst = dst.reshape(self.tiles_x, self.tiles_y, 2)

        dst[0, :] = src[0, :]
        dst[-1, :] = src[-1, :]
        dst[:, 0] = src[:, 0]
        dst[:, -1] = src[:, -1]

        src = src.reshape(-1, 2)
        dst = dst.reshape(-1, 2)

        return transformations.PiecewiseAffine(
            src,
            dst,
            **self.kwargs
        )


class RandomPadding(Random):
    def __init__(self, pad_x, pad_y):
        self.pad_x = pad_x
        self.pad_y = pad_y

    def draw(self, rs):
        x = rs.randint(0, self.pad_x)
        y = rs.randint(0, self.pad_y)

        return transformations.Padding(((self.pad_x - x, x), (self.pad_y - y, y), (0, 0)))