from math import floor, ceil

import numpy as np
import skimage.transform
import skimage.color
import skimage.exposure
import skimage.util
import scipy.misc

from elaugment.transformations import Transformation


class FlipLr(Transformation):
    """
    Flips the input along the horizontal middle axis.
    """
    def transform(self, x):
        return np.fliplr(x)


class FlipUd(Transformation):
    """
    Flips the input along the vertical middle axis.
    """
    def transform(self, x):
        return np.flipud(x)


class Affine(Transformation):
    """
    Affine transformation using :class:`skimage.transform.AffineTransform`.
    """
    def __init__(
            self,
            image_size,
            rotation=0,
            translation=0,
            scale=(1, 1),
            shear=0,
            **kwargs
    ):
        """
        Image size needs to be known to make rotation around the center possible.

        :param image_size: Size of the input image.
        :param scale: Scale horizontal and vertical (sx, sy).
        :param rotation: Rotation angle in counter-clockwise direction as radians.
        :param shear: Shear angle in counter-clockwise direction as radians.
        :param translation: Translate horizontal and vertical (tx, ty).
        :type image_size: int
        :type scale: (int, int)
        :type rotation: int
        :type shear: int
        :type translation: (int, int)
        """
        center_shift = 0
        if image_size:
            center_shift = image_size / 2

        tf_center = skimage.transform.SimilarityTransform(translation=-center_shift)
        tf_uncenter = skimage.transform.SimilarityTransform(translation=center_shift)

        tf_rotate = skimage.transform.AffineTransform(
            scale=scale,
            rotation=np.deg2rad(rotation),
            translation=translation,
            shear=np.deg2rad(shear)
        )

        self.tf = tf_center + tf_rotate + tf_uncenter
        self.kwargs = kwargs

    def transform(self, x):
        return skimage.transform.warp(
            x,
            self.tf,
            **self.kwargs
        )


class PiecewiseAffine(Transformation):
    """
    PiecewiseAffine transformation using :class:`skimage.transform.PiecewiseAffineTransform`.
    """
    def __init__(
            self,
            relaugmenttive_src,
            relaugmenttive_dst,
            **kwargs
    ):
        """
        :param relaugmenttive_src: Source coordinates.
        :param relaugmenttive_dst: Destination coordinates.
        :type relaugmenttive_src: (N, 2) array
        :type relaugmenttive_dst: (N, 2) array
        """

        tf = skimage.transform.PiecewiseAffineTransform()

        self.tf = tf
        self.relaugmenttive_src = relaugmenttive_src
        self.relaugmenttive_dst = relaugmenttive_dst
        self.kwargs = kwargs

    def transform(self, x):
        src = self.relaugmenttive_src.copy()
        src[:, 0] = src[:, 0] * x.shape[0]
        src[:, 1] = src[:, 1] * x.shape[1]

        dst = self.relaugmenttive_dst.copy()
        dst[:, 0] = dst[:, 0] * x.shape[0]
        dst[:, 1] = dst[:, 1] * x.shape[1]

        self.tf.estimate(src, dst)
        return skimage.transform.warp(
            x,
            self.tf,
            **self.kwargs
        )


class CropByFloat(Transformation):
    """
    Creates a crop of a given size relaugmenttiv to the top and left edge.
    """
    def __init__(self, width, height, top, left):
        """
        The image size should be in pixels while the edge distance should be the relaugmenttiv distance.

        :param width: Width of the crop in pixels.
        :param height: Height of the crop in pixels.
        :param top: Distance to the upper edge relaugmenttive to the image height. Must be between 0 and 1.
        :param left: Distance to the left edge relaugmenttive to the image width. Must be between 0 and 1.
        :type width: int
        :type height: int
        :type top: float
        :type left: float
        """
        self.width = width
        self.height = height
        self.top = top
        self.left = left

    def transform(self, x):
        left = int(round(self.left * (x.shape[1] - self.width)))
        top = int(round(self.top * (x.shape[0] - self.height)))

        return x[top:(top + self.height), left:(left + self.width)]


class CropAbsolute(Transformation):
    """
    Crops an image patch with the given size from the given position in pixels.
    """
    def __init__(self, width, height, top, left):
        """
        :param width: Width of the image patch to crop. Must be between 0.0 and the original image width
        :param height: Height of the image patch to crop. Must be between 0.0 and the original image height
        :param top: Margin from the top.
        :param left: Margin from the left.
        :type width: int
        :type height: int
        :type top: int
        :type left: int
        """
        self.width = width
        self.height = height
        self.top = top
        self.left = left

    def transform(self, x):
        return x[self.top:(self.top+self.height), self.left:(self.left+self.width)]


class CropRelaugmenttive(Transformation):
    """
    Crops an image patch relaugmenttive to its size.
    This transformation is especially interesting for random crops since the image size is not known at "draw"-time.
    """
    def __init__(self, width, height, top, left):
        """
        All values are independent of the actual image size in pixels.

        :param width: Width of the image patch to crop. Must be float between 0.0 and 1.0
        :param height: Height of the image patch to crop. Must be float between 0.0 and 1.0
        :param top: Margin from the top. Must be float between 0.0 and 1.0
        :param left: Margin from the left. Must be float between 0.0 and 1.0
        :type width: float
        :type height: float
        :type top: float
        :type left: float
        """
        self.width = width
        self.height = height
        self.top = top
        self.left = left

    def transform(self, x):
        left = int(round(self.left * x.shape[1]))
        width = int(round(self.width * x.shape[1]))

        top = int(round(self.top * x.shape[0]))
        height = int(round(self.height * x.shape[0]))

        return x[top:(top + height), left:(left + width)]


class CenterCrop(Transformation):
    """
    Crops a quadractic patch from the center of the input.
    """
    def __init__(self, crop_size):
        """
        :param crop_size: Size of the cropped image patch. Must be smaller than width and height of the input.
        :type crop_size: int
        """
        self.crop_size = crop_size

    def transform(self, x):
        height, width = x.shape[0], x.shape[1]
        vertical_diff = height - self.crop_size
        horizontal_diff = width - self.crop_size

        top = round(0.5 * vertical_diff)
        bottom = vertical_diff - top
        left = round(0.5 * horizontal_diff)
        right = horizontal_diff - left

        return x[floor(top):x.shape[0]-bottom, left:x.shape[1]-right]


class TopLeftCrop(Transformation):
    """
    Crops a quadratic path from the top-left corner of the input.
    """
    def __init__(self, crop_size):
        """
        :param crop_size: Size of the cropped image patch. Must be smaller than width and height of the input.
        :type crop_size: int
        """
        self.crop_size = crop_size

    def transform(self, x):
        return x[0:self.crop_size, 0:self.crop_size]


class TopRightCrop(Transformation):
    """
    Crops a quadratic path from the top-right corner of the input.
    """
    def __init__(self, crop_size):
        """
        :param crop_size: Size of the cropped image patch. Must be smaller than width and height of the input.
        :type crop_size: int
        """
        self.crop_size = crop_size

    def transform(self, x):
        height, width = x.shape[0], x.shape[1]
        return x[0:self.crop_size, width-self.crop_size:width]


class BottomLeftCrop(Transformation):
    """
    Crops a quadratic path from the bottom-left corner of the input.
    """
    def __init__(self, crop_size):
        """
        :param crop_size: Size of the cropped image patch. Must be smaller than width and height of the input.
        :type crop_size: int
        """
        self.crop_size = crop_size

    def transform(self, x):
        height, width = x.shape[0], x.shape[1]
        return x[height-self.crop_size:height, 0:self.crop_size]


class BottomRightCrop(Transformation):
    """
    Crops a quadratic path from the bottom-right corner of the input.
    """
    def __init__(self, crop_size):
        """
        :param crop_size: Size of the cropped image patch. Must be smaller than width and height of the input.
        :type crop_size: int
        """
        self.crop_size = crop_size

    def transform(self, x):
        height, width = x.shape[0], x.shape[1]
        return x[height-self.crop_size:height, width-self.crop_size:width]


class ResizeKeepRatio(Transformation):
    def __init__(self, size, operation=min):
        self.size = size
        self.operation = operation

    def transform(self, x):
        height, width = x.shape[0], x.shape[1]

        side = self.operation(height, width)
        rescale = side / self.size

        new_height = int(height / rescale)
        new_width = int(width / rescale)

        x = scipy.misc.imresize(x, (new_height, new_width))
        return x


class Resize(Transformation):
    """
    Resizes the input image.

    Wraps skimage.transform.resize so all arguments get passed along.
    See skimage.transform.resize for more information.
    """
    def __init__(self, output_shape, anti_aliasing=True, **kwargs):
        """
        Anti-aliasing true to hide warning (defaults to True in skimage 0.15).

        :param output_shape: (output_height, output_width)
        :param anti_aliasing: True to suppress warning.
        :param kwargs: Remaining keyword arguments are passed to skimage.transforms.resize.
        :type output_shape: (int, int)
        """
        self.output_shape = output_shape
        self.anti_aliasing = anti_aliasing
        self.kwargs = kwargs

    def transform(self, x):
        return skimage.transform.resize(x, self.output_shape, anti_aliasing=self.anti_aliasing, **self.kwargs)


class Padding(Transformation):
    """
    Adds padding to an image.
    """
    def __init__(self, padding=((0, 0), (0, 0), (0, 0)), mode='reflect'):
        """
        :param padding: Tuple of pairs following np.pad's padding convention.
        :param mode: See np.pad
        """
        self.padding = padding
        self.mode = mode

    def transform(self, x):
        padding = self.padding
        
        if len(x.shape) == 2:
            padding = padding[:2]
        if len(x.shape) == 3:
            padding = padding[:3]

        return np.pad(x, padding, mode=self.mode)


class Rotate90(Transformation):
    """
    Rotates the input image by k*90°.
    """
    def __init__(self, k):
        """
        :param k: Number of 90° rotations to do.
        :type k: int
        """
        self.k = k

    def transform(self, x):
        x = np.rot90(x, self.k)
        return x


class ColorAdjustment(Transformation):
    """
    Adjusts hue, saturation and brightness of RGB images.
    Clips the values between 0 and 1 for valid hsv images.
    """
    def __init__(self, hue_shift=0, sat_shift=0, val_shift=0):
        """
        :param hue_shift: Must be between 0.0 and 1.0.
        :param sat_shift: Must be between 0.0 and 1.0.
        :param val_shift: Must be between 0.0 and 1.0.
        :type hue_shift: float
        :type sat_shift: float
        :type val_shift: float
        """
        self.hue_shift = hue_shift
        self.sat_shift = sat_shift
        self.val_shift = val_shift

    def transform(self, x):
        hsv_image = skimage.color.rgb2hsv(x)
        hsv_image[:, :, 0] = ((hsv_image[:, :, 0] + self.hue_shift)*255 % 255) / 255
        hsv_image[:, :, 1] = np.clip(hsv_image[:, :, 1]*self.sat_shift, 0, 1)
        hsv_image[:, :, 2] = np.clip(hsv_image[:, :, 2]*self.val_shift, 0, 1)
        rgb_image = skimage.color.hsv2rgb(hsv_image)

        return rgb_image


class ColorPerturbation(Transformation):
    """
    Changes the image colors based on eigenvectors and eigenvalues.
    In difference to Krizhevsky et. al. this transformations does not use random alphas!

    Implements color perturbation from:
        "ImageNet Classification with Deep Convolutional Neural Networks"
        Alex Krizhevsky and Sutskever, Ilya and Hinton, Geoffrey E,
        http://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf
    """
    def __init__(self, alphas, clip=True):
        """

        :param alphas: Magnitude of perturbation.
        :param clip: If True clip output between 0 and 1.
        :type alphas: list of floats
        :type clip: bool
        """
        self.alphas = alphas
        self.clip = clip

    def transform(self, x):
        if len(x.shape) == 3:
            height, width, channels = x.shape
            img_rgb_col = x.reshape(height*width, channels)

        else:
            img_rgb_col = x

        cov = np.cov(img_rgb_col.T)
        eigvals, eigvects = np.linalg.eigh(cov)
        random_eigvals = np.sqrt(np.abs(eigvals)) * np.array(self.alphas)
        scaled_eigvects = np.dot(eigvects, random_eigvals)
        x = x + scaled_eigvects

        if self.clip:
            x = np.clip(x, 0, 1)

        return x


class NumpyCopy(Transformation):
    def transform(self, x):
        return np.copy(x)
