# elaugment
elaugment is a Python package for reproducable data-augmentations. In difference to most other libraries random parameters for transformations are drawn seperate from the transformations. This makes it very easy apply the same transformations to several images. An example where this behaviour is useful is semantic segmentation, when you need to modify the input and the mask in the same way.

This library is currently in it's early stages so interfaces may break and some operations are slow. 

## Installation
1. Clone this repository
2. Run 
``` pip install elaugment ```

## Usage

elaugment essentially only provides two types classes: `Random` and `Transformation`. A objects of type `Random` must implement the `draw(rs)` method that generates a randomized transformation given a numpy random state. Then this transformation is returned a an instance of a `Transformation`-subclass. The numpy random state is necessary to draw random parameters from a fixed random seed. Here is a simple example:

```
import numpy as np
from elaugment.image.random import RandomCrop

rs = np.random.RandomState(seed=0)

rt = RandomCrop(224)
t1 = rt.draw(rs)
t2 = rt.draw(rs)

# a == b --> True
a = t1(original_image)
b = t1(original_image) 

# c == a --> False
c = t2(original_image)

```

A transformation is always deterministic, which means it can applied multiple times but will always apply the same transformation to the input. Any new call to a random-object, however, will return a different transformation each time.

## Examples
See `/examples` for an comprehensive overview. Or if github does not render the notebooks click [here](https://nbviewer.jupyter.org/github/Mctigger/elaugment/blob/master/examples/transformations.ipynb). 

Here is a list of some transformations that are current implemented:

```
transforms = {
    'Identity': Identity(),
    'FlipLr': transformations.FlipLr(),
    'FlipUd': transformations.FlipUd(),
    'Affine - rotation 30°': transformations.Affine(img.shape[0], rotation=30),
    'Affine - scale (3, 1,5)': transformations.Affine(img.shape[0], scale=(3, 1.5)),
    'Affine - translation (30, -30)': transformations.Affine(img.shape[0], translation=(30, -30)),
    'Affine - shear 30': transformations.Affine(img.shape[0], shear=30),
    'Affine - everything': transformations.Affine(
        img.shape[0], 
        shear=30, 
        rotation=30,
        scale=(2, 1.5),
        translation=(30, -30)
    ),
    'CropAbsolute': transformations.CropAbsolute(256, 256, 128, 128),
    'CropRelaugmenttive': transformations.CropRelaugmenttive(0.5, 0.5, 0.25, 0.25),
    'CropByFloat': transformations.CropByFloat(256, 256, top=0.25, left= 0.25),
    'CenterCrop': transformations.CenterCrop(crop_size=256),
    'TopLeftCrop': transformations.TopLeftCrop(crop_size=256),
    'TopRightCrop': transformations.TopRightCrop(crop_size=256),
    'BottomLeftCrop': transformations.BottomLeftCrop(crop_size=256),
    'BottomRightCrop': transformations.BottomRightCrop(crop_size=256),
    'Resize': transformations.Resize((256, 256), mode='reflect'),
    'Rotate90 - k=1': transformations.Rotate90(k=1),
    'Rotate90 - k=2': transformations.Rotate90(k=2),
    'ColorAdjustment': transformations.ColorAdjustment(hue_shift=0.5, sat_shift=0.8, val_shift=0.9),
    'ColorPertubation': transformations.ColorPerturbation(alphas=[0.2, 0.2, 0.2]),
    'RandomDistortion': random.RandomDistortion(5, 5, 0.1, 0.1).draw(rs),
}
```

## Contribute
Transformations are easy to integrate into elaugment, so just create pull-requests when you feel like it!
