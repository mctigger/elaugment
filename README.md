# elaugment
elaugment is a Python package for reproducable data-augmentations. In difference to most other libraries random parameters for transformations are drawn seperate from the transformations. This makes it very easy apply the same transformations to several images. An example where this behaviour is useful is semantic segmentation, when you need to modify the input and the mask in the same way.

This library is currently in it's early stages so interfaces may break and some operations are slow. 

## Installation
1. Clone this repository
2. Run 
``` pip install elaugment ```

## Examples & Usage
See `/examples` for an comprehensive overview. Or if github does not render the notebooks click [here](https://nbviewer.jupyter.org/github/Mctigger/elaugment/blob/master/examples/transformations.ipynb). 

## Contribute
Transformations are easy to integrate into elaugment, so just create pull-requests when you feel like it!
