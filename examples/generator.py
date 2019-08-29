import numpy as np

from elaugment.generator import TransformationsGenerator
from elaugment.image.random import RandomFlipLr, RandomFlipUd, RandomCrop

tg = TransformationsGenerator([
    RandomFlipLr(), 
    RandomFlipUd(),
    RandomCrop(3)
])

# Create dummy image.
input_image = np.arange(25).reshape(5, 5)
print(input_image)

# Draw a random transformation t. t is a normal, deterministic transformation. 
t = next(tg)
# Apply t.
transformed_image = t(input_image)
print(transformed_image)
