import numpy as np

from elaugment.image.transformations import FlipLr

t = FlipLr()

# Create dummy image
input_image = np.arange(9).reshape(3, 3)
print(input_image)

# Flip image
transformed_image = t(input_image)
print(transformed_image)