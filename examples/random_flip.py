import numpy as np

from elaugment.image.random import RandomFlipLr

rs = np.random.RandomState()
rt = RandomFlipLr()

# Create dummy image
input_image = np.arange(9).reshape(3, 3)
print(input_image)

# Random flip image 
transformed_image = rt.draw(rs)(input_image)
print(transformed_image)
