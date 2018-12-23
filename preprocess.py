'''
Basic functions to preprocess frame data, helping us save on space.
'''

import numpy as np
def to_grayscale(image):
    return np.mean(image, axis=2).astype(np.uint8)
    
def downsample(image):
    return image[::2, ::2]
        
def preprocess(image):
    return to_grayscale(downsample(image))
