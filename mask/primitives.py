"""Basic mask primitives used by other mask generators"""
import numpy as np

from .mask_ops import expand_mask


def random_mask(image,
                ratio=0.5,
                blocksize=1,
                dtype='uint8'):
    """Random mask with probability given by `ratio`"""

    height, width, _ = image.shape

    _len = height*width//blocksize//blocksize
    _mask = np.zeros(_len)
    _mask[:int(_len*ratio)] = 255
    np.random.shuffle(_mask)
    _mask = _mask.reshape((height//blocksize,
                           width//blocksize))

    mask = expand_mask(_mask, blocksize=blocksize, dtype=dtype)

    return mask

def random_mask_base(shape, probability=0.5):
    return probability < np.random.random(shape)

def grid_mask(image,
              period=2,
              blocksize=1,
              dtype='uint8'):

    h, w, _ = image.shape

    mask = np.ones((h//blocksize, w//blocksize), dtype=dtype)
    mask[::period, ::period] = False

    return mask
