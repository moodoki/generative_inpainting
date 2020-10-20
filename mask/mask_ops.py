import numpy as np


def expand_mask(mask, blocksize, channels=3, dtype='uint8', h=-1, w=-1):
    """Expand mask with of minimal size so that each pixel in
    mask is blocksize x blocksize. Final dimension is also expanded
    and repeated to n channels
    """
    #h, w = mask.shape
    _mask = np.zeros((h*blocksize, w*blocksize), dtype=dtype)
    if blocksize == 1:
        _mask = mask
    else:
        for i in range(blocksize):
            for j in range(blocksize):
                _mask[i::blocksize, j::blocksize] = mask

    _mask = np.expand_dims(_mask, axis=-1)

    mask = np.concatenate([_mask] * channels, axis=2)
    return mask


def reduce_mask(mask, blocksize):
    if len(mask.shape) == 3:
        mask = mask[:, :, 0]

    return np.array(mask[::blocksize, ::blocksize] > 0)


def slice_block(in_arr, i, j, blocksize):
    return in_arr[i*blocksize:(i+1)*blocksize, j*blocksize:(j+1)*blocksize]
