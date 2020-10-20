"""Implementations for mask generator functions"""
from functools import partial
import numpy as np
import cv2
from .mask_ops import expand_mask, slice_block
from .primitives import random_mask, grid_mask
from .mask_factory import MaskFnFactory


def _luma_edge_scores(image, edge_mode, blocksize):
    h, w, _ = image.shape

    if edge_mode == 'default':
        image_luma = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
        edge = cv2.Canny(image_luma[:, :, 0], 128, 64)/255
        score = np.zeros((h//blocksize, w//blocksize))
    else:
        raise ValueError(f'{edge_mode} not supported')

    prob_scale = blocksize*blocksize/2
    for i in range(h//blocksize):
        for j in range(w//blocksize):
            score[i, j] = np.sum(slice_block(edge, i, j, blocksize))/prob_scale

    return score


def edge_guided_mask(image, edge_mode='default', blocksize=8, min_prob=0.01,
                     with_prob=False
                     ):
    """Randomly generated mask, patches with more edges are given higher probabilites"""
    keep_prob = np.clip(_luma_edge_scores(image, edge_mode, blocksize) + min_prob, min_prob, 1.0)

    _mask = np.array(keep_prob < np.random.random(
        keep_prob.shape), dtype='uint8')*255

    mask = expand_mask(_mask, blocksize=blocksize, dtype='uint8')

    if with_prob:
        return mask, keep_prob
    else:
        return mask


def grid_edge_threshold(image, edge_mode='default', blocksize=8, threshold=0.1,
                        grid_period=2,
                        with_scores=False):
    """Regular grid | edge score
    Edge scores are computed for each patch.
    Patches are OR-ed with a regular grid with periodicity `grid_prob`
    """
    scores = _luma_edge_scores(image, edge_mode, blocksize)
    grid = grid_mask(image, period=grid_period, blocksize=blocksize, dtype='bool')

    scores_mask = scores < threshold
    _mask = np.array(scores_mask & grid, dtype='uint8') * 255

    mask = expand_mask(_mask, blocksize=blocksize, dtype='uint8')

    if with_scores:
        return mask, scores

    return mask


def random_edge_threshold(image, edge_mode='default', blocksize=8, threshold=0.1,
                          mask_prob=0.25,
                          with_scores=False):
    """Random | edge score
    Edge scores are computed for each patch.
    Patches are OR-ed with a randomly generated mask of prob `mask_prob`
    """
    scores = _luma_edge_scores(image, edge_mode, blocksize)

    r_mask = mask_prob < np.random.random(scores.shape)

    scores_mask = scores < threshold
    _mask = np.array(scores_mask & r_mask, dtype='uint8') * 255

    mask = expand_mask(_mask, blocksize=blocksize, dtype='uint8')

    if with_scores:
        return mask, scores

    return mask


# Register functions
MaskFnFactory.register('edge', lambda blocksize: partial(edge_guided_mask,
                                                         blocksize=blocksize,
                                                         with_prob=False,
                                                         min_prob=0.5))
MaskFnFactory.register('random', lambda blocksize: partial(random_mask,
                                                           blocksize=blocksize,
                                                           ratio=0.3))
MaskFnFactory.register('edge_threshold', lambda blocksize: partial(grid_edge_threshold,
                                                                   blocksize=blocksize))
MaskFnFactory.register('edge_threshold_random', lambda blocksize: partial(random_edge_threshold,
                                                                          blocksize=blocksize,
                                                                          mask_prob=0.33))
