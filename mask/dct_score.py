"""Mask generation based on DCT complexity"""
from functools import partial
import tensorflow as tf
import numpy as np
from absl import logging
from .mask_ops import expand_mask
from .mask_factory import MaskFnFactory

def _tf_dct2(a):
    a_x = tf.transpose(a, [0, 1, 2, 4, 5, 3])
    dct_x = tf.signal.dct(a_x, norm='ortho')
    a = tf.transpose(dct_x, [0, 1, 2, 5, 3, 4])

    a_y = tf.transpose(a, [0, 1, 2, 3, 5, 4])
    dct_y = tf.signal.dct(a_y, norm='ortho')
    a = tf.transpose(dct_y, [0, 1, 2, 3, 5, 4])
    return a

def _tf_idct2(a):
    a_x = tf.transpose(a, [0, 1, 2, 4, 5, 3])
    dct_x = tf.signal.idct(a_x, norm='ortho')
    a = tf.transpose(dct_x, [0, 1, 2, 5, 3, 4])

    a_y = tf.transpose(a, [0, 1, 2, 3, 5, 4])
    dct_y = tf.signal.idct(a_y, norm='ortho')
    a = tf.transpose(dct_y, [0, 1, 2, 3, 5, 4])
    return a

def _tf_img_to_patches(img, patchsize=8):
    sizes = [1, patchsize, patchsize, 1]
    strides = [1, patchsize, patchsize, 1]
    img_patches = tf.image.extract_patches(img,
                                           sizes=sizes,
                                           strides=strides,
                                           rates=[1, 1, 1, 1],
                                           padding='VALID'
                                          )
    img_patches = tf.reshape(img_patches,
        [img_patches.shape[0], img_patches.shape[1], img_patches.shape[2],
            patchsize, patchsize, img.shape[-1]])
    return img_patches

def _tf_patches_to_img(patches):
    img_concat_r = tf.concat(tf.split(patches, patches.shape[1], axis=1), axis=3)

    img_concat_c = tf.concat(tf.split(img_concat_r, patches.shape[2], axis=2), axis=4)
    img_composed = tf.squeeze(img_concat_c, axis=[1,2])

    return img_composed

def tf_block_dct2d(im, blocksize=8, dtype=tf.float64, rearrange=True):
    if len(im.shape) == 3:
        is_batched = False                            # pylint: disable=unused-variable
        im = tf.expand_dims(im, axis=0)

    im = tf.cast(im, dtype)

    im_patches = _tf_img_to_patches(im, blocksize)
    im_dct = _tf_dct2(im_patches)
    if rearrange:
        return _tf_patches_to_img(im_dct)

    return im_dct

def tf_block_idct2d(im_dct, blocksize=8, dtype=tf.float64, rearrange=True):
    if len(im.shape) == 3:
        is_batched = False                            # pylint: disable=unused-variable
        im = tf.expand_dims(im, axis=0)
    im_dct = tf.cast(im_dct, dtype)

    im_dct_patches = _tf_img_to_patches(im_dct, blocksize)
    im_patches = _tf_idct2(im_dct_patches)
    if rearrange:
        return _tf_patches_to_img(im_patches)

    return im_patches


def compute_patch_scores(img_dct_patches, threshold=128):
    thresholded_patches = tf.abs(img_dct_patches) > threshold
    return tf.math.count_nonzero(thresholded_patches, axis=[-3, -2, -1])


def rgb_to_yuv(images):
    images = tf.convert_to_tensor(images)
    """Thin wrapper around `tf.image.rgb_to_yuv` to ensure datatype is float32"""
    if images.dtype == tf.uint8:
        logging.info('converting image dtype from uint8 to float32')
        images = tf.image.convert_image_dtype(images, tf.float32, saturate=True)
    images_yuv = tf.image.rgb_to_yuv(images)
    return images_yuv


def rgb_to_ycbcr(images):
    images = tf.convert_to_tensor(images)
    """YUV vs YCbCr difference is that YCbCr is digital"""
    images_yuv = rgb_to_yuv(images)
    images_ycbcr = tf.image.convert_image_dtype(images_yuv, tf.uint8, saturate=True)
    return images_ycbcr


def large_dct_comps(blocksize=8, dct_threshold=64, scores_threshold=0.7, cvt_color=True):

    def mask_fn(image):
        image = rgb_to_ycbcr(image) if cvt_color else image
        img_dct = tf_block_dct2d(image, blocksize=blocksize, rearrange=False)

        scores = compute_patch_scores(img_dct, threshold=dct_threshold)
        max_score = tf.cast(tf.reduce_max(scores), tf.float32)
        s_threshold = tf.cast(max_score * scores_threshold, tf.int64)
        mask = tf.squeeze(scores > s_threshold)
        mask = np.array(mask, dtype='uint8')
        logging.info(f'{np.count_nonzero(mask)} of {mask.shape}')

        mask = expand_mask(mask, blocksize, channels=image.shape[-1], dtype='uint8')

        return mask

    return mask_fn


def keep_mid_dct_comps(blocksize=8, dct_threshold=64, scores_threshold=(0.15, 0.7), cvt_color=True):

    def mask_fn(image):
        image = rgb_to_ycbcr(image) if cvt_color else image
        img_dct = tf_block_dct2d(image, blocksize=blocksize, rearrange=False)

        scores = compute_patch_scores(img_dct, threshold=dct_threshold)
        max_score = tf.cast(tf.reduce_max(scores), tf.float32)
        s_threshold = tf.cast(max_score * scores_threshold[1], tf.int64)
        sl_threshold = tf.cast(max_score * scores_threshold[0], tf.int64)

        mask_large = np.array(tf.squeeze(scores > s_threshold))
        mask_small = np.array(tf.squeeze(scores < sl_threshold))
        mask = np.array(mask_large | mask_small, dtype='uint8')*255
        logging.info(f'{np.count_nonzero(mask)} of {mask.shape}')

        mask = expand_mask(mask, blocksize, channels=image.shape[-1], dtype='uint8')

        return mask

    return mask_fn

def discard_mid_dct_comps(blocksize=8, dct_threshold=64, scores_threshold=(0.4, 0.6), cvt_color=True):

    def mask_fn(image):
        image = rgb_to_ycbcr(image) if cvt_color else image
        img_dct = tf_block_dct2d(image, blocksize=blocksize, rearrange=False)

        scores = compute_patch_scores(img_dct, threshold=dct_threshold)
        max_score = tf.cast(tf.reduce_max(scores), tf.float32)
        s_threshold = tf.cast(max_score * scores_threshold[1], tf.int64)
        sl_threshold = tf.cast(max_score * scores_threshold[0], tf.int64)

        mask_large = np.array(tf.squeeze(scores < s_threshold))
        mask_small = np.array(tf.squeeze(scores > sl_threshold))
        mask = np.array(mask_large & mask_small, dtype='uint8')*255
        logging.info(f'{np.count_nonzero(mask)} of {mask.shape}')

        mask = expand_mask(mask, blocksize, channels=image.shape[-1], dtype='uint8')

        return mask

    return mask_fn


def random_large_dct(blocksize=8, dct_threshold=64, scores_threshold=0.7, mask_prob=0.9, cvt_color=True):
    def mask_fn(image):
        image = rgb_to_ycbcr(image) if cvt_color else image
        img_dct = tf_block_dct2d(image, blocksize=blocksize, rearrange=False)

        scores = compute_patch_scores(img_dct, threshold=dct_threshold)
        max_score = tf.cast(tf.reduce_max(scores), tf.float32)
        s_threshold = tf.cast(max_score * scores_threshold, tf.int64)

        mask_large = np.array(tf.squeeze(scores > s_threshold))
        mask_random = mask_prob < np.random.random(mask_large.shape)
        mask = np.array(mask_large | mask_random, dtype='uint8')*255
        logging.info(f'Discarding {np.count_nonzero(mask)} of {mask.shape}')

        mask = expand_mask(mask, blocksize, channels=image.shape[-1], dtype='uint8')

        return mask

    return mask_fn


def random_small_dct(blocksize=8, dct_threshold=64, scores_threshold=0.17, mask_prob=0.3, cvt_color=True):
    def mask_fn(image):
        image = rgb_to_ycbcr(image) if cvt_color else image
        img_dct = tf_block_dct2d(image, blocksize=blocksize, rearrange=False)

        scores = compute_patch_scores(img_dct, threshold=dct_threshold)
        max_score = tf.cast(tf.reduce_max(scores), tf.float32)
        s_threshold = tf.cast(max_score * scores_threshold, tf.int64)

        mask_large = np.array(tf.squeeze(scores < s_threshold))
        mask_random = mask_prob < np.random.random(mask_large.shape)
        mask = np.array(mask_large & mask_random, dtype='uint8')*255
        logging.info(f'Discarding {np.count_nonzero(mask)} of {mask.shape}')

        mask = expand_mask(mask, blocksize, channels=image.shape[-1], dtype='uint8')

        return mask

    return mask_fn

def binarize_by_range(in_scores, normalize=True,
                      threshold_low=0, threshold_high=1,
                      mode='include', random=0.3):
    """
    `include` or `exclude` patches in mask with scores between low and high
    thresholds.
    random is (0, 1] for number of patches to discard randomly.
    Set to <= 0 so that no patches are discarded randomly.
    """
    logging.info(f'{mode}-ing {threshold_low} to {threshold_high} + random {random}')
    if normalize:
        max_score = tf.cast(tf.reduce_max(in_scores), tf.float32)
        scores = tf.cast(in_scores, tf.float32)/max_score
    else:
        scores = in_scores

    # regions where mask is true will be removed

    # patches with scores lower than `threshold_low` set to masked_out
    m1 = np.array(tf.squeeze(scores < threshold_low), dtype=np.bool)
    # patches with scores higher than `threshold_high` set to masked_out
    m2 = np.array(tf.squeeze(scores > threshold_high), dtype=np.bool)

    if mode == 'exclude':
        m1 = np.logical_not(m1)
        m2 = np.logical_not(m2)
        m = m1 & m2
    elif mode == 'include':
        m = m1 | m2
    else:
        raise ValueError(f'Mode {mode} not supported', mode)
    mr = _random_mask(abs(random), m1.shape)

    if random >= 0:
        mask = np.array(m | mr, dtype='uint8')
    else:
        mask = np.array(m & mr, dtype='uint8')
    logging.info(f'******** Mask shape: {m1.shape}, {m2.shape}, {mr.shape}, {mask.shape}')
    logging.info(f'******** Mask shape: {np.count_nonzero(m1)}, {np.count_nonzero(m2)}, {np.count_nonzero(mr)}, {np.count_nonzero(mask)}')

    return mask

__zigzag_idx = np.reshape(np.arange(64), (8,8))
__zigzag_idx = np.concatenate([np.diagonal(__zigzag_idx[::-1,:], k)[::(2*(k % 2)-1)] for k in range(1-__zigzag_idx.shape[0], __zigzag_idx.shape[0])])
def __idx_to_rc(idx, width=8, height=8):
    w = idx % width
    h = idx // height
    return h, w

__zigzag_idx = list(map(__idx_to_rc, __zigzag_idx))

def zigzag_8x8_tf(input_block):
    print(__zigzag_idx)
    return tf.gather_nd(input_block, [__zigzag_idx, __zigzag_idx], batch_dims=1)

def zigzag_8x8_tf_blocks(img_dct):
    print("*****************************************", img_dct)
    img_t = tf.transpose(img_dct, [0, 3, 4, 5, 1, 2])
    print("*****************************************", img_t)
    z_t = zigzag_8x8_tf(img_t)
    print("*****************************************", z_t)
    z = tf.transpose(z_t, [0, 3, 4, 1, 2])
    print("*****************************************", z)
    return z

def frequency_weighted_score(image_dct, threshold=64):

    # Extra weighting on chroma, losely based on psychovisual
    luma_weights = np.arange(64, dtype=np.float32)/64
    chroma_weights = np.arange(64, dtype=np.float32)/128
    component_weights = np.stack([luma_weights, chroma_weights, chroma_weights], axis=-1)

    # zero out small components
    logging.info(image_dct.shape)
    image_dct = tf.where(image_dct < threshold,  tf.cast(0, tf.float64), image_dct)

    img_vectorized = zigzag_8x8_tf_blocks(image_dct)
    logging.info(img_vectorized.shape)
    scores = tf.reduce_sum(img_vectorized * component_weights, axis=[-1, -2])

    logging.info(scores.shape)

    return scores


_noop = lambda x: x
_random_mask = lambda mask_prob, shape: mask_prob < np.random.random(shape)

class DctMaskGenerator:
    """Generic mask generator based on DCT component"""
    def __init__(self,
                 preprocess_fn='to_ycbcr',
                 scoring_fn='dct_patch_128',
                 binarizer_fn='include_0_0.7_0.3',
                 blocksize=8,
                 FLAGS=None,
                 ):
        super().__init__()

        self._preprocess = self._get_preprocess_fn(preprocess_fn)
        self._compute_block_scores = self._get_scoring_fn(scoring_fn)
        self._binarize_mask = self._get_binarizer_fn(binarizer_fn)
        self._scores = None
        self.blocksize = blocksize
        self.flags = FLAGS

    def _get_preprocess_fn(self, preprocess_fn):
        if preprocess_fn is None:
            logging.info('No preprocessing')
            return _noop
        elif callable(preprocess_fn):
            logging.info('Using custom callable preprocess')
            return preprocess_fn
        elif preprocess_fn == 'to_ycbcr':
            logging.info('Converting to YCbCr')
            return rgb_to_ycbcr
        elif preprocess_fn == 'to_yuv':
            logging.info('Converting to YUV')
            return tf.image.rgb_to_yuv

        raise ValueError(f'Unsupported preprocess {preprocess_fn}', preprocess_fn)

    def _get_scoring_fn(self, scoring_fn):
        if scoring_fn is None:
            return _noop
        elif callable(scoring_fn):
            return scoring_fn
        elif isinstance(scoring_fn, str) and 'dct_patch_' in scoring_fn:
            threshold = int(scoring_fn.split('_')[-1])
            logging.info(f'DCT patch scoring with threshold:{threshold}')
            return partial(compute_patch_scores, threshold=threshold)
        elif isinstance(scoring_fn, str) and 'dct_fw_' in scoring_fn:
            threshold = int(scoring_fn.split('_')[-1])
            logging.info(f'DCT frequency weighted scoring with threshold:{threshold}')
            return partial(frequency_weighted_score, threshold=threshold)

        raise ValueError(f'Unsupported scoring method {scoring_fn}', scoring_fn)

    def _get_binarizer_fn(self, binarizer_fn):
        if binarizer_fn is None:
            return _noop
        elif callable(binarizer_fn):
            return binarizer_fn
        elif isinstance(binarizer_fn, str):
            mode, low, high, rand = binarizer_fn.split('_')
            return partial(binarize_by_range,
                           mode=mode,
                           threshold_low=float(low),
                           threshold_high=float(high),
                           random=float(rand))

        raise ValueError(f'Unsupported binarizer method {binarizer_fn}', binarizer_fn)

    def __call__(self, image):
        if len(image.shape) == 4:
            shape=image.shape[1:-1]
        elif len(image.shape) == 3:
            shape=image.shape[:-1]
        else:
            raise ValueError(f"Expects rank 3 or 4 tensor, got {image.shape}", image)
        img = self._preprocess(image)
        img_dct = tf_block_dct2d(img, blocksize=self.blocksize, rearrange=False)
        self._scores = self._compute_block_scores(img_dct)
        mask = tf.py_function(self._binarize_mask, [self.scores], tf.uint8)
        mask = tf.py_function(expand_mask,
            [mask, self.blocksize, 1, 'uint8', self.flags.height, self.flags.width],
            tf.uint8
        )
        print('_+_+_+_+_+_+_+_+_+_', mask)
        mask.set_shape(image.shape)
        print('_+_+_+_+_+_+_+_+_+_', mask)
        mask = tf.reduce_sum(mask, axis=0, keepdims=True)


        return mask

    @property
    def scores(self):
        return self._scores


MaskFnFactory.register('discard_large_dct', large_dct_comps)
MaskFnFactory.register('keep_mid_dct', keep_mid_dct_comps)
MaskFnFactory.register('discard_mid_dct', discard_mid_dct_comps)
MaskFnFactory.register('random_large_dct', random_large_dct)
MaskFnFactory.register('random_small_dct', random_small_dct)

MaskFnFactory.register('ycc_discard_random_mid', partial(DctMaskGenerator,
                                                 preprocess_fn='to_ycbcr',
                                                 scoring_fn='dct_patch_64',
                                                 binarizer_fn='exclude_0.5_0.7_0.3'
                                                 ))
MaskFnFactory.register('dct_generic', DctMaskGenerator)
