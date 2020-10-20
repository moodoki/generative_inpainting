"""Mask generator functions"""
from . import edge_guided                               # noqa
from . import dct_score                                 # noqa
from .mask_factory import MaskFnFactory

__all__ = ['MaskFnFactory']
