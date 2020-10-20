"""Factory for generating mask functions"""
import numpy as np


class MaskFnFactoryMeta(type):
    def __contains__(cls, name):
        return name in cls.builders

    @property
    def methods(cls):
        return list(cls.builders.keys())


class MaskFnFactory(metaclass=MaskFnFactoryMeta):
    """Factory for different mask generation methods"""
    builders = {}

    @classmethod
    def register(cls, name, build_fn):
        cls.builders[name] = build_fn

    @classmethod
    def create_mask_fn(cls, name, **kwargs):
        return cls.builders[name](**kwargs)


MaskFnFactory.register('nomask', lambda blocksize: lambda x: np.zeros_like(x)*255)
