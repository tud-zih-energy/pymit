from .entropy import *
from .mutualinformation import I, I_cond

import numpy as np
import mephisto as mp

_lib = None


def _set_library(lib):
    if not isinstance(lib, str):
        raise TypeError('Please specifiy lib as one of [np, numpy, mp, mephisto]')

    global _lib
    if lib in ['np', 'numpy']:
        _lib = np
        from .mutualinformation import _I_impl, _I_cond_impl, _transform3D
        _lib._I_impl = _I_impl
        _lib._I_cond_impl = _I_cond_impl
        _lib._transform3D = _transform3D
        del _I_impl, _I_cond_impl, _transform3D
    elif lib in ['mp', 'mephisto']:
        _lib = mp
    else:
        raise ValueError('Please specifiy lib as one of [np, numpy, mp, mephisto]')


_set_library('numpy')
