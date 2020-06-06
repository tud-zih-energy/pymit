from .entropy import *
from .mutualinformation import *

import numpy as np
import mephisto as mp

_lib = np


def _set_library(lib):
    if not isinstance(lib, str):
        raise TypeError('Please specifiy lib as one of [np, numpy, mp, mephisto]')

    global _lib
    if lib in ['np', 'numpy']:
        _lib = np
    elif lib in ['mp', 'mephisto']:
        _lib = mp
    else:
        raise ValueError('Please specifiy lib as one of [np, numpy, mp, mephisto]')
