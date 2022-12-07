"""Operations which handle numpy and tensorflow.compat.v1 automatically."""

import numpy as np

from .. import backend as bkd
from .. import config
from ..backend import is_tensor, tf, as_tensor, paddle, get_preferred_backend


def istensorlist(values):
    return any(map(is_tensor, values))


def convert_to_array(value):
    """Convert a list to numpy array or tensorflow tensor."""
    if istensorlist(value):
        if get_preferred_backend() == "paddle":
            return bkd.concat(value, axis=0)  # for paddle
        return as_tensor(value, dtype=config.real(bkd.lib))  # for tf
    value = np.array(value)
    if value.dtype != config.real(np):
        return value.astype(config.real(np))
    return value


def hstack(tup):
    if not is_tensor(tup[0]) and isinstance(tup[0], list) and tup[0] == []:
        tup = list(tup)
        if istensorlist(tup[1:]):
            tup[0] = bkd.as_tensor([], dtype=config.real(bkd.lib))
        else:
            tup[0] = np.array([], dtype=config.real(np))
    return bkd.concat(tup, 0) if is_tensor(tup[0]) else np.hstack(tup)


def roll(a, shift, axis):
    return tf.roll(a, shift, axis) if is_tensor(a) else np.roll(a, shift, axis=axis)


def zero_padding(array, pad_width):
    # SparseTensor
    if isinstance(array, (list, tuple)) and len(array) == 3:
        indices, values, dense_shape = array
        indices = [(i + pad_width[0][0], j + pad_width[1][0]) for i, j in indices]
        dense_shape = (
            dense_shape[0] + sum(pad_width[0]),
            dense_shape[1] + sum(pad_width[1]),
        )
        return indices, values, dense_shape
    if is_tensor(array):
        return tf.pad(array, tf.constant(pad_width))
    return np.pad(array, pad_width)


def padding_array(array, nprocs):
    npad = (nprocs - len(array) % nprocs) % nprocs  # pad npad elements, %nprocs at last in case of nprocs=1
    if npad == 0:
        return array
    if isinstance(array, np.ndarray):
        datapad = array[-1, :].reshape([-1, array[-1, :].shape[0]])
        for i in range(npad):
            array = np.append(array, datapad, axis=0)
    return array


def sub(array, nprocs, n):
    """sub

    Args:
        nprocs (int): number of GPUs
        n (int): index of GPU
    """
    N = len(array)
    subN = N // nprocs
    s = subN * n
    e = subN * (n + 1)
    return array[s: e]
