import numpy as np
import pandas as pd
import scipy.signal

def correlate(a, b, mode="valid"):

    if len(a) > len(b):
        a, b = b, a

    a = pd.Series(a)
    b = pd.Series(b)
    n = len(a)

    a = a - np.mean(a)
    b = b - np.mean(b)

    c = scipy.signal.correlate(b, a, mode=mode)

    # if mode == "valid":
    #     norm = n * np.std(a) * b.rolling(n).std().dropna().values
    # elif mode == "same":
    #     norm = n * np.std(a) * b.rolling(n, min_periods=0, center=True).std().values

    norm = n * np.std(a) * np.std(b)
    if norm == 0:
        c[:] = 0
    else:
        c /= norm

    return (c)


def distance(
    obj_a,
    obj_b,
    mode="valid",
    # reduce=_reduce,
    force_triangle_ineq=False
):
    """
    Return the distance between object obj_a and object obj_b.

    Arguments:
    - obj_a: object
        First object to consider.
    - obj_b: object
        Second object to consider.
    """
    dist = 1 - np.max(np.abs(ndcorrelate(obj_a, obj_b, mode=mode, reduce=reduce)))

    if force_triangle_ineq is True:
        if dist == 0:
            return (0)
        else:
            return ((dist + 1) / 2)

    else:
        return (dist)


def ndcorrelate(a, b, mode="valid"):

    assert a.ndim == b.ndim, "a and b must have the same number of dimensions"

    if a.ndim == 1:
        return (correlate(a, b, mode=mode))

    assert a.shape[:-1] == b.shape[:-1]

    na, nb = a.shape[-1], b.shape[-1]

    if na > nb:
        a, b = b, a
        na, nb = nb, na

    a = a.reshape(-1, na)
    b = b.reshape(-1, nb)
    n = a.shape[0]

    if mode == "valid":
        c = np.zeros((n, nb - na + 1))
    elif mode == "same":
        c = np.zeros((n, nb))
    for i in range(n):
        c[i] = correlate(a[i], b[i], mode=mode)

    return (c)
