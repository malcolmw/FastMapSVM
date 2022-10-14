import cupy
import numpy as np

print('Importing fastmapsvm.accelerate')

def correlate(a, b, axis=-1):
    z = cupy.fft.fftshift(
        cupy.fft.irfft(
            cupy.fft.rfft(a, axis=axis)
            *
            cupy.conj(
                cupy.fft.rfft(b, axis=axis)
            )
        ),
        axes=axis
    )
    norm = cupy.sqrt(
        a.shape[-1] * cupy.var(a, axis=axis)
        *
        b.shape[-1] * cupy.var(b, axis=axis)
    )
    norm = norm[..., cupy.newaxis]

    return cupy.nan_to_num(z / norm, neginf=0, posinf=0)

def correlation_distance(a, b, axis=-1):
    '''
    Compute the pair-wise correlation distance matrix.
    '''
    return 1 - cupy.clip(
        cupy.max(
            cupy.nanmean(
                cupy.abs(
                    correlate(a, b, axis=axis)
                ),
                axis=-2
            ),
            axis=-1
        ),
        0, 1
    )

#def hilbert(x, axis=-1):
#    N = x.shape[axis]
#
#    Xf = cupy.fft.fft(x, N, axis=axis)
#    h = cupy.zeros(N, dtype=Xf.dtype)
#    if N % 2 == 0:
#        h[0] = h[N // 2] = 1
#        h[1:N // 2] = 2
#    else:
#        h[0] = 1
#        h[1:(N + 1) // 2] = 2
#
#    if x.ndim > 1:
#        ind = [np.newaxis] * x.ndim
#        ind[axis] = slice(None)
#        h = h[tuple(ind)]
#    x = cupy.fft.ifft(Xf * h, axis=axis)
#    return x
#
#def correlate(X, kernel, axis=-1):
#    if axis != -1:
#        raise NotImplementedError
#    N = max(X.shape[axis], kernel.shape[axis])
#    X = X - cupy.mean(X, axis=axis)[..., cupy.newaxis]
#    kernel = kernel - np.mean(kernel, axis=axis)[..., cupy.newaxis]
#    norm = N * cupy.std(X, axis=axis) * cupy.std(kernel, axis=axis)
#    cc = cupyx.scipy.ndimage.correlate1d(X, kernel, mode='constant')
#    cc /= norm[..., cupy.newaxis]
#    cc[~cupy.isfinite(cc)] = 0
#    return cc
#
#def hilbert_correlation_distance(X, kernel, axis=-1, stride=1):
#    Hx = cupy.abs(hilbert(X))
#    Hk = cupy.abs(hilbert(kernel))
#    cc = cupy.stack([
#        correlate(Hx[:, chan, ..., ::stride], Hk[chan, ..., ::stride])
#        for chan in range(3)
#    ], axis=1)
#    return 1 - cupy.max(cupy.mean(cupy.abs(cc), axis=1), axis=-1)
#
#def correlation_distance(X, kernel, axis=-1, stride=1):
#    cc = cupy.stack([
#        correlate(X[:, chan, ..., ::stride], kernel[chan, ..., ::stride])
#        for chan in range(3)
#    ], axis=1)
#    return 1 - cupy.max(cupy.mean(cupy.abs(cc), axis=1), axis=-1)
