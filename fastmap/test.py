import core

import h5py
import numpy as np

get_array_module = lambda array: np

def correlation_distance(a, b, axis=-1):
    '''
    Compute the pair-wise correlation distance matrix.
    '''
    xp = get_array_module(a)
    return 1 - xp.clip(
        xp.max(
            xp.nanmean(
                xp.abs(
                    correlate(a, b, axis=axis)
                ),
                axis=-2
            ),
            axis=-1
        ),
        0, 1
    )
    

def correlate(a, b, axis=-1):
    xp = get_array_module(a)
    
    z = xp.fft.fftshift(
        xp.fft.irfft(
            xp.fft.rfft(a, axis=axis)
            *
            xp.conj(
                xp.fft.rfft(b, axis=axis)
            )
        ),
        axes=axis
    )
    norm = xp.sqrt(
        a.shape[-1] * xp.var(a, axis=axis)
        *
        b.shape[-1] * xp.var(b, axis=axis)
    )
    norm = norm[..., xp.newaxis]

    return xp.nan_to_num(z / norm, neginf=0, posinf=0)

    
class FastMap(core.FastMapABC):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._distance_func = correlation_distance


def test():

    import sklearn.pipeline
    import sklearn.preprocessing
    import sklearn.svm

    data_path = '../data/ridgecrest.hdf5'
    with h5py.File(data_path, mode='r') as in_file:
        X_train = in_file['/X/train'][:]
        y_train = in_file['/y/train'][:]
        
        X_test = in_file['/X/test'][:]
        y_test = in_file['/y/test'][:]
    
        
    pipe = sklearn.pipeline.Pipeline([
        ('fastmap', FastMap(2)),
        ('scaler', sklearn.preprocessing.StandardScaler()),
        ('svc', sklearn.svm.SVC())
    ])
    pipe.fit(X_train, y_train)
    print(pipe.score(X_test, y_test))
    
    W = pipe['fastmap'].transform(X_train)
    W_test = pipe['fastmap'].transform(X_test)
    
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    for i in range(2):
        idxs = np.argwhere(y_train == i).flatten()
        ax.scatter(
            W[idxs, 0],
            W[idxs, 1]
        )
    plt.show()
    

    fig, ax = plt.subplots()
    for i in range(2):
        idxs = np.argwhere(y_test == i).flatten()
        ax.scatter(
            W_test[idxs, 0],
            W_test[idxs, 1]
        )
    plt.show()
    
    return pipe
    

    
if __name__ == '__main__':
    pipe = test()
