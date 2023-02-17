#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 16 14:24:10 2023

@author: malcolmw
"""

print('Importing fastmap.core')

# _xp_ is the array module to use for batch distance calculations.
# It can be either numpy or cupy. By default, it is numpy.
global _xp_
global get_array
global get_array_module

import h5py
import numpy as np
import pathlib
import tqdm

_xp_ = np
get_array = lambda array: array
get_array_module = lambda array: np

DEFAULT_BATCH_SIZE = 1024
EPSILON = 1e-9

class FastMap:

    def __init__(self, distance_func, n_dim, model_path, overwrite=False):
        '''
        Implements the FastMap algorithm.

        Parameters
        ----------
        distance_func : function
            The distance function D(A, B) that defines the distance between
            objects A and B.
        n_dim : int
            The number of Euclidean dimensions.
        model_path : str, pathlib.Path
            Path to store model.
        overwrite : bool, optional
            Overwrite pre-existing model if one exists. The default is False.

        Returns
        -------
        None.

        '''
        self._distance_func = distance_func
        self._ihyprpln = 0
        self._n_dim = n_dim
        self._init_hdf5(model_path, overwrite=overwrite)
        
        
    def __enter__(self):
        return self
    
    
    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
        
        
    @property
    def hdf5(self):
        '''
        Returns
        -------
        h5py.File
            HDF5 model backend.

        '''
        return self._hdf5
        
        
    @property
    def n_dim(self):
        '''
        Returns
        -------
        int
            Dimensionality of embedding.

        '''
        return self._n_dim
    
    
    @property
    def n_obj(self):
        '''
        Returns
        -------
        int
            The number of objects in the train set.

        '''
        return len(self.X)
    
    @property
    def pivot_ids(self):
        '''
        Returns
        -------
        h5py.DataSet
            Indices of pivot objects.

        '''
        if '/pivot_ids' not in self.hdf5:
            self.hdf5.create_dataset(
                '/pivot_ids',
                (self.n_dim, 2),
                np.uint16,
                fillvalue=np.nan
            )

        return self.hdf5['/pivot_ids']
    
    @property
    def supervised(self):
        '''
        Returns
        -------
        bool
            Whether the embedding is supervised.

        '''
        return self._supervised
    
    
    @property
    def W(self):
        '''
        Returns
        -------
        numpy.array
            Images of embedded objects.

        '''
        if not hasattr(self, '_W'):
            self._W = np.full(
                (self.n_obj, self.n_dim), 
                np.nan,
                dtype=np.float32
            )

        return self._W
    
    
    @property
    def W_piv(self):
        if '/W_piv' not in self.hdf5:
            self.hdf5.create_dataset(
                '/W_piv',
                (self.n_dim, 2, self.n_dim),
                np.float32,
                fillvalue=np.nan
            )

        return self.hdf5['/W_piv']

    
    @property
    def X(self):
        '''
        Returns
        -------
        numpy.array or cupy.array
            Embedded objects in original data domain.

        '''
        return self._X

    @X.setter
    def X(self, value):
        self._X = _xp_.array(value)
        
        
    @property
    def X_piv(self):
        if '/X_piv' not in self.hdf5:
            self.hdf5.create_dataset(
                '/X_piv',
                (self.n_dim, 2, *self.X.shape[1:]),
                self.X.dtype,
                fillvalue=np.nan
            )

        return self.hdf5['/X_piv']
        
        
    @property
    def y(self):
        '''
        Returns
        -------
        numpy.array
            Class labels of training data if run in supervised mode.

        '''
        return self._y

    @y.setter
    def y(self, value):
        if value is not None:
            self._y = np.array(value)
            self._supervised = True
        else:
            self._supervised = False
        
        
    def _choose_pivots(self, n_proc=None):
        '''
        A heuristic algorithm to choose distant pivot objects adapted
        from Faloutsos and Lin (1995).

        Parameters
        ----------
        n_proc : int, optional
            The number of processors to use if running in multiprocessing mode.
            The default is None.

        Returns
        -------
        i_obj : int
            The index of pivot object #1.
        j_obj : int
            The index of pivot object #2.

        '''

        forbidden = self.pivot_ids[:self._ihyprpln].flatten()

        while True:
            if self.supervised is True:
                idxs = np.argwhere(self.y == 1).flatten()
            else:
                idxs = np.arange(self.n_obj)
            j_obj = np.random.choice(idxs)
            if j_obj not in forbidden:
                break

        furthest = self.furthest(
            j_obj, 
            label=0 if self.supervised else None, 
            n_proc=n_proc
        )
        for i_obj in furthest:
            if i_obj not in forbidden:
                break

        furthest = self.furthest(
            i_obj, 
            label=1 if self.supervised else None, 
            n_proc=n_proc
        )
        for j_obj in furthest:
            if j_obj not in forbidden:
                break

        return i_obj, j_obj
        
        
    def _init_hdf5(self, path, overwrite=False):
        '''
        Initializes the HDF5 backend.

        Parameters
        ----------
        path : str, pathlib.Path
            Path to store model. Open as read-only if it already
            exists; as read/write otherwise.
        overwrite : bool, optional
            Overwrite pre-existing model if one exists. The default is False.

        Returns
        -------
        None.

        '''
        path = pathlib.Path(path)
        if path.exists() and not overwrite:
            raise OSError(
                f'File already exists at {path}. Move this file first or '
                 'specify overwrite=True.'
            )
        self._hdf5 = h5py.File(path, mode='w')
        self._hdf5.attrs['n_dim'] = self.n_dim
        
        
    def close(self):
        '''
        Close the HDF5 backend.

        Returns
        -------
        None.

        '''
        self.hdf5.close()
        
        
    def cupy(self):
        '''
        Use the cupy module to accelerate array calculations using GPU.

        Returns
        -------
        None.

        '''
        global _xp_
        global get_array
        global get_array_module
        try:
            import cupy as _xp_
            get_array_module = lambda array: _xp_.get_array_module(array)
            get_array = lambda array: array.get()
        except ModuleNotFoundError:
            print('CuPy module is not installed. Falling back to NumPy.')

        

    def distance_matrix(
        self, 
        i_objs, 
        j_objs, 
        X_1=None, 
        X_2=None, 
        W_1=None,
        W_2=None,
        batch_size=DEFAULT_BATCH_SIZE
    ):
        # """
        # Return the distance between objects at indices i_objs and kernel object at
        # index ikernel on the ihyprpln^th hyperplane.

        # Arguments:
        # - iobj: int
        #     Index of first object to consider.
        # - jobj: int
        #     Index of second object to consider.

        # Keyword arguments:
        # - ihyprpln: int=0
        #     Index of hyperplane on which to compute distance.
        # """

        if X_1 is None:
            X_1 = self.X
        if X_2 is None:
            X_2 = self.X
        if W_1 is None:
            W_1 = self.W
        if W_2 is None:
            W_2 = self.W

        X_j = _xp_.array(X_2[j_objs])
        dW  = _xp_.square(_xp_.array(W_1[i_objs] - W_2[j_objs]))

        dist = _xp_.concatenate([
            self._distance_func(
                X_1[i_objs[i: i+batch_size]],
                X_j
            )
            for i in range(0, len(i_objs), batch_size)
        ])

        for i in range(self._ihyprpln):
            dist = _xp_.sqrt(_xp_.clip(dist**2 - dW[:, i], 0,  _xp_.inf))

        return dist
            
            
    def furthest(self, i_obj, label=None, n_proc=None):
        """
        Return the index of the object furthest from object with index
        *i_obj*.
        """

        if label is None:
            idxs = np.arange(self.n_obj)
        else:
            idxs = np.argwhere(self.y == label).flatten()

        dW = _xp_.square(_xp_.array(self.W[idxs] - self.W[[i_obj]]))
        dist = self._distance_func(
            self.X[idxs],
            self.X[[i_obj]]
        )
        for i in range(self._ihyprpln):
            dist = _xp_.sqrt(_xp_.clip(dist**2 - dW[:, i], 0, _xp_.inf))
        
        idxs = idxs[get_array(_xp_.argsort(dist))]
        return idxs[-1::-1]


    def train(
        self, 
        X, 
        y=None, 
        n_proc=None, 
        batch_size=DEFAULT_BATCH_SIZE,
        show_progress=True
    ):
        f'''
        Train the FastMap embedding using the input X, y data.

        Parameters
        ----------
        X : numpy.array or cupy.array
            Objects to embed. These objects must be represented as an
            n-D array.
        y : array-like, optional
            Binary Class labels for supervised mode. The default is None.
        n_proc : int, optional
            Number of processes to use if running multiprocessing mode.
            The default is None.
        batch_size : int, optional
            Batch size for batch processing. The default is 
            {DEFAULT_BATCH_SIZE}.
        show_progress : bool, optional
            Show TQDM progress bar. The default is True.

        Returns
        -------
        None.

        '''
        self.X = X
        self.y = y
            

        for self._ihyprpln in tqdm.tqdm(range(self.n_dim)):
            i_piv, j_piv = self._choose_pivots(n_proc=n_proc)
            self.pivot_ids[self._ihyprpln] = [i_piv, j_piv]
            self.X_piv[self._ihyprpln, 0] = self.X[i_piv]
            self.X_piv[self._ihyprpln, 1] = self.X[j_piv]
            
            d_ij = self.distance_matrix([i_piv], [j_piv], batch_size=batch_size)
            d  = _xp_.square(self.distance_matrix(np.arange(self.n_obj), i_piv))
            d -= _xp_.square(self.distance_matrix(np.arange(self.n_obj), j_piv))
            # d = d.get()
            d += d_ij ** 2
            ####### Avoid divide by zero.
            d /= (2 * d_ij + EPSILON)
            #### Hack for negative distances.
            d = _xp_.clip(d, 0, _xp_.inf)
            ####
            self.W[:, self._ihyprpln] = get_array(d)

        for i_dim, (i_piv, j_piv) in enumerate(self.pivot_ids):
            self.W_piv[i_dim, 0] = self.W[i_piv]
            self.W_piv[i_dim, 1] = self.W[j_piv]

        
def test():

    data_path = '/home/malcolmw/git/FastMap/data/ridgecrest.hdf5'
    with h5py.File(data_path, mode='r') as in_file:
        X = in_file['/X/train'][:]
        y = in_file['/y/train'][:]
    
    with FastMap(
        correlation_distance, 
        2,
        '/home/malcolmw/scratch/fastmap.hdf5', 
        overwrite=True
    ) as fm:
        fm.train(X)
        W = fm.W
    
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    ax.scatter(W[:, 0], W[:, 1])
    # for i in range(2):
    #     idxs = np.argwhere(y == i).flatten()
    #     ax.scatter(
    #         W[idxs, 0],
    #         W[idxs, 1]
    #     )
    plt.show()
    
    
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

        
if __name__ == '__main__':
    test()