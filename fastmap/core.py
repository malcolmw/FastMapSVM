#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 16 14:24:10 2023

@author: malcolmw
"""

print('Importing fastmap.core')

# _xp_ is the array module to use for batch distance calculations.
# It can be either numpy or cupy. The default is numpy.
global _xp_
global get_array

import numpy as np
import tqdm

_xp_ = np
get_array = lambda array: array

DEFAULT_BATCH_SIZE = 1024
EPSILON = 1e-9

class FastMapABC:

    def __init__(
        self, 
        distance_func, 
        n_dim,
        show_progress=False,
        batch_size=DEFAULT_BATCH_SIZE
    ):
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
        show_progress: bool, optional
            Show TQDM progress bar. The default is False.

        Returns
        -------
        None.

        '''
        self._distance_func = distance_func
        self._ihyprpln = 0
        self._n_dim = n_dim
        self._batch_size = batch_size
        self.show_progress = show_progress
        
        
    @property
    def batch_size(self):
        '''
        Returns
        -------
        int
            Batch size.

        '''
        return self._batch_size

    @batch_size.setter
    def batch_size(self, value):
        if not isinstance(value, int):
            raise TypeError('batch_size must be an int.')
        self._batch_size = value
        
        
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
        if not hasattr(self, '_pivot_ids'):
            self._pivot_ids = np.full(
                (self.n_dim, 2),
                np.nan,
                dtype=np.uint16,
            )
        return self._pivot_ids
    
    
    @property
    def show_progress(self):
        return self._show_progress
    
    @show_progress.setter
    def show_progress(self, value):
        if value not in (True, False):
            raise(ValueError('show_progress must be either True or False.'))
        self._show_progress = value
    
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
    def W_piv(self):
        if not hasattr(self, '_W_piv'):
            self._W_piv = np.full(
                (self.n_dim, 2, self.n_dim),
                np.nan,
                dtype=np.float32
            )
        return self._W_piv

    
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
        if not hasattr(self, '_X_piv'):
            self._X_piv = np.full(
                (self.n_dim, 2, *self.X.shape[1:]),
                np.nan,
                dtype=self.X.dtype
            )
        return self._X_piv
        
        
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

        
        
    def cupy(self):
        '''
        Use the cupy module to accelerate array calculations using GPU.

        Returns
        -------
        None.

        '''
        global _xp_
        global get_array
        try:
            import cupy as _xp_
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
        W_2=None
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
                X_1[i_objs[i: i+self.batch_size]],
                X_j
            )
            for i in range(0, len(i_objs), self.batch_size)
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


    def fit(
        self, 
        X, 
        y=None, 
        n_proc=None
    ):
        '''
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

        Returns
        -------
        None.

        '''
        self.X = X
        self.y = y
        
        self.W = np.full(
            (self.n_obj, self.n_dim), 
            np.nan,
            dtype=np.float32
        )
            
        wrapper = tqdm.tqdm if self.show_progress is True else lambda x: x
        for self._ihyprpln in wrapper(range(self.n_dim)):
            i_piv, j_piv = self._choose_pivots(n_proc=n_proc)
            self.pivot_ids[self._ihyprpln] = [i_piv, j_piv]
            self.X_piv[self._ihyprpln, 0] = self.X[i_piv]
            self.X_piv[self._ihyprpln, 1] = self.X[j_piv]
            
            d_ij = self.distance_matrix([i_piv], [j_piv])
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
            
        del(self._pivot_ids, self._X, self.W)
        if hasattr(self, '_y'):
            del(self._y)

        self._ihyprpln = 0
        return self


    def transform(self, X):
        """
        Return the embedding (images) of the given objects, `X`.
        """

        n_obj = len(X)

        W = _xp_.zeros((n_obj, self.n_dim), dtype=(_xp_).float32)
        X_piv = _xp_.array(self.X_piv[:])
        W_piv = _xp_.array(self.W_piv[:])
        wrapper = tqdm.tqdm if self.show_progress is True else lambda x: x
        for i_batch, i_start in enumerate(wrapper(range(
                0,
                n_obj,
                self.batch_size
        ))):
            X_batch = _xp_.array(X[i_start: i_start+self.batch_size])
            W_batch = W[i_start: i_start+self.batch_size]
            d_ij0 = self._distance_func(X_piv[:, [0]], X_piv[:, [1]])
            d_k0 = self._distance_func(
                X_batch[:, _xp_.newaxis, _xp_.newaxis],
                X_piv[_xp_.newaxis]
            )
            for self._ihyprpln in range(self.n_dim):
                dW_ij = _xp_.square(W_piv[self._ihyprpln, [0]] - W_piv[self._ihyprpln, 1])
                dW_ik = _xp_.square(W_batch - W_piv[self._ihyprpln, 0])
                dW_jk = _xp_.square(W_batch - W_piv[self._ihyprpln, 1])
                d_ij = d_ij0[self._ihyprpln].copy()
                d_ik = d_k0[:, self._ihyprpln, 0].copy()
                d_jk = d_k0[:, self._ihyprpln, 1].copy()
                for i in range(self._ihyprpln):
                    d_ij = _xp_.sqrt(_xp_.clip(d_ij**2 - dW_ij[:, i], 0, _xp_.inf))
                    d_ik = _xp_.sqrt(_xp_.clip(d_ik**2 - dW_ik[:, i], 0, _xp_.inf))
                    d_jk = _xp_.sqrt(_xp_.clip(d_jk**2 - dW_jk[:, i], 0, _xp_.inf))
                W_batch[:, self._ihyprpln]  = _xp_.square(d_ik)
                W_batch[:, self._ihyprpln] += _xp_.square(d_ij)
                W_batch[:, self._ihyprpln] -= _xp_.square(d_jk)
                W_batch[:, self._ihyprpln] /= (d_ij * 2 + EPSILON)
                W[i_start: i_start+self.batch_size, self._ihyprpln]  =  W_batch[:, self._ihyprpln]

        return get_array(W)

        
if __name__ == '__main__':
    pass