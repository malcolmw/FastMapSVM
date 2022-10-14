import cupy
import h5py
import itertools
import multiprocessing as mp
import numpy as np
import pandas as pd
import pathlib
import pickle
import scipy.signal
import sklearn.model_selection
import sklearn.svm
import tqdm as tqdm
import types

# A small value to avoid divide by zero.
EPSILON = 1e-9

print('Importing fastmapsvm.core')

def _init_pdist_obj(_distance, _ihyprpln):

    global distance, ihyprpln

    distance = _distance
    ihyprpln = _ihyprpln

def _init_pdist_hdf5(_distance, _ihyprpln, _X1, _X2, _W1, _W2):

    global distance, ihyprpln, X1, X2, W1, W2

    distance = _distance
    ihyprpln = _ihyprpln
    X1 = _X1
    X2 = _X2
    W1 = _W1
    W2 = _W2


def _pdist_obj(args):

    global distance, ihyprpln
    Xi, Xj, Wi, Wj = args

    dist = distance(Xi, Xj)
    for i in range(ihyprpln):
        if dist**2 < (Wi[i] - Wj[i])**2:
            return (0)
        dist = np.sqrt(dist**2 - (Wi[i] - Wj[i])**2)

    return (dist)

def _pdist_hdf5(args):

    global distance, ihyprpln, X1, X2, W1, W2

    iobj, jobj = args
    Xi, Xj = X1[iobj], X2[jobj]
    Wi, Wj = W1[iobj], W2[jobj]

    dist = distance(Xi, Xj)
    for i in range(ihyprpln):
        if dist**2 < (Wi[i] - Wj[i])**2:
            return (0)
        dist = np.sqrt(dist**2 - (Wi[i] - Wj[i])**2)

    return dist

def _pembed_hdf5_init(_X, _X_piv, _W_piv, _distance, _preprocess):
    global X, X_piv, W_piv, distance, preprocess
    X = _X
    X_piv = _X_piv
    W_piv = _W_piv
    distance = _distance
    preprocess = _preprocess
    if preprocess is not None:
        X_piv = preprocess(X_piv)


def _pembed_hdf5(kobj):
    """
    Return the embedding (images) of the given objects, `X`.
    """
    global X, X_piv, W_piv, distance, preprocess

    x = X[kobj]
    if preprocess is not None:
        x = preprocess(x)
    ndim = W_piv.shape[0]
    W = np.zeros((ndim,), dtype=np.float32)

    for ihyprpln in range(ndim):
        X_piv_0 = X_piv[ihyprpln, 0]
        X_piv_1 = X_piv[ihyprpln, 1]
        W_piv_0 = W_piv[ihyprpln, 0]
        W_piv_1 = W_piv[ihyprpln, 1]

        d_ij = distance(X_piv_0, X_piv_1)
        d_ik = distance(X_piv_0, x)
        d_jk = distance(X_piv_1, x)

        for jhyprpln in range(ihyprpln):

            dW_ij = np.clip((W_piv_0[jhyprpln] - W_piv_1[jhyprpln])**2, 0, np.inf)
            d_ij = np.sqrt(d_ij**2 - dW_ij)

            dW_ik = np.clip((W_piv_0[jhyprpln] - W[jhyprpln])**2, 0, np.inf)
            d_ik = np.sqrt(d_ik**2 - dW_ik)

            dW_jk = np.clip((W_piv_1[jhyprpln] - W[jhyprpln])**2, 0, np.inf)
            d_jk = np.sqrt(d_jk**2 - dW_jk)

        W[ihyprpln]  = np.square(d_ik)
        W[ihyprpln] += np.square(d_ij)
        W[ihyprpln] -= np.square(d_jk)
        W[ihyprpln] /= (d_ij * 2 + EPSILON)

    return W


class FastMapSVM(object):


    def __init__(self, distance, ndim, model_path, batch_size=None):
        self._batch_size = batch_size
        self._distance = distance
        self._ihyprpln = 0
        self._ndim = ndim
        self._init_hdf5(pathlib.Path(model_path))

    @property
    def batch_size(self):
        return self._batch_size

    @batch_size.setter
    def batch_size(self, value):
        self._batch_size = value

    @property
    def hdf5(self):
        """
        HDF5 model backend.
        """

        return (self._hdf5)

    @property
    def ndim(self):
        """
        Dimensionality of embedding.
        """
        return (self._ndim)


    @property
    def X_piv(self):
        """
        Pivot objects.
        """

        if "X_piv" not in self.hdf5:
            self.hdf5.create_dataset(
                "X_piv",
                (self.ndim, 2, *self.X.shape[1:]),
                self.X.dtype,
                fillvalue=np.nan
            )

        return (self.hdf5["X_piv"])

    @property
    def pivot_ids(self):
        """
        Indices of pivot objects.
        """

        if "pivot_ids" not in self.hdf5:
            self.hdf5.create_dataset(
                "pivot_ids",
                (self.ndim, 2),
                np.uint16,
                fillvalue=np.nan
            )

        return (self.hdf5["pivot_ids"])


    @property
    def W_piv(self):

        if "W_piv" not in self.hdf5:
            self.hdf5.require_dataset(
                "W_piv",
                (self.ndim, 2, self.ndim),
                np.float32,
                exact=True,
                fillvalue=np.nan
            )

        return (self.hdf5["W_piv"])

    @property
    def W(self):

        if "W" not in self.hdf5:
            self.hdf5.require_dataset(
                "W",
                (self.X.shape[0], self.ndim),
                np.float32,
                exact=True,
                fillvalue=np.nan
            )

        return (self.hdf5["W"])


    @property
    def X(self):

        return (self._X)

    @X.setter
    def X(self, value):

        self._X = value


    @property
    def y(self):

        return (self.hdf5["y"])

    @y.setter
    def y(self, value):
        if "y" not in self.hdf5:
            self.hdf5.create_dataset("y", data=value)
        else:
            raise (AttributeError("Attribute already initialized."))


    def _choose_pivots(self, nproc=None):
        """
        A heuristic algorithm to choose distant pivot objects adapted
        from Faloutsos and Lin (1995).
        """

        forbidden = self.pivot_ids[:self._ihyprpln].flatten()

        while True:
            jobj = np.random.choice(np.argwhere(self.y[:] == 1).flatten())
            if jobj not in forbidden:
                break

        furthest = self.furthest(jobj, label=0, nproc=nproc)
        for iobj in furthest:
            if iobj not in forbidden:
                break

        furthest = self.furthest(iobj, label=1, nproc=nproc)
        for jobj in furthest:
            if jobj not in forbidden:
                break

        return iobj, jobj


    def _choose_pivots_closest(self, nproc=None):
        """
        A heuristic algorithm to choose distant pivot objects adapted
        from Faloutsos and Lin (1995).
        """

        forbidden = self.pivot_ids[:self._ihyprpln].flatten()
        if self._ihyprpln > 0:
            iobj = self.pivot_ids[self._ihyprpln-1, 0]
        else:
            iobj = np.random.choice(np.argwhere(self.y[:] == 0).flatten())

        closest = self.closest(iobj, label=1, nproc=nproc)
        for jobj in closest:
            if jobj not in forbidden:
                break

        furthest = self.furthest(jobj, label=0, nproc=nproc)
        for iobj in furthest:
            if iobj not in forbidden:
                break

        return iobj, jobj

    def _choose_pivots_furthest(self, nproc=None):
        """
        A heuristic algorithm to choose distant pivot objects adapted
        from Faloutsos and Lin (1995).
        """

        forbidden = self.pivot_ids[:self._ihyprpln].flatten()

        if self._ihyprpln == 0:
            # Get a random seed object if this is the first dimension.
            iobj = np.random.choice(np.argwhere(self.y[:] == 1).flatten())
        else:
            # Othewise, get the pivot object for the positive class from
            # the previous dimension as seed.
            iobj = self.pivot_ids[self._ihyprpln-1, 1]

        # Find the positive-class object that is furthest from the seed
        # object.
        furthest = self.furthest(iobj, label=1, nproc=nproc)
        for jobj in furthest:
            if jobj not in forbidden:
                break

        # Find the negative-class object that is furthest from the
        # object discovered above.
        furthest = self.furthest(jobj, label=0, nproc=nproc)
        for iobj in furthest:
            if iobj not in forbidden:
                break

        return iobj, jobj


    def _init_hdf5(self, path):
        """
        Initialize the HDF5 backend to store pivot objects and images
        of training data.

        Arguments:
        - path: pathlib.Path
            The path to the backend. Open as read-only if it already;
            exists; as read/write otherwise.
        """

        self._hdf5 = h5py.File(path, mode="a")
        self._hdf5.attrs["ndim"] = self.ndim

        return (True)


    def distance(self, iobj, jobj, X1=None, X2=None, W1=None, W2=None):
        """
        Return the distance between object at index iobj and object at
        index jobj on the ihyprpln^th hyperplane.

        Arguments:
        - iobj: int
            Index of first object to consider.
        - jobj: int
            Index of second object to consider.

        Keyword arguments:
        - ihyprpln: int=0
            Index of hyperplane on which to compute distance.
        """

        if X1 is None:
            X1 = self.X
        if X2 is None:
            X2 = self.X
        if W1 is None:
            W1 = self.W
        if W2 is None:
            W2 = self.W

        dist = self._distance(X1[iobj], X2[jobj])

        for i in range(self._ihyprpln):
            if dist**2 < (W1[iobj, i] - W2[jobj, i])**2:
                return (0)

            dist = np.sqrt(dist**2 - (W1[iobj, i] - W2[jobj, i])**2)

        return dist

    def _batch_distance(self, iobjs, ikernel, X1=None, X2=None, W1=None, W2=None):
        """
        Return the distance between objects at indices iobjs and kernel object at
        index ikernel on the ihyprpln^th hyperplane.

        Arguments:
        - iobj: int
            Index of first object to consider.
        - jobj: int
            Index of second object to consider.

        Keyword arguments:
        - ihyprpln: int=0
            Index of hyperplane on which to compute distance.
        """

        if X1 is None:
            X1 = self.X
        if X2 is None:
            X2 = self.X
        if W1 is None:
            W1 = self.W
        if W2 is None:
            W2 = self.W


        dist = cupy.zeros(len(iobjs), dtype=cupy.float32)
        dW = cupy.square(cupy.array(W1[iobjs] - W2[ikernel]))
        x2 = cupy.array(X2[ikernel])

        for istart in range(0, len(iobjs), self.batch_size):
            x1 = cupy.array(X1[iobjs[istart: istart+self.batch_size]])
            n = x1.shape[0]

            #dist[istart: istart+n] = self._distance(
            #    cupy.array(X1[iobjs[istart: istart+self.batch_size]]),
            #    cupy.array(X2[ikernel])
            #)
            dist[istart: istart+n] = self._distance(x1, x2)

            for ihyprpln in range(self._ihyprpln):
                dist[istart: istart+n] = cupy.sqrt(cupy.clip(
                    cupy.square(dist[istart: istart+n]) - dW[istart: istart+n, ihyprpln],
                    0,
                    cupy.inf
                ))

        return dist


    def batch_distance(self, iobjs, ikernel, X1=None, X2=None, W1=None, W2=None, verbose=False):
        """
        Return the distance between objects at indices iobjs and kernel object at
        index ikernel on the ihyprpln^th hyperplane.

        Arguments:
        - iobj: int
            Index of first object to consider.
        - jobj: int
            Index of second object to consider.

        Keyword arguments:
        - ihyprpln: int=0
            Index of hyperplane on which to compute distance.
        """

        if X1 is None:
            X1 = self.X
        if X2 is None:
            X2 = self.X
        if W1 is None:
            W1 = self.W
        if W2 is None:
            W2 = self.W

        X_kernel = cupy.array(X2[ikernel])
        dW = cupy.square(cupy.array(W1[iobjs] - W2[ikernel]))


        dist = cupy.concatenate([
            self._distance(
                cupy.array(X1[iobjs[i: i+self.batch_size]]),
                X_kernel
            )
            for i in range(0, len(iobjs), self.batch_size)
        ])


        if verbose is True:
            print(
                f'batch_distance(0, {self._ihyprpln}):',
                W1[0]
            )

        for i in range(self._ihyprpln):
            #if verbose is True:
            #    print(
            #        f'batch_distance(0, {self._ihyprpln}, {i}):',
            #        dist,
            #    )
            dist = cupy.sqrt(cupy.clip(dist**2 - dW[:, i], 0, cupy.inf))



        return dist

    def distance_matrix(self, iobjs, jobjs, X1=None, X2=None, W1=None, W2=None, verbose=False):
        """
        Return the distance between objects at indices iobjs and kernel object at
        index ikernel on the ihyprpln^th hyperplane.

        Arguments:
        - iobj: int
            Index of first object to consider.
        - jobj: int
            Index of second object to consider.

        Keyword arguments:
        - ihyprpln: int=0
            Index of hyperplane on which to compute distance.
        """

        if X1 is None:
            X1 = self.X
        if X2 is None:
            X2 = self.X
        if W1 is None:
            W1 = self.W
        if W2 is None:
            W2 = self.W

        Xj = cupy.array(X2[jobjs])
        dW = cupy.square(cupy.array(W1[iobjs] - W2[jobjs]))

        dist = cupy.concatenate([
            self._distance(
                cupy.array(X1[iobjs[i: i+self.batch_size]]),
                Xj
            )
            for i in range(0, len(iobjs), self.batch_size)
        ])

        for i in range(self._ihyprpln):
            dist = cupy.sqrt(cupy.clip(dist**2 - dW[:, i], 0, cupy.inf))

        return dist


    def embed_dep(self, X, nproc=None):
        """
        Return the embedding (images) of the given objects, `X`.
        """

        nobj = X.shape[0]

        if self.batch_size is not None:
            W = cupy.zeros((nobj, self.ndim), dtype=cupy.float32)
            X_piv = cupy.array(self.X_piv[:])
            W_piv = cupy.array(self.W_piv[:])
            for ibatch, istart in enumerate(tqdm.tqdm(range(
                    0,
                    nobj,
                    self.batch_size
            ))):
                X_batch = cupy.array(X[istart: istart+self.batch_size])
                W_batch = W[istart: istart+self.batch_size]
                for self._ihyprpln in range(self.ndim):
                    d_ij = self._distance(
                        X_piv[self._ihyprpln, [0]],
                        X_piv[self._ihyprpln, 1]
                    )
                    d_ik = self._distance(X_batch, X_piv[self._ihyprpln, 0])
                    d_jk = self._distance(X_batch, X_piv[self._ihyprpln, 1])
                    dW_ij = cupy.square(W_piv[self._ihyprpln, [0]] - W_piv[self._ihyprpln, 1])
                    dW_ik = cupy.square(W_batch - W_piv[self._ihyprpln, 0])
                    dW_jk = cupy.square(W_batch - W_piv[self._ihyprpln, 1])
                    for i in range(self._ihyprpln):
                        d_ij = cupy.sqrt(cupy.clip(d_ij**2 - dW_ij[:, i], 0, cupy.inf))
                        d_ik = cupy.sqrt(cupy.clip(d_ik**2 - dW_ik[:, i], 0, cupy.inf))
                        d_jk = cupy.sqrt(cupy.clip(d_jk**2 - dW_jk[:, i], 0, cupy.inf))
                    W_batch[:, self._ihyprpln]  = cupy.square(d_ik)
                    W_batch[:, self._ihyprpln] += cupy.square(d_ij)
                    W_batch[:, self._ihyprpln] -= cupy.square(d_jk)
                    W_batch[:, self._ihyprpln] /= (d_ij * 2 + EPSILON)
                    W[istart: istart+self.batch_size, self._ihyprpln]  =  W_batch[:, self._ihyprpln]
            W = W.get()

        else:
            W = np.zeros((nobj, self.ndim), dtype=np.float32)
            for self._ihyprpln in tqdm.tqdm(range(self.ndim)):

                Xpiv = self.X_piv[self._ihyprpln]
                Wpiv = self.W_piv[self._ihyprpln]

                d_ij = self.distance(0, 1, X1=Xpiv, X2=Xpiv, W1=Wpiv, W2=Wpiv)
                d_ik = self.pdist(0, kobj, X1=Xpiv, X2=X, W1=Wpiv, W2=W, nproc=nproc)
                d_jk = self.pdist(1, kobj, X1=Xpiv, X2=X, W1=Wpiv, W2=W, nproc=nproc)

                W[:, self._ihyprpln]  = np.square(d_ik)
                W[:, self._ihyprpln] += np.square(d_ij)
                W[:, self._ihyprpln] -= np.square(d_jk)
                W[:, self._ihyprpln] /= (d_ij * 2 + EPSILON)

        return W


    def embed(self, X, nproc=None):
        """
        Return the embedding (images) of the given objects, `X`.
        """

        nobj = X.shape[0]

        if self.batch_size is not None:
            W = cupy.zeros((nobj, self.ndim), dtype=cupy.float32)
            X_piv = cupy.array(self.X_piv[:])
            W_piv = cupy.array(self.W_piv[:])
            for ibatch, istart in enumerate(tqdm.tqdm(range(
                    0,
                    nobj,
                    self.batch_size
            ))):
                X_batch = cupy.array(X[istart: istart+self.batch_size])
                W_batch = W[istart: istart+self.batch_size]
                d_ij0 = self._distance(X_piv[:, [0]], X_piv[:, [1]])
                d_k0 = self._distance(
                    X_batch[:, cupy.newaxis, cupy.newaxis],
                    X_piv[cupy.newaxis]
                )
                for self._ihyprpln in range(self.ndim):
                    dW_ij = cupy.square(W_piv[self._ihyprpln, [0]] - W_piv[self._ihyprpln, 1])
                    dW_ik = cupy.square(W_batch - W_piv[self._ihyprpln, 0])
                    dW_jk = cupy.square(W_batch - W_piv[self._ihyprpln, 1])
                    d_ij = d_ij0[self._ihyprpln].copy()
                    d_ik = d_k0[:, self._ihyprpln, 0].copy()
                    d_jk = d_k0[:, self._ihyprpln, 1].copy()
                    for i in range(self._ihyprpln):
                        d_ij = cupy.sqrt(cupy.clip(d_ij**2 - dW_ij[:, i], 0, cupy.inf))
                        d_ik = cupy.sqrt(cupy.clip(d_ik**2 - dW_ik[:, i], 0, cupy.inf))
                        d_jk = cupy.sqrt(cupy.clip(d_jk**2 - dW_jk[:, i], 0, cupy.inf))
                    W_batch[:, self._ihyprpln]  = cupy.square(d_ik)
                    W_batch[:, self._ihyprpln] += cupy.square(d_ij)
                    W_batch[:, self._ihyprpln] -= cupy.square(d_jk)
                    W_batch[:, self._ihyprpln] /= (d_ij * 2 + EPSILON)
                    W[istart: istart+self.batch_size, self._ihyprpln]  =  W_batch[:, self._ihyprpln]
            W = W.get()

        else:
            W = np.zeros((nobj, self.ndim), dtype=np.float32)
            for self._ihyprpln in tqdm.tqdm(range(self.ndim)):

                Xpiv = self.X_piv[self._ihyprpln]
                Wpiv = self.W_piv[self._ihyprpln]

                d_ij = self.distance(0, 1, X1=Xpiv, X2=Xpiv, W1=Wpiv, W2=Wpiv)
                d_ik = self.pdist(0, kobj, X1=Xpiv, X2=X, W1=Wpiv, W2=W, nproc=nproc)
                d_jk = self.pdist(1, kobj, X1=Xpiv, X2=X, W1=Wpiv, W2=W, nproc=nproc)

                W[:, self._ihyprpln]  = np.square(d_ik)
                W[:, self._ihyprpln] += np.square(d_ij)
                W[:, self._ihyprpln] -= np.square(d_jk)
                W[:, self._ihyprpln] /= (d_ij * 2 + EPSILON)

        return W



    def embed_database(self, nproc=None, batch_size=None):
        """
        Compute and store the image of every object in the database.
        """
        n = self.X.shape[0]

        if batch_size is not None:
            assert self.batch_mode is True
            if nproc is not None:
                print('Classifier is in batch mode.')

        for self._ihyprpln in tqdm.tqdm(range(self.ndim)):

            ipiv, jpiv = self._choose_pivots(nproc=nproc)
            self.pivot_ids[self._ihyprpln] = [ipiv, jpiv]
            self.X_piv[self._ihyprpln, 0] = self.X[ipiv]
            self.X_piv[self._ihyprpln, 1] = self.X[jpiv]
            if self.batch_size is not None:
                d_ij = self.distance_matrix([ipiv], [jpiv]).get()

                d  = cupy.square(self.distance_matrix(np.arange(n), ipiv))
                d -= cupy.square(self.distance_matrix(np.arange(n), jpiv))
                d = d.get()
            else:
                d_ij = self.distance(ipiv, jpiv)
                d  = np.square(self.pdist(np.arange(n), ipiv, nproc=nproc))
                d -= np.square(self.pdist(np.arange(n), jpiv, nproc=nproc))
            d += d_ij ** 2
            ####### Avoid divide by zero.
            d /= (2 * d_ij + EPSILON)
            #### Hack for negative distances.
            d = np.clip(d, 0, np.inf)
            ####
            self.W[:, self._ihyprpln] = d

        for idim, (ipiv, jpiv) in enumerate(self.pivot_ids):
            self.W_piv[idim, 0] = self.W[ipiv]
            self.W_piv[idim, 1] = self.W[jpiv]

        return (True)

    def embed_database_dep(self, nproc=None, batch_size=None):
        """
        Compute and store the image of every object in the database.
        """

        n = self.X.shape[0]

        if batch_size is not None:
            assert self.batch_mode is True
            if nproc is not None:
                print('Classifier is in batch mode.')

        for self._ihyprpln in tqdm.tqdm(range(self.ndim)):

            ipiv, jpiv = self._choose_pivots(nproc=nproc)
            self.pivot_ids[self._ihyprpln] = [ipiv, jpiv]
            self.X_piv[self._ihyprpln, 0] = self.X[ipiv]
            self.X_piv[self._ihyprpln, 1] = self.X[jpiv]
            if self.batch_size is not None:
                d_ij = self.batch_distance([ipiv], jpiv).get()

                d  = cupy.square(self.batch_distance(np.arange(n), ipiv))
                d -= cupy.square(self.batch_distance(np.arange(n), jpiv))
                d = d.get()
            else:
                d_ij = self.distance(ipiv, jpiv)
                d  = np.square(self.pdist(np.arange(n), ipiv, nproc=nproc))
                d -= np.square(self.pdist(np.arange(n), jpiv, nproc=nproc))
            d += d_ij ** 2
            ####### Avoid divide by zero.
            d /= (2 * d_ij + EPSILON)
            #### Hack for negative distances.
            d = np.clip(d, 0, np.inf)
            ####
            self.W[:, self._ihyprpln] = d

        for idim, (ipiv, jpiv) in enumerate(self.pivot_ids):
            self.W_piv[idim, 0] = self.W[ipiv]
            self.W_piv[idim, 1] = self.W[jpiv]

        return (True)

    def fit(
        self,
        X, y,
        kernel=("linear", "rbf"),
        C=[2**n for n in range(-4, 4)],
        gamma=[2**n for n in range(-4, 4)],
        nproc=None,
        scaler=None
    ):
        self.X = X
        self.y = y
        self.embed_database(nproc=nproc)
        self._scaler = scaler

        params = dict(kernel=kernel, C=C, gamma=gamma)
        svc = sklearn.svm.SVC(probability=False)
        clf = sklearn.model_selection.GridSearchCV(svc, params, n_jobs=nproc)
        W = self.W[:]
        if self._scaler is not None:
            self._scaler.fit(W)
            W = self._scaler.transform(W)

        clf.fit(W, self.y[:])
        self._clf = sklearn.svm.SVC(
            **{**clf.best_estimator_.get_params(), 'probability': True}
        )
        self._clf.fit(W, self.y[:])

        self.hdf5.create_dataset("clf", data=np.void(pickle.dumps(self._clf)))
        self.hdf5.create_dataset("scaler", data=np.void(pickle.dumps(self._scaler)))
        self.hdf5.attrs['batch_size'] = self.batch_size


    def furthest(self, iobj, label=None, nproc=None):
        """
        Return the index of the object furthest from object with index
        *iobj*.
        """

        if label is None:
            idxs = np.arange(self.y.shape[0])
        else:
            idxs = np.argwhere(self.y[:] == label).flatten()

        if self.batch_size is not None:
            dW = cupy.square(cupy.array(self.W[idxs] - self.W[[iobj]]))
            dist = self._distance(
                cupy.array(self.X[idxs]),
                cupy.array(self.X[[iobj]])
            )
            for i in range(self._ihyprpln):
                dist = cupy.sqrt(cupy.clip(dist**2 - dW[:, i], 0, cupy.inf))

            return idxs[cupy.argsort(dist).get()][-1::-1]

        else:
            return idxs[np.argsort(self.pdist(iobj, idxs, nproc=nproc))[-1::-1]]


    def _furthest(self, iobj, label=None, nproc=None):
        """
        Return the index of the object furthest from object with index
        *iobj*.
        """
        if label is None:
            idxs = np.arange(self.y.shape[0])
        else:
            idxs = np.argwhere(self.y[:] == label).flatten()

        if self.batch_size is not None:
            dist = self.batch_distance(
                idxs,
                iobj
            )
            try:
                return idxs[cupy.argsort(dist).get()][-1::-1]
            except:
                print(idxs)
                print(dist)
                raise
        else:
            return idxs[np.argsort(self.pdist(iobj, idxs, nproc=nproc))[-1::-1]]


    def _closest(self, iobj, label=None, nproc=None):
        """
        Return the index of the object closest to object with index
        *iobj*.
        """
        if label is None:
            idxs = np.arange(self.y.shape[0])
        else:
            idxs = np.argwhere(self.y[:] == label).flatten()

        return idxs[np.argsort(self.pdist(iobj, idxs, nproc=nproc))]

    def closest(self, iobj, label=None, nproc=None):
        """
        Return the index of the object furthest from object with index
        *iobj*.
        """

        if label is None:
            idxs = np.arange(self.y.shape[0])
        else:
            idxs = np.argwhere(self.y[:] == label).flatten()

        if self.batch_size is not None:
            dW = cupy.square(cupy.array(self.W[idxs] - self.W[[iobj]]))
            dist = self._distance(
                cupy.array(self.X[idxs]),
                cupy.array(self.X[[iobj]])
            )
            for i in range(self._ihyprpln):
                dist = cupy.sqrt(cupy.clip(dist**2 - dW[:, i], 0, cupy.inf))

            return idxs[cupy.argsort(dist).get()]

        else:
            return idxs[np.argsort(self.pdist(iobj, idxs, nproc=nproc))[-1::-1]]


    def load(path, distance):
        self = FastMapSVM.__new__(FastMapSVM)
        self._hdf5 = h5py.File(path, mode="a")
        self._distance = distance
        self._ihyprpln = 0
        self._ndim = self.hdf5.attrs["ndim"]

        self._clf = pickle.loads(np.void(self.hdf5["clf"]))
        self._scaler = pickle.loads(np.void(self.hdf5["scaler"]))
        self.batch_size = self.hdf5.attrs['batch_size']

        return (self)



    def pdist(self, iobj, jobj, X1=None, X2=None, W1=None, W2=None, nproc=None):

        iobj = np.atleast_1d(iobj)
        jobj = np.atleast_1d(jobj)

        if X1 is None:
            X1 = self.X
        if X2 is None:
            X2 = self.X
        if W1 is None:
            W1 = self.W
        if W2 is None:
            W2 = self.W

        if nproc in (1, None):
            # Forgo the overhead of start multiple processes only
            # one core is being used.
            return np.array(list(
                self.distance(iobj, jobj, X1=X1, X2=X2, W1=W1, W2=W2)
                for iobj, jobj in itertools.product(iobj, jobj)
            ))

        if any(isinstance(x, h5py.Dataset) for x in (X1, X2, W1, W2)):
            # This is the preferred way of doing things.
            initargs = (self._distance, self._ihyprpln, X1, X2, W1, W2)
            with mp.Pool(
                processes=nproc,
                initializer=_init_pdist_hdf5,
                initargs=initargs
            ) as pool:
                generator = itertools.product(iobj, jobj)
                return np.array(list(pool.imap(_pdist_hdf5, generator)))
        else:
            # This is a workaround for Macs because multiple processes cannot
            # simultaneously have the same HDF5 file open. :|
            initargs= (self._distance, self._ihyprpln)
            with mp.Pool(
                processes=nproc,
                initializer=_init_pdist_obj,
                initargs=initargs
            ) as pool:
                generator = (
                    (X1[iobj], X2[jobj], W1[iobj], W2[jobj])
                    for iobj, jobj in itertools.product(iobj, jobj)
                )
                return np.array(list(pool.imap(_pdist_obj, generator)))


    def predict(self, X, return_image=False, nproc=None):

        W = self.embed(X, nproc=nproc)
        if self._scaler is not None:
            W = self._scaler.transform(W)

        if return_image is True:
            return (W, self._clf.predict(W))
        else:
            return (self._clf.predict(W))


    def predict_proba(self, X, return_image=False, nproc=None):

        W = self.embed(X, nproc=nproc)
        if self._scaler is not None:
            W = self._scaler.transform(W)

        if return_image is True:
            return (W, self._clf.predict_proba(W))
        else:
            return (self._clf.predict_proba(W))
