import h5py
import itertools
import multiprocessing as mp
import numpy as np
import pandas as pd
import pathlib
import scipy.signal
import sklearn.model_selection
import sklearn.svm


_reduce = lambda c: np.mean(c, axis=0)


def _init_pdist(fastmap, _X1, _X2, _W1, _W2):

    global self, X1, X2, W1, W2
    
    self = fastmap
    X1 = _X1
    X2 = _X2
    W1 = _W1
    W2 = _W2

    
def _pdist(iobj, jobj):
    
    global self, X1, X2, W1, W2

    dist = self._distance(X1[iobj], X2[jobj])

    for i in range(self._ihyprpln):
        dist = np.sqrt(dist**2 - (W1[iobj, i] - W2[jobj, i])**2)

    return (dist)


def correlate(a, b, mode="valid"):

    if len(a) > len(b):
        a, b = b, a

    a = pd.Series(a)
    b = pd.Series(b)
    n = len(a)

    a = a - np.mean(a)
    b = b - np.mean(b)
    
    c = scipy.signal.correlate(b, a, mode=mode)
    
    if mode == "valid":
        norm = n * np.std(a) * b.rolling(n).std().dropna().values
    elif mode == "same":
        norm = n * np.std(a) * b.rolling(n, min_periods=0, center=True).std().values
    c /= norm
    
    return (c)


def distance(
    obj_a, 
    obj_b, 
    mode="valid", 
    reduce=_reduce, 
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


def ndcorrelate(a, b, mode="valid", reduce=_reduce):

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
    
    return (reduce(c))


class FastMapSVM(object):
    
    def __init__(self, distance, ndim, model_path):
        self._distance = distance
        self._ihyprpln = 0
        self._ndim = ndim
        self._init_hdf5(pathlib.Path(model_path))
    
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

    
    def _choose_pivots(self):
        """
        A heuristic algorithm to choose distant pivot objects adapted
        from Faloutsos and Lin (1995).
        """
        
        jobj = np.random.choice(np.argwhere(self.y[:] == 1).flatten())
        
        while jobj in self.pivot_ids[:self._ihyprpln].flatten():
            jobj = np.random.choice(np.argwhere(self.y[:] == 1).flatten())

        iobj = self.furthest(jobj, label=0)
        jobj = self.furthest(iobj, label=1)
        
        return (iobj, jobj)
    

    def _init_hdf5(self, path):
        """
        Initialize the HDF5 backend to store pivot objects and images
        of training data.
        
        Arguments:
        - path: pathlib.Path
            The path to the backend. Open as read-only if it already;
            exists; as read/write otherwise.
        """
        
        self._hdf5 = h5py.File(path, mode="w")
            
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
            dist = np.sqrt(dist**2 - (W1[iobj, i] - W2[jobj, i])**2)

        return (dist)


    def embed(self, X):
        """
        Return the embedding (images) of the given objects, `X`.
        """
        
        nobj = X.shape[0]
        kobj = np.arange(nobj)
        W = np.zeros((nobj, self.ndim), dtype=np.float32)
        
        for self._ihyprpln in range(self.ndim):
            
            Xpiv = self.X_piv[self._ihyprpln]
            Wpiv = self.W_piv[self._ihyprpln]
            
            d_ij = self.distance(0, 1, X1=Xpiv, X2=Xpiv, W1=Wpiv, W2=Wpiv)
            d_ik = self.pdist(0, kobj, X1=Xpiv, X2=X, W1=Wpiv, W2=W)
            d_jk = self.pdist(1, kobj, X1=Xpiv, X2=X, W1=Wpiv, W2=W)
            
            W[:, self._ihyprpln]  = np.square(d_ik)
            W[:, self._ihyprpln] += np.square(d_ij)
            W[:, self._ihyprpln] -= np.square(d_jk)
            W[:, self._ihyprpln] /= (d_ij * 2)
            
        return (W)


    def embed_database(self):
        """
        Compute and store the image of every object in the database.
        """
        
        n = self.X.shape[0]
        
        for self._ihyprpln in range(self.ndim):

            ipiv, jpiv = self._choose_pivots()
            self.pivot_ids[self._ihyprpln] = [ipiv, jpiv]
            self.X_piv[self._ihyprpln, 0] = self.X[ipiv]
            self.X_piv[self._ihyprpln, 1] = self.X[jpiv]
            d_ij = self.distance(ipiv, jpiv)
            
            d  = np.square(self.pdist(np.arange(n), ipiv))
            d -= np.square(self.pdist(np.arange(n), jpiv))
            d += d_ij ** 2
            d /= (2 * d_ij)
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
        gamma=[2**n for n in range(-4, 4)]
    ):
        self.X = X
        self.y = y
        self.embed_database()
        
        params = dict(kernel=kernel, C=C, gamma=gamma)
        clf = sklearn.model_selection.GridSearchCV(sklearn.svm.SVC(), params)
        clf.fit(self.W[:], self.y[:])
        self._clf = clf.best_estimator_

    
    def furthest(self, iobj, label=None):
        """
        Return the index of the object furthest from object with index 
        *iobj*.
        """

        if label is None:
            idxs = np.arange(self.y.shape[0])
        else:
            idxs = np.argwhere(self.y[:] == label).flatten()
        
        return (idxs[np.argmax(self.pdist(iobj, idxs))])


    def pdist(self, iobj, jobj, X1=None, X2=None, W1=None, W2=None):

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

        with mp.Pool(initializer=_init_pdist, initargs=(self, X1, X2, W1, W2)) as pool:
            iterator = itertools.product(iobj, jobj)
                      
            return (np.array(pool.starmap(_pdist, iterator)))


    def predict(self, X, return_image=False):
        
        W = self.embed(X)
        
        if return_image is True:
            return (W, self._clf.predict(W))
        else:
            return (self._clf.predict(W))