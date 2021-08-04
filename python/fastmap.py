import h5py
import numpy as np
import pathlib

class FastMap(object):
        
    @property
    def database(self):
        
        return (self._database)
    
    
    @property
    def hdf5(self):
        
        return (self._hdf5)

    
    @property
    def image(self):
        
        if not hasattr(self, "_image"):
            self.hdf5.require_dataset(
                "image", 
                (self.database.shape[0], self.ndim), 
                np.float32, 
                exact=True,
                fillvalue=np.nan
            )
            
        return (self.hdf5["image"])


    @property
    def ndim(self):
    
        return (self._ndim)

    
    @property
    def pivots(self):
        
        if not hasattr(self, "_pivots"):
            self.hdf5.require_dataset(
                "pivots", 
                (self.ndim, 2, *self.database.shape[1:]), 
                self.database.dtype,
                exact=True,
                fillvalue=np.nan
            )
            
        return (self.hdf5["pivots"])


    def __init__(self, database, distance, ndim, path):
        self._database = database
        self._distance = distance
        self._ihyprpln = 0
        self._ndim = ndim
        self._init_hdf5(pathlib.Path(path))
        
        

    def _choose_pivots(self):
        """
        A heuristic algorithm to choose distant pivot objects 
        (Faloutsos and Lin, 1995).
        
        .. todo: Make sure you don't reuse a pivot object.
        """
        
        jobj = np.random.randint(0, high=self.database.shape[0])
        iobj = self.furthest(jobj)
        jobj = self.furthest(iobj)
        
        return (iobj, jobj)

    
    def _init_hdf5(self, path):
        """
        Initialize the HDF5 backend to store pivot objects.
        
        Arguments:
        - path: pathlib.Path
            The path to the backend. Open as read-only if it already;
            exists; as read/write otherwise.
        """
        if path.exists():
            self._hdf5 = h5py.File(path, mode="r")
        else:
            self._hdf5 = h5py.File(path, mode="w")
            
        return (True)


    def close(self):
        """
        Close the HDF5 backend.
        """
        self.hdf5.close()
        
        return (True)
    
    
    def distance(self, iobj, jobj):
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

        dist = self._distance(self.database[iobj], self.database[jobj])
        
        for i in range(self._ihyprpln):
            dist = np.sqrt(dist**2 - (self.image[iobj, i] - self.image[jobj, i])**2)

        return (dist)
    
    
    def embed_database(self):
        """
        Compute and store the image of every object in the database.
        """
        
        for self._ihyprpln in range(self.ndim):

            ipiv, jpiv = self._choose_pivots()
            self.pivots[self._ihyprpln, 0] = self.database[ipiv]
            self.pivots[self._ihyprpln, 1] = self.database[jpiv]
            d_ij = self.distance(ipiv, jpiv)
            self.image[:, self._ihyprpln] = [
                (
                      self.distance(kobj, ipiv)**2 
                    + d_ij**2 
                    - self.distance(kobj, jpiv)**2
                )  
                /  
                ( 2 * d_ij) 
                for kobj in range(self.database.shape[0])
            ]

        return (True)
    
    def furthest(self, iobj):
        """
        Return the index of the object furthest from object with index 
        *iobj*.
        """
        
        nobj = self.database.shape[0]
        
        return (
            np.argmax([self.distance(iobj, jobj) for jobj in range(nobj)])
        )


def correlate(a, b):
    """
    Return the normalized cross-correlation of a and b.
    """
    
    a = (a - np.mean(a)) / (np.std(a) * len(a))
    b = (b - np.mean(b)) / (np.std(b))
    corr = np.correlate(a, b, "full")
    
    return (corr)


def distance(obj_a, obj_b, force_triangle_ineq=False):
    """
    Return the *metric* distance between object obj_a and object obj_b.
    
    Arguments:
    - obj_a: object
        First object to consider.
    - obj_b: object
        Second object to consider.
    """
    dist = 1 - np.max(np.abs(correlate(obj_a, obj_b)))
    
    if force_triangle_ineq is True:
        if dist == 0:
            return (0)
        else:
            return ((dist + 1) / 2)

    else:
        return (dist)