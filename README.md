# FastMapSVM: An Algorithm for Classifying Complex Objects
This is the official repository for the FastMapSVM algorithm associated with **"FastMapSVM: Classifying Complex Objects Using the FastMap algorithm and Support-Vector Machines"** (White et al., in review). The pre-print is available on arXiv: https://arxiv.org/abs/2204.05112.

**Disclaimer**: The FastMapSVM concept was first presented in an underappreciated paper by Ban et al., (2009). We independently re-invented the framework and subsequently discovered the work of Ban et al. We are now in the process of revising our manuscript to omit any unmerited claims of novelty.

![Perspicuous Visualization](resources/readme_figure.png)

# Installation
```bash
>$ pip install .
```

# Quick Start
```python
import fastmapsvm as fm
import h5py
import numpy as np
import sklearn.metrics

model_path = "data/ridgecrest_model.hdf5" # Model output path.
data_path  = "data/ridgecrest.hdf5"       # Test/training data path.

# Define a distance function on pairs of objects.
def correlation_distance(a, b):
    """
    Returns the correlation distance between objects a and b.
    """
    return (
        1 - np.max(np.abs(np.mean(fm.distance.ndcorrelate(a, b, mode="same"), axis=0)))
    )

with h5py.File(data_path, mode="r") as f5:
    # Instantiate the model.
    clf = fm.FastMapSVM(
        correlation_distance, # Distance function.
        8,                    # Number of Euclidean dimensions.
        model_path            # Model output path.
    )
    # Train the model.
    clf.fit(
        f5["/X/train"], # Model training inputs.
        f5["/y/train"]  # Target training labels.
    )
    # Note: Model training inputs can be either an HDF5 DataSet OR a NumPy array
    # with an arbitrary number of dimensions. Objects in the data set should be
    # indexed along the first axis.
    
    # Test the model.
    y_true = f5["/y/test"][:] # Target test labels.
    proba = clf.predict_proba(
        f5["/X/test"] # Model test inputs.
    )
    
# Plot confusion matrix.
sklearn.metrics.ConfusionMatrixDisplay.from_predictions(
    y_true, 
    proba[:, 1] > 0.5
)
```
![Confusion Matrix](resources/confusion_matrix.png)

# Limitations
There is a known limitation of this code on Mac OS X. The way that HDF5 files are handled by the `h5py` backend on Mac OS X breaks the multiprocessing paradigm implemented. The workaround is to pass `numpy` arrays to the `FastMapSVM.fit()`, `FastMapSVM.predict()`, and `FastMapSVM.predict_proba()` methods instead of passing handles to HDF5 Datasets (as is shown in the quickstart tutorial above).

# References
White, M. C. A., Sharma, K., Li, A., Kumar, T. K. S., & Nakata, N. (2022). FastMapSVM: Classifying Complex Objects Using the FastMap Algorithm and Support-Vector Machines. _ArXiv_. http://arxiv.org/abs/2204.05112

Ban, T., Kadobayashi, Y., & Abe, S. (2009) Sparse kernel feature analysis using FastMap and its variants. _2009 International Joint Conference on Neural Networks_, pp. 256-263, doi: 10.1109/IJCNN.2009.5178835.
