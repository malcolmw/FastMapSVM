import numpy as np
import pandas as pd
import pathlib
import scipy.signal
import tqdm

nnoise_train = 8192
neq_train    = 8192
sos = scipy.signal.butter(2, [1, 20], btype="bandpass", output="sos", fs=100)

dataf0 = pd.concat(
    [pd.read_csv(f"/home/malcolmw/proj/fastmapsvm/data/stead/chunk{i}.csv") for i in range(1, 7)],
    ignore_index=True
)

dataf_noise = dataf0[dataf0["chunk"] == 1]
dataf_noise_train = dataf_noise.sample(n=nnoise_train)
dataf_noise_test  = dataf_noise[~dataf_noise.index.isin(dataf_noise_train.index)]
dataf_noise_train = dataf_noise_train.reset_index(drop=True)
dataf_noise_test  = dataf_noise_test.reset_index(drop=True)

dataf_eq = dataf0[dataf0["chunk"] > 1]
dataf_eq_train = dataf_eq.sample(n=neq_train)
dataf_eq_test  = dataf_eq[~dataf_eq.index.isin(dataf_eq_train.index)]
dataf_eq_train = dataf_eq_train.reset_index(drop=True)
dataf_eq_test  = dataf_eq_test.reset_index(drop=True)

n_train = len(dataf_noise_train) + len(dataf_eq_train)
n_test = len(dataf_noise_test) + len(dataf_eq_test)

with h5py.File("/home/malcolmw/proj/fastmapsvm/data/stead/train.hdf5", mode="w") as f5out:
    X = f5out.create_dataset("X", shape=(n_train, 3, 6000), dtype=np.float32)
    y = f5out.create_dataset("y", shape=(n_train,), dtype=np.uint8)
    i = 0
    for label, dataf in enumerate((dataf_noise_train, dataf_eq_train)):
        for j, row in tqdm.tqdm(dataf.iterrows(), total=len(dataf)):
            ichunk = row["chunk"]
            handle = row["trace_name"]
            with h5py.File(f"/home/malcolmw/proj/fastmapsvm/data/stead/chunk{ichunk}.hdf5", mode="r") as f5in:
                x = f5in[f"/data/{handle}"][:]
                x = scipy.signal.sosfiltfilt(sos, x, axis=0)
                try:
                    x = (x - np.mean(x, axis=0)) / np.std(x, axis=0)
                except:
                    x = x - np.mean(x, axis=0)
                X[i] = x.T
                y[i] = label
            i += 1

with h5py.File("/home/malcolmw/proj/fastmapsvm/data/stead/test.hdf5", mode="w") as f5out:
    X = f5out.create_dataset("X", shape=(n_test, 3, 6000), dtype=np.float32)
    y = f5out.create_dataset("y", shape=(n_test,), dtype=np.uint8)
    i = 0
    for label, dataf in enumerate((dataf_noise_test, dataf_eq_test)):
        for j, row in tqdm.tqdm(dataf.iterrows(), total=len(dataf)):
            ichunk = row["chunk"]
            handle = row["trace_name"]
            with h5py.File(f"/home/malcolmw/proj/fastmapsvm/data/stead/chunk{ichunk}.hdf5", mode="r") as f5in:
                x = f5in[f"/data/{handle}"][:]
                x = scipy.signal.sosfiltfilt(sos, x, axis=0)
                try:
                    x = (x - np.mean(x, axis=0)) / np.std(x, axis=0)
                except:
                    x = x - np.mean(x, axis=0)
                X[i] = x.T
                y[i] = label
            i += 1
