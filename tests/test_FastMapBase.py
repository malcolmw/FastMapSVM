#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 22 13:07:30 2022

@author: malcolmw
"""

import pathlib
import tempfile
import unittest

import h5py
import numpy as np

class TestFastMapBase(unittest.TestCase):
    
    def test_import(self):
        import fastmap as fm

    def test_embed(self):
        import fastmap as fm
        
        def correlation_distance(a, b):
            xcorr = fm.distance.ndcorrelate(a, b, mode="same")
            distance = 1 - np.max(np.abs(np.mean(xcorr, axis=0)))
            return (distance)
        
        with tempfile.TemporaryDirectory() as tmp_dir, h5py.File("../data/ridgecrest.hdf5", mode="r") as in_file:
            fmap = fm.FastMapBase(
                correlation_distance,
                2,
                pathlib.Path(tmp_dir).joinpath("model.hdf5")
            )
            fmap.X = in_file["X/train"]
            fmap.embed_database()
            
            import matplotlib.pyplot as plt
            plt.close("all")
            fig, ax = plt.subplots()
            ax.scatter(fmap.W[:, 0], fmap.W[:, 1], c=in_file["y/train"])
            
            
class TestFastMapSVM(unittest.TestCase):
    def test_fit(self):
        import fastmap as fm
        import sklearn.metrics
        
        def correlation_distance(a, b):
            xcorr = fm.distance.ndcorrelate(a, b, mode="same")
            distance = 1 - np.max(np.abs(np.mean(xcorr, axis=0)))
            return (distance)
        
        with tempfile.TemporaryDirectory() as tmp_dir, h5py.File("../data/ridgecrest.hdf5", mode="r") as in_file:
            clf = fm.FastMapSVM(
                correlation_distance,
                8,
                pathlib.Path(tmp_dir).joinpath("model.hdf5")
            )
            clf.fit(in_file["X/train"], in_file["y/train"])
            
            y_true = in_file["/y/test"][:] # Target test labels.
            proba = clf.predict_proba(
                in_file["/X/test"] # Model test inputs.
            )
            sklearn.metrics.ConfusionMatrixDisplay.from_predictions(
                y_true, 
                proba[:, 1] > 0.5
            )


if __name__ == '__main__':
    unittest.main()