# K-Means from Scratch

# Using scikit-learn to compare
# Using GPU to run code

import matplotlib.pyplot as plt

# For scikit-learn comparison
import sklearn as sk

# For from scratch kmeans
import cupy as cp
from numba import jit, cuda


mnist = cp.load('mnist.csv')

class cluster():
    def __init__(self,kn,file):
        self.kn = kn
        self.file = str(file)
        self.data = self.load_file()

    def load_file(self):
        data = cp.load(self.file)