# EM for GMM on MNIST

# Using GPU to run code via cuPy

import os

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import matplotlib.pyplot as plt
import keras
from tqdm import tqdm
import click

# For from scratch kmeans
import cupy as cp

class em_gmm():

    def __init__(self,tl,ttl,iter,k):
        self.tl = tl
        self.ttl = ttl
        self.k = k
        self.iter = iter
        self.xtrain, self.xtest,self.ytest = self.data_prep()
        self.weight = 1/10
        
        

    def data_prep(self):
        assert self.tl <= 60000
        assert self.ttl <= 10000
        (X1,Y1), (X2,Y2) = keras.datasets.mnist.load_data()
        xtrain = cp.divide(cp.array(X1.reshape(self.tl,784)),255)
        xtest = cp.divide(cp.array(X2.reshape(self.ttl,784)),255)
        ytest = cp.array(Y2[self.ttl])
        assert xtrain.ndim == 2
        assert xtest.ndim == 2
        assert ytest.ndim == 1

        return xtrain,xtest,ytest

    def train(self):
        initial_means = cp.random.choice(self.tl,self.k,replace=False)

    def e_step(self):
        return
    
    def m_step(self):
        return
    
    def log_like(self):
    

@click.command()
@click.option(
    '--train','-t',
    default=60000,
    show_default=True,
    help='Number of Training Items'
)
@click.option(
    '--test','-tt',
    default=10000,
    show_default=True,
    help='Number of Testing Items'
)
@click.option(
    '--iter','-i',
    default=30,
    show_default=True,
    help='Max Iterations'
)
@click.option(
    '--k','-k',
    default=10,
    show_default=True,
    help='Number of Labels'
)


def main(train,test,iter,k):
    gmm = em_gmm(test,train,iter,k)
    return

if __name__ == "__main__":
    plt.ion()
    main()