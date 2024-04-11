# K-Means from Scratch

# Using GPU to run code via cuPy

import os

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import matplotlib.pyplot as plt
import keras
from tqdm import tqdm
import click

# For from scratch kmeans
import cupy as cp

class cluster():
    def __init__(self,maxiter):
        self.kn = 10
        self.maxiter = maxiter
        self.xtrain, self.ytrain, self.xtest,self.ytest = self.data_prep()
        self.centers = self.train_fit()

    def data_prep(self):
        (X1,Y1), (X2,Y2) = keras.datasets.mnist.load_data()
        xtrain = cp.array(X1.reshape(60000,784))
        ytrain = cp.array(Y1)
        xtest = cp.array(X2.reshape(10000,784))
        ytest = cp.array(Y2)
        assert xtrain.ndim == 2
        assert ytrain.ndim == 1
        assert xtest.ndim == 2
        assert ytest.ndim == 1

        return xtrain,ytrain,xtest,ytest
    
    def fit_step(self,data,pred,centers):
        dist = cp.linalg.norm(data[:, None, :] - centers[None, :, :],axis = 2)
        new_pred = cp.argmin(dist, axis = 1)

        pred = new_pred

        c = cp.arange(self.kn)
        mask = pred == c[:,None]
        sums = cp.where(mask[:, :, None], data, 0).sum(axis = 1)
        counts = cp.count_nonzero(mask, axis = 1).reshape((self.kn,1))
        centers = sums/counts

        return pred,centers   

    def train_fit(self):
        pred = cp.zeros(60000)
        initial_center = cp.random.choice(60000,self.kn,replace=False)
        centers = self.xtrain[initial_center]
        change = cp.zeros(self.maxiter)
        acc = cp.zeros(self.maxiter)

        for i in tqdm(range(self.maxiter)):
            new_pred, new_centers = self.fit_step(self.xtrain,pred,centers)

            if cp.all(new_pred == pred):
                acc[i-1::] = acc[i-2]
                change[i-1::] = 0
                break
            change[i-1] = cp.linalg.norm(new_pred - pred)
            pred = new_pred
            centers = new_centers

            acc[i-1] = cp.sum(cp.count_nonzero(self.ytrain == pred))/60000

        fig, ax1 = plt.subplots()
        
        ax2 = ax1.twinx()
        ax1.plot(cp.asnumpy(acc), 'r-',label = "Accuracy")
        ax2.plot(cp.asnumpy(change),'g-',label = "Change in Centroids")
        ax1.set_xlabel('Iterations')
        ax1.set_ylabel('Accuracy %')
        ax2.set_ylabel('Change in Centroids')
        ax1.legend(loc='upper right', bbox_to_anchor=(0.5, 1.15), fancybox=True, shadow=True)
        ax2.legend(loc='upper left', bbox_to_anchor=(0.5, 1.15), fancybox=True, shadow=True)
        plt.savefig('accuracy_change.png')

        return centers

@click.command()
@click.option(
    '--maxiter','-m',
    default=100,
    show_default=True,
    help='Max Iterations'
)

def main(maxiter):
    c1 = cluster(maxiter)
    # run(c1)

if __name__ == "__main__":
    plt.ion()
    main()