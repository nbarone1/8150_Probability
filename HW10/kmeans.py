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
    def __init__(self,epochs,maxiter):
        self.kn = 10
        self.epoch = epochs
        self.maxiter = maxiter
        self.xtrain, self.ytrain, self.xtest,self.ytest = self.data_prep()
        self.centers = self.train()

    def data_prep(self):
        (X1,Y1), (X2,Y2) = keras.datasets.mnist.load_data()
        xtrain = cp.divide(cp.array(X1.reshape(60000,784)),255)
        ytrain = cp.array(Y1)
        xtest = cp.divide(cp.array(X2.reshape(10000,784)),255)
        ytest = cp.array(Y2)
        assert xtrain.ndim == 2
        assert ytrain.ndim == 1
        assert xtest.ndim == 2
        assert ytest.ndim == 1

        return xtrain,ytrain,xtest,ytest
    
    def minibatch(self,data):
        mini = cp.random.choice(60000,1024,replace=False)
        d2 = data[mini]
        return d2
    
    def fit_step(self,data,centers):
        minib = self.minibatch(data)
        pred = cp.zeros(len(minib))
        dist = cp.linalg.norm(minib[:, None, :] - centers[None, :, :],axis = 2)
        pred = cp.argmin(dist, axis = 1)

        c = cp.arange(self.kn)
        mask = pred == c[:,None]
        sums = cp.where(mask[:, :, None], minib, 0).sum(axis = 1)
        counts = cp.count_nonzero(mask, axis = 1).reshape((self.kn,1))
        centers = sums/counts

        return centers
    
    def pred_step(self,data,centers):
        dist = cp.linalg.norm(data[:, None, :] - centers[None, :, :],axis = 2)
        pred = cp.argmin(dist, axis = 1)
        return pred

    def batch_fit(self,pred,centers):
        ocenters = centers

        for i in range(self.maxiter):
            new_centers = self.fit_step(self.xtrain,centers)
            new_pred = self.pred_step(self.xtrain,new_centers)

            if cp.all(new_pred == pred):
                change = cp.linalg.norm(ocenters - centers)
                acc = cp.divide(cp.sum(cp.equal(self.ytrain,pred)),len(self.ytrain))
                break            
            pred = new_pred
            centers = new_centers

        change = cp.linalg.norm(ocenters - centers)
        acc = cp.divide(cp.sum(cp.equal(self.ytrain,pred)),len(self.ytrain))

        return change, acc, pred,centers
    
    def train(self):
        pred = cp.zeros(60000)
        initial_center = cp.random.choice(60000,self.kn,replace=False)
        centers = self.xtrain[initial_center]
        acc = cp.zeros(self.epoch)
        change = cp.zeros(self.epoch)
        for i in tqdm(range(self.epoch)):
            change[i],acc[i],pred,centers = self.batch_fit(pred,centers)

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
    
    def test_step(self):
        fpred = self.pred_step(self.xtest,self.centers)
        acc = cp.divide(cp.sum(cp.equal(self.ytest,fpred)),len(self.ytest))
        acc *= 100
        print("Accuracy on Testing Data is %5.2f %" % (acc))

def run(cluster):
    cluster.test_step()


@click.command()
@click.option(
    '--epochs','-e',
    default=50,
    show_default=True,
    help='Max Iterations'
)
@click.option(
    '--maxiter','-m',
    default=60,
    show_default=True,
    help='Max Iterations'
)

def main(epochs,maxiter):
    c1 = cluster(epochs,maxiter)
    run(c1)

if __name__ == "__main__":
    plt.ion()
    main()