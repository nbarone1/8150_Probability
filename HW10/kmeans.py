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
    def __init__(self,epochs,kn,maxiter,train,test):
        self.kn = kn
        self.epoch = epochs
        self.tl = train
        self.ttl = test
        self.maxiter = maxiter
        self.xtrain, self.ytrain, self.xtest,self.ytest = self.data_prep()
        self.centers = self.train()

    def data_prep(self):
        assert self.tl <= 60000
        assert self.ttl <= 10000
        (X1,Y1), (X2,Y2) = keras.datasets.mnist.load_data()
        xtrain = cp.divide(cp.array(X1.reshape(60000,784)),255)
        xtrain = xtrain[:self.tl,:]
        ytrain = cp.array(Y1[:self.tl])
        xtest = cp.divide(cp.array(X2.reshape(10000,784)),255)
        xtest = xtest[:self.ttl,:]
        ytest = cp.array(Y2[:self.ttl])
        assert xtrain.ndim == 2
        assert ytrain.ndim == 1
        assert xtest.ndim == 2
        assert ytest.ndim == 1

        return xtrain,ytrain,xtest,ytest
    
    def fit_step(self,data,centers):
        data1 = data
        pred = cp.zeros(len(data1))
        dist = cp.linalg.norm(data1[:, None, :] - centers[None, :, :],axis = 2)
        pred = cp.argmin(dist, axis = 1)

        c = cp.arange(self.kn)
        mask = pred == c[:,None]
        sums = cp.where(mask[:, :, None], data1, 0).sum(axis = 1)
        counts = cp.count_nonzero(mask, axis = 1).reshape(self.kn,1)
        centers = sums/counts

        return centers
    
    def label_step(self,data,centers):
        label = {}
        dist = cp.linalg.norm(data[:, None, :] - centers[None, :, :],axis = 2)
        pred = cp.argmin(dist, axis = 1)
        for i in range(self.kn):
            index = cp.where(pred == i)
            num = cp.bincount(self.ytrain[index]).argmax()
            label[i] = num.item()
        return label
    
    def pred_step(self,data,centers):
        label = self.label_step(data,centers)
        dist = cp.linalg.norm(data[:, None, :] - centers[None, :, :],axis = 2)
        pred = cp.argmin(dist, axis = 1)
        res = cp.zeros(len(pred))
        for i in range(len(pred)):
            z = pred[i].item()
            res[i] = label.get(z)
        return res

    def batch_fit(self,pred,centers):
        for i in tqdm(range(self.maxiter)):
            new_centers = self.fit_step(self.xtrain,centers)
            new_pred = self.pred_step(self.xtrain,new_centers)

            if cp.all(new_pred == pred):
                break            
            pred = new_pred
            centers = new_centers

        return pred,centers
    
    def train(self):
        pred = cp.zeros(self.tl)
        initial_center = cp.random.choice(self.tl,self.kn,replace=False)
        centers = self.xtrain[initial_center]
        for i in tqdm(range(self.epoch)):
            pred,ncenters = self.batch_fit(pred,centers)
            centers = ncenters
        
        testpred = self.test_step(self.xtest,centers)

        centers = cp.multiply(centers,255)
        print(cp.max(centers))

        fig, bx1 = plt.subplots(2,5)
        c0,c1,c2,c3,c4,c5,c6,c7,c8,c9 = cp.array_split(centers,10,axis = 0)
        c0 = cp.reshape(c0,(28,28))
        c1 = cp.reshape(c1,(28,28))
        c2 = cp.reshape(c2,(28,28))
        c3 = cp.reshape(c3,(28,28))
        c4 = cp.reshape(c4,(28,28))
        c5 = cp.reshape(c5,(28,28))
        c6 = cp.reshape(c6,(28,28))
        c7 = cp.reshape(c7,(28,28))
        c8 = cp.reshape(c8,(28,28))
        c9 = cp.reshape(c9,(28,28))
        c0 = cp.asnumpy(c0)
        c1 = cp.asnumpy(c1)
        c2 = cp.asnumpy(c2)
        c3 = cp.asnumpy(c3)
        c4 = cp.asnumpy(c4)
        c5 = cp.asnumpy(c5)
        c6 = cp.asnumpy(c6)
        c7 = cp.asnumpy(c7)
        c8 = cp.asnumpy(c8)
        c9 = cp.asnumpy(c9)
        bx1[0,0].matshow(c0,cmap='gray', vmin=0, vmax=255)
        bx1[0,1].matshow(c1,cmap='gray', vmin=0, vmax=255)
        bx1[0,2].matshow(c2,cmap='gray', vmin=0, vmax=255)
        bx1[0,3].matshow(c3,cmap='gray', vmin=0, vmax=255)
        bx1[0,4].matshow(c4,cmap='gray', vmin=0, vmax=255)
        bx1[1,0].matshow(c5,cmap='gray', vmin=0, vmax=255)
        bx1[1,1].matshow(c6,cmap='gray', vmin=0, vmax=255)
        bx1[1,2].matshow(c7,cmap='gray', vmin=0, vmax=255)
        bx1[1,3].matshow(c8,cmap='gray', vmin=0, vmax=255)
        bx1[1,4].matshow(c9,cmap='gray', vmin=0, vmax=255)
        fig.savefig('centroids.png')

        return centers,testpred
    
    def test_step(self,data,centers):
        label = self.label_step(data,centers)
        dist = cp.linalg.norm(data[:, None, :] - centers[None, :, :],ord=3,axis = 2)
        pred = cp.argmin(dist, axis = 1)
        res = cp.zeros(len(data))
        for i in range(len(data)):
            z = pred[i].item()
        res[i] = label.get(z)
        return res


@click.command()
@click.option(
    '--epochs','-e',
    default=5,
    show_default=True,
    help='Max Iterations'
)
@click.option(
    '--kn',
    default=10,
    show_default=True,
    help='Max Iterations'
)
@click.option(
    '--maxiter','-m',
    default=10,
    show_default=True,
    help='Max Iterations'
)
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

def main(epochs,kn,maxiter,train,test):
    c1 = cluster(epochs,kn,maxiter,train,test)

if __name__ == "__main__":
    plt.ion()
    main()