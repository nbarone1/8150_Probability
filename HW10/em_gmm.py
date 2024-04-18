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

class theta():
    def __init__(self,x,k):
        initial_means = cp.random.choice(len(x),k,replace=False)
        self.means = x[initial_means]
        self.covs = cp.ones(k)

    def retrieve_mean(self,cluster):
        return self.means[cluster]
    
    def retrieve_covs(self,cluster):
        return self.covs[cluster]
    
    def update_mean(self,cluster,value):
        self.means[cluster] = value

    def update_covs(self,cluster,value):
        self.covs[cluster] = value

class em_gmm():

    def __init__(self,train_length,test_length,iter,k):
        self.trainl = train_length
        self.testl = test_length
        self.k = k
        self.iter = iter
        self.loglike = 0
        self.xtrain, self.ytrain, self.xtest,self.ytest = self.data_prep()
        self.probs = cp.divide(cp.ones(self.k),self.k)
        self.q = cp.zeros(train_length,k)
        assert cp.sum(self.probs) == 1
        self.theta = theta(self.xtrain,self.k)
        
        

    def data_prep(self):
        assert self.trainl <= 60000
        assert self.testl <= 10000
        (X1,Y1), (X2,Y2) = keras.datasets.mnist.load_data()
        xtrain = cp.divide(cp.array(X1.reshape(60000,784)),255)
        xtrain = xtrain[:self.trainl,:]
        ytrain = cp.array(Y1[:self.trainl])
        xtest = cp.divide(cp.array(X2.reshape(10000,784)),255)
        xtest = xtest[:self.testl,:]
        ytest = cp.array(Y2[:self.testl])
        assert xtrain.ndim == 2
        assert ytrain.ndim == 1
        assert xtest.ndim == 2
        assert ytest.ndim == 1

        return xtrain,ytrain,xtest,ytest
    
    def q_calc(self,x,mean,cov,prob):
        d = cp.linalg.norm(x-mean)**2
        c = cov**2
        ep = -d/(2*c)
        qnum = prob*cp.exp(ep)
        qden = (2*c*cp.pi)**(392)
        q_val = qnum/qden
        return q_val
    
    def update_q(self):
        q = cp.zeros(self.trainl,self.k)
        for i in range(self.trainl):
            x = self.xtrain[i]
            for c in range(self.k):
                m = self.theta.retrieve_mean(c)
                c = self.theta.retrieve_covs(c)
                p = self.probs[c]
                q[i][c] = self.q_calc(x,m,c,p)
        
        self.q = cp.linalg.norm(q,axis=1)

    def e_step(self):
        return
    
    def m_step(self):
        return
    
    def log_like(self):
        loglike = 0
        for x in range(self.xtrain):
            for c in range(self.k):
                
        if loglike == self.loglike:
            return -1
        assert self.loglike < loglike
        return loglike
        
    

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