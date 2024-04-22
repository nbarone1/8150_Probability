# EM for GMM on MNIST

import os

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import matplotlib.pyplot as plt
import keras
from tqdm import tqdm
import click

import cupy as cp

class theta():
    def __init__(self,x,k):
        # Move probabilities into this class
        initial_means = cp.random.choice(len(x),k,replace=False)
        self.means = x[initial_means]
        self.covs = cp.ones(k)
        self.probs = cp.divide(cp.ones(k),k)

    def retrieve_mean(self,cluster):
        return self.means[cluster]
    
    def retrieve_covs(self,cluster):
        return self.covs[cluster]
    
    def retrieve_probs(self,cluster):
        return self.probs[cluster]
    
    def update_mean(self,cluster,value):
        self.means[cluster] = value

    def update_covs(self,cluster,value):
        self.covs[cluster] = value

    def update_probs(self,cluster,value):
        self.probs[cluster]= value

class em_gmm():

    def __init__(self,train_length,test_length,iter,k):
        self.trainl = train_length
        self.testl = test_length
        self.k = k
        self.iter = iter
        self.loglike = 0
        self.xtrain, self.ytrain, self.xtest,self.ytest = self.data_prep()
        self.q = cp.zeros((train_length,k))
        self.theta = theta(self.xtrain,self.k)
        
    def data_prep(self):
        assert self.trainl <= 60000
        assert self.testl <= 10000
        # Testing if it works when we do not normalize the data
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
        ep = -d/(2*cp.pi*c)
        diff = cp.log(prob) + ep - 392*cp.log(2*c*cp.pi)
        # Take the exponential value after doing all of the calcuations
        # Avoid the underflow
        q_val = cp.exp(diff)
        return q_val
    
    def update_q(self,data):
        q = cp.zeros((len(data),self.k))
        # Switch loop order for faster run ?
        for i in range(len(data)):
            x = data[i]
            for c in range(self.k):
                m = self.theta.retrieve_mean(c)
                co = self.theta.retrieve_covs(c)
                p = self.theta.retrieve_probs(c)
                a = self.q_calc(x,m,co,p)
                q[i,c] = self.q_calc(x,m,co,p)
        q_max = cp.max(q,axis =1)
        q = cp.divide(q,q_max[:,None])
        q_row_sums = cp.sum(q,axis=1)
        self.q = cp.divide(q,q_row_sums[:,None])

    def log_like(self,data):
        loglike = 0
        # Switch loop order for faster run ?
        for i in range(len(data)):
            x = data[i]
            for c in range(self.k):
                m = self.theta.retrieve_mean(c)
                co = self.theta.retrieve_covs(c)
                p = self.theta.retrieve_probs(c)
                d = cp.linalg.norm(x-m)**2
                co = co**2
                ep = -d/(2*co*cp.pi)
                diff = cp.log(p) + ep - 392*cp.log(2*co*cp.pi)
                a = (self.q[i,c])
                loglike += diff*self.q[i,c]
                
        if loglike == self.loglike:
            return 1
        self.loglike = loglike
        return 0

    def e_step(self):
        self.update_q(self.xtrain)
        s = self.log_like(self.xtrain)
        return s
    
    def new_means(self):
        qrows = cp.sum(self.q,axis = 0)
        for c in range(self.k):
            mu = cp.zeros(784)
            for i in range(self.trainl):
                qsum = self.q[i,c]
                x = self.xtrain[i]
                mu += cp.multiply(x,qsum)
            mu = cp.divide(mu,qrows[c])
            self.theta.update_mean(c,mu)

    def new_covs(self):
        for c in range(self.k):
            val = 0
            for i in range(self.trainl):
                d = cp.linalg.norm(self.xtrain[i] - self.theta.retrieve_mean(c))**2
                q = self.q[i,c]
                val += d*q
            val = val/(4*cp.pi*784)
            val = cp.sqrt(val)
            self.theta.update_covs(c,val)

    def new_prob(self):
        qsums = cp.sum(self.q,axis = 0)
        prob_check = 0
        for c in range(self.k):
            a = qsums[c]
            new_prob = qsums[c]/self.trainl
            self.theta.update_probs(c,new_prob)
            prob_check += new_prob
        fyck = cp.sum(prob_check)
        assert cp.sum(prob_check) == 1
    
    def m_step(self):
        self.new_covs()
        self.new_means()
        self.new_prob()
        return

    def dictionary(self):
        self.dict = dict.fromkeys(range(self.k))
        pred_val = cp.argmax(self.q,axis=1)
        for i in range(self.k):
            index = cp.where(pred_val == i)
            num = cp.bincount(self.ytrain[index]).argmax()
            self.dict[i] = num.item()
    
    def pred(self):
        q = cp.zeros((len(self.xtest),self.k))
        for i in range(len(self.xtest)):
            x = self.xtest[i]
            for c in range(self.k):
                m = self.theta.retrieve_mean(c)
                co = self.theta.retrieve_covs(c)
                p = self.theta.retrieve_probs(c)
                q[i,c] = self.q_calc(x,m,co,p)
        
        q_row_sums = cp.sum(q,axis=1)
        self.q = cp.divide(q,q_row_sums[:,None])
        pred_val = cp.argmax(q,axis=1)
        pred = cp.array(len(self.ytest))
        for i in range(len(pred_val)):
            z = pred_val[i].item()
            pred[i] = self.dict.get(z)
        accuracy = cp.where(self.ytest == pred)/len(self.ytest)
        return pred_val,accuracy
    
    def run(self):
        for i in tqdm(range(self.iter)):
            s = self.e_step()
            if s == 1:
                break
            self.m_step()

        self.dictionary()

        pred_val, accuracy = self.pred()

        plot_numbers = cp.ceil(self.k/5)
        fig, ax1 = plt.subplots(plot_numbers,5)
        for c in range(self.k):
            plot_position1 = cp.floor(c/5)
            plot_position2 = c % 5
            data = self.theta.retrieve_mean(c)
            data = cp.reshape(data,(28,28))
            ax1[plot_position1,plot_position2].matshow(data,cmap='gray', vmin=0, vmax=255)

        fig.suptitle('GMM with "{:.0%}"'.format(accuracy))
        fig.savefig('em_means.pdf')
        
    
        
    

@click.command()
@click.option(
    '--train','-t',
    default=5000,
    show_default=True,
    help='Number of Training Items'
)
@click.option(
    '--test','-tt',
    default=1000,
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
    gmm = em_gmm(train,test,iter,k)
    gmm.run()

if __name__ == "__main__":
    plt.ion()
    main()