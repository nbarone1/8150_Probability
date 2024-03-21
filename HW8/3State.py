# Metropolis Hasting Sampling + Extras

from scipy import stats
import tqdm
import numpy as np
from numpy.linalg import eig

class metro_hast_chain:
    def __init__(self,N,E,P,PP):
        self.chain_length = N
        self.elements = np.array(E)
        self.initial_prob = np.array(P)
        self.proposal_prob = np.array(PP)
        self.chain = []
        self.X = stats.rv_discrete(self.elements,self.initial_prob)
        self.Y = stats.rv_discrete(self.elements,self.proposal_prob)
        self.M = self.initial_prob*self.proposal_prob.T

    def start(self):
        self.chain.append(self.X.rvs(1))

    def advance(self):
        for i in tqdm(range(self.chain_length)):
            x0 = self.chain[i-1]
            x = self.Y.rvs(1)
            if x > np.random.uniform(0,1):
                self.chain.append(x)
            else:
                self.chain.append(x0)

    def stationary_dist(self):
        L,V = eig(self.M)
        l1i = np.argmax(L)
        l1v = L.max()
