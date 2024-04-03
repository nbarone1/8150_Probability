# Sample Casino for Problem 1

# Import Statements
import numpy as np
from scipy import stats
from tqdm import tqdm
from scipy.linalg import eig
import click

# Creating R.V. X, Y
# For Y, YF and YC are for the outcomes of fair and cheating rolls

class hidden_cas():
    def __init__(self,a,YF,YC,x0):
        # a is probability of changing casino state
        self.a = a
        self.YF = stats.rv_discrete(name = "YF",values = ((1,2,3,4,5,6), YF))
        self.YC = stats.rv_discrete(name = "YC",values = ((1,2,3,4,5,6), YC))
        self.cas_start = x0
        self.M = np.asmatrix(np.array([.95,.05],[.05,.95]))

    def sample_t(self,t):
        # For part a
        # Take initial state, do t iterations
        state = self.cas_start
        # xc and yc are states and rolls at each step i
        xc = [state]
        y0 = self.YF.rvs(1)-1
        yc = [y0]
        for i in tqdm(range(t)):
            p = np.random.uniform(0,1)
            if p < self.a:
                if state == "C":
                    state = "F"
                    xc.append(state)
                    yc.append(self.YF.rvs(1)-1)
                elif state == "F":
                    state == "C"
                    xc.append(state)
                    yc.append(self.YC.rvs(1)-1)

        return xc[-1], yc[-1]
    
    def stationary_dist(self):
        L,V = eig(self.M,left = True,right = False)
        l1v = L.max()
        stat_dist = V[:, 0].T/sum(V[:,0])
        return l1v,stat_dist
        
    def forward(self,T):
        xc,yc = self.sample_t(T)
        a0 = self.YF.pmf(yc[0])
        alpha = [a0]
        