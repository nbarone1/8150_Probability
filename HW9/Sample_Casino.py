# Sample Casino for Problem 1

# Import Statements
import numpy as np
from scipy import stats
from tqdm import tqdm
from scipy.linalg import eig
import matplotlib.pyplot as plt
import click

# Creating R.V. X, Y
# For Y, YF and YC are for the outcomes of fair and cheating rolls

class hidden_cas():
    def __init__(self,a,YF,YC,T):
        # a is probability of changing casino state
        self.a = a
        self.YF = stats.rv_discrete(name = "YF",values = ((1,2,3,4,5,6), YF))
        self.YC = stats.rv_discrete(name = "YC",values = ((1,2,3,4,5,6), YC))
        # Case 1 = Fair, 2 = Cheat
        self.cas_start = 0
        self.T = T
        self.M = np.asmatrix(np.array([.95,.05],[.05,.95]))
        self.xc,self.yc = self.build_instances(T)
        self.alphaF, self.alphaC = self.forward()
        self.betaF,self.betaC = self.backward()
        self.Z = self.find_Z()

    def build_instances(self):
        # For part a
        # Take initial state, do t iterations
        state = self.cas_start
        # xc and yc are states and rolls at each step i
        xc = [state]
        y0 = self.YF.rvs(1)-1
        yc = [y0]
        for i in tqdm(range(self.T)):
            p = np.random.uniform(0,1)
            if p < self.a:
                if state == 1:
                    state = 0
                    xc.append(state)
                    yc.append(self.YF.rvs(1)-1)
                elif state == 0:
                    state == 1
                    xc.append(state)
                    yc.append(self.YC.rvs(1)-1)

        return xc,yc
    
    def find_Z(self):
        Z = 0
        return Z
    
    def sample(self,t):
        return self.xc[t-1],self.yc[t-1]
        
    def forward(self):
        alphaF = [self.YF.pmf(self.yc[0])]
        alphaC = [0]
        for i in tqdm(range(self.T)):
            aF = self.M[0,0]*alphaF[i-1]*self.YF.pmf(self.yc[i]) + self.M[0,1]*alphaC[i-1]*self.YC.pmf(self.yc[i])
            aC = self.M[1,0]*alphaF[i-1]*self.YF.pmf(self.yc[i]) + self.M[1,1]*alphaC[i-1]*self.YC.pmf(self.yc[i])
            alphaF.append(aF)
            alphaC.append(aC)

        return alphaF,alphaC
    
    def backward(self):
        betaFd = [1]
        betaCd = [1]
        for i in tqdm(range(self.T)):
            bF = self.M[0,0]*betaFd[i-1]*self.YF.pmf(self.yc[i]) + self.M[0,1]*betaCd[i-1]*self.YC.pmf(self.yc[i])
            bC = self.M[1,0]*betaFd[i-1]*self.YF.pmf(self.yc[i]) + self.M[1,1]*betaCd[i-1]*self.YC.pmf(self.yc[i])
            betaFd.append(bF)
            betaCd.append(bC)
        betaF = betaFd[::-1]
        betaC = betaCd[::-1]
        return betaF, betaC
    
    def t_is_cheat(self,t):
        return self.betaC[t-1]*self.alphaC[t-1]*(1/self.Z)
    
    def plots(self):
        x = []
        yfb = []
        ymcmc = []
        for i in tqdm(range(self.T)):
            x.append(i)
            yfb.append(self.t_is_cheat(i))

        plt.plot(x,yfb,label = "Forward/Backward")
        plt.plot(x,ymcmc,label = "MCMC")
        plt.show()