# Sample Casino for Problem 1

# Import Statements
import numpy as np
from scipy import stats
from tqdm import tqdm
import matplotlib.pyplot as plt
import click

# Creating R.V. X, Y
# For Y, YF and YC are for the outcomes of fair and cheating rolls

class hidden_cas():
    def __init__(self,a,YF,YC,T):
        # a is probability of changing casino state
        self.a = float(a)
        print(YF)
        print(YC)
        self.YF = stats.rv_discrete(name = "YF",values = ((1,2,3,4,5,6), YF))
        self.YC = stats.rv_discrete(name = "YC",values = ((1,2,3,4,5,6), YC))
        # Case 1 = Fair, 2 = Cheat
        self.cas_start = 0
        self.T = T
        self.M = np.asmatrix([[1-self.a, self.a],[self.a, 1-self.a]])
        self.emission = np.asmatrix([YF,YC])
        self.xc,self.yc = self.build_instances()
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
                elif state == 0:
                    state == 1
            xc.append(state)
            if state == 0:
                yc.append(self.YF.rvs(1)-1)
            elif state == 1:
                yc.append(self.YC.rvs(1)-1)

        return xc,yc
    
    def transition_matrix(self):
        M = []
        return M

    def find_Z(self):
        Z = np.dot(self.alphaF,self.betaF)+np.dot(self.alphaC,self.betaC)
        return Z
    
    def sample(self,t):
        return self.xc[t-1],self.yc[t-1]
    
    def mcmc(self):
        return
        
    def forward(self):
        alphaF = [self.YF.pmf(self.yc[0])]
        alphaC = [0]
        for i in tqdm(range(self.T)):
            aF = self.M[0,0]*alphaF[i-1]*self.emission[0,self.yc[i]-1] + self.M[0,1]*alphaC[i-1]*self.emission[1,self.yc[i]-1]
            aC = self.M[1,0]*alphaF[i-1]*self.emission[0,self.yc[i]-1] + self.M[1,1]*alphaC[i-1]*self.emission[1,self.yc[i]-1]
            alphaF.append(aF)
            alphaC.append(aC)

        return alphaF,alphaC
    
    def backward(self):
        betaFd = [1]
        betaCd = [1]
        for i in tqdm(range(self.T)):
            bF = self.M[0,0]*betaFd[i-1]*self.emission[0,self.yc[i]-1] + self.M[0,1]*betaCd[i-1]*self.emission[1,self.yc[i]-1]
            bC = self.M[1,0]*betaFd[i-1]*self.emission[0,self.yc[i]-1] + self.M[1,1]*betaCd[i-1]*self.emission[1,self.yc[i]-1]
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
        for i in tqdm(range(self.T)):
            x.append(i)
            yfb.append(self.t_is_cheat(i))
        
        ymcmc = []

        plt.plot(x,yfb,label = "Forward/Backward")
        # plt.plot(x,ymcmc,label = "MCMC")
        plt.show()
        plt.savefig('hidden_casino.png')

@click.command()
@click.option(
    '--a',
    type = float,
    default=.05,
    show_default=True,
    help='Probability of switching between Cheat and Fair'
)
@click.option(
    '--yf',
    type = np.array,
    default=(1/6,1/6,1/6,1/6,1/6,1/6),
    show_default=True,
    help='Probability of Fair Die'
)
@click.option(
    '--yc',
    type = np.array,
    default=(19/100,19/100,19/100,19/100,19/100,1/20),
    show_default=True,
    help='Probability of Cheat Die'
)
@click.option(
    '--T',
    default=200,
    show_default=True,
    help='T number of observed rolls'
)

def main(a,yf,yc,t):
    print(yf)
    casino = hidden_cas(a,yf,yc,t)
    casino.plots()

if __name__ == "__main__":
    plt.ion()
    main()