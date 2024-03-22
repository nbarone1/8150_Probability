# Metropolis Hasting Sampling + Extras

from scipy import stats
from tqdm import tqdm
import numpy as np
from scipy.linalg import eig
import click

class metro_hast_chain:
    def __init__(self,N,E,P,PP):
        self.chain_length = N
        self.elements = np.array(E)
        self.initial_prob = np.array(P)
        self.proposal_prob = np.array(PP)
        self.chain = []
        self.X = stats.rv_discrete(name="X",values=(self.elements,self.initial_prob))
        self.Y = stats.rv_discrete(name="Y",values=(self.elements,self.proposal_prob))
        self.M = np.asmatrix(self.initial_prob*self.proposal_prob.reshape(3,1))

    def start(self):
        s = self.X.rvs(1)-1
        self.chain.append(s)
        return s

    def advance(self):
        for i in tqdm(range(self.chain_length)):
            x0 = self.chain[i-1]
            x = self.Y.rvs(1)-1
            if x > np.random.uniform(0,1):
                self.chain.append(x)
            else:
                self.chain.append(x0)

    def stationary_dist(self):
        L,V = eig(self.M,left = True,right = False)
        l1v = L.max()
        stat_dist = V[:, 0].T/sum(V[:,0])
        return l1v,stat_dist
    
@click.command()
@click.option(
    '--num','-n',
    default=100,
    type = int,
    show_default=True,
    help='Number of itartations in the chain'
)
@click.option(
    '--element','-e',
    default=[1,2,3],
    type = list,
    show_default=True,
    help='List of elements/states'
)
@click.option(
    '--probability','-p',
    default=[1/2,1/3,1/6],
    type=list,
    show_default=True,
    help='Probabilities of each element/state'
)
@click.option(
    '--proposal','-pp',
    default=[.99,.009,.001],
    type = list,
    show_default=True,
    help='Probability of Proposing a state'
)

def main(num,element,probability,proposal):
    mhc3 = metro_hast_chain(num,element,probability,proposal)
    start_state = mhc3.start()
    print(start_state)
    mhc3.advance()
    eigenvalue, stationary = mhc3.stationary_dist()
    print(stationary)
    print(eigenvalue)
    chain = mhc3.chain
    print(chain)


if __name__ == "__main__":
    main()