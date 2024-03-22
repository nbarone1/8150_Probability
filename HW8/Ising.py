# Nicholas Barone
# HW8 Problem 2 File

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as manimation; manimation.writers.list()
from tqdm import tqdm
import click

# (a) On paper, for ease of reading

# (b)

# IsingLattice class
class IsingLattice:
    def __init__(self,size):
        self.size = size
        self.system = self._build_system()

    @property
    def sqr_size(self):
        return(self.size, self.size)
    
    def _build_system(self):
        """Build the Lattice and the underlying system.
        
        Set for random dispersion of -1 and 1 values.

        Return:
        Ising Lattice of size by size with randomly assigned -1 and 1 values at each coordinate.
        """

        system = np.random.choice([-1,1], self.sqr_size)

        return system

    def _bc(self,i):
        """Checking lattice coordinate is inside of the bounds and applying boundary condition if it falls outside.
        
        Assuming square lattice.

        Parameters:
        i: int: lattice site coordinate (either x or y)

        Return:
        int: correct coordiante
        
        """
        if i >= self.size:
            return 0
        if i < 0:
            return self.size - 1
        else:
            return i
    
    def node_diff(self,M,N):
        """
        Calculate the state differences at a node (N,M) with its neighbors and applied boundary condition.
        
        Parameters:
        N: int: x-coordinate of node
        M: int: y-coordinate of node

        Return:
        int: The sum of state differences between the node (N,M) and its neighbors.
        """

        return self.system[M,N] * (self.system[self._bc(N - 1), M] + self.system[self._bc(N + 1), M]+ self.system[N, self._bc(M - 1)] + self.system[N, self._bc(M + 1)])
    
    # If we choose to calculate P(w)
    # def system_diff(self):
    #     """
    #     Calculate the difference across all nodes in the system.

    #     Return:
    #     int: difference across all nodes and neighbors in the system.
    #     """

    #     SD = 0

    #     for i in range(self.size):
    #         for j in range(self.size):
    #             SD += np.exp(self.node_diff(i,j))
        
    #     return SD

    def config_change(self):
        """ Decide whether or not to change the state at (M,N)
        
        Checking the value of node-diff(M,N)
        
        if node_diff <= 0 flip, else if e^(-node_diff) > uniform random acceptance condition flip
        
        Return:
        lattice: updated with the potential flip at (M,N)"""
        M,N = np.random.randint(0,self.size,2)

        if self.node_diff(M,N) <= 0:
            self.system[M,N] *= -1
        elif np.exp(-self.node_diff(M,N)) < np.random.uniform(0,1):
            self.system[M,N] *= 1

    def same(self,N,M):
        """
        Determine if a node as neighbors of all the same values
        
        Return:
        0,1: 0 if not all neighbors are the same, 1 if they are all the same
        """

        # indic is indicator function under our logic system
        # neighbors = [self.system[self._bc(N - 1), M], self.system[self._bc(N + 1), M],
        #     self.system[N, self._bc(M - 1)], self.system[N, self._bc(M + 1)]]
        # indic = [1 if x == self.system[M,N] else 0 for x in neighbors]
        # print(indic)

        # if 0 in indic:
        #     return 0
        # else:
        #     return 1

        if self.node_diff(N,M) == 4:
            return 1
        else:
            return 0





def run(lattice,burn_in,iterations,video=True):
    """ Run the simulation
    
    Inputs:
    lattice: Ising Model: model that will be simulated on
    burn_in: int: burn-in number
    iterations: int: number of iterations
    video: True/False: display video playback
    
    Return:
    saved file with video of simulation
    E[f(Y)] where f is sum(i,j) product of indicator function on (i,j) and its neighbor in the lattice
    """

    # Run a set number of burn in iterations
    for b in tqdm(range(burn_in)):
        lattice.config_change()

    # Set up image recording
    FFMpegWriter = manimation.writers['ffmpeg']
    writer = FFMpegWriter(fps=10)

    fig = plt.figure()

    # Run Iterations and record images
    with writer.saving(fig, "ising.mp4", 50):
        for i in tqdm(range(iterations)):
            lattice.config_change()

            if video and i % 10000 == 0:
                img = plt.imshow(lattice.system,cmap="jet",interpolation="nearest")
                writer.grab_frame()
                img.remove()
    
    plt.close('all')

    efw = 0
    for i in range(lattice.size):
        for j in range(lattice.size):
            efw += lattice.same(i,j)
            
    print(lattice.system)
    print(efw)

@click.command()
@click.option(
    '--size','-s',
    default=100,
    show_default=True,
    help='Number of sites, M, in the MxM lattice'
)
@click.option(
    '--burn_in','-b',
    default=1000,
    type=int,
    show_default=True,
    help='Number of burn in iterations to run'
)
@click.option(
    '--iterations','-i',
    default=4_000_000,
    type=int,
    show_default=True,
    help='Number of iterations to run the simulation for'
)
@click.option(
    '--video',
    default = True,
    is_flag=False,
    help='Record a video of the simulation progression'
)
def main(size,burn_in,iterations,video):
    # Our main function to run the simulation
    lattice = IsingLattice(size)
    run(lattice,burn_in,iterations,video)

if __name__ == "__main__":
    plt.ion()
    main()