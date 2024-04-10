# K-Means from Scratch

# Using scikit-learn to compare
# Using GPU to run code

import matplotlib.pyplot as plt
import matplotlib.animation as manimation
import keras
from tqdm import tqdm
import click

# For from scratch kmeans
import cupy as cp

class cluster():
    def __init__(self,kn,maxiter):
        self.kn = kn
        self.maxiter = maxiter
        self.train, self.test = self.data_prep()
        self.size = len(self.data)
        self.pred, self.centers = self.fit()

    def data_prep(self):
        (X1,Y1), (X2,Y2) = keras.datasets.mnist.load_data()
        train = (X1[:5000].reshape(5000,784),Y1[:5000])
        test = (X2[:1000].reshape(1000,784), Y2[:1000])

        return train,test
    
    def fit_step(self,pred,centers):
        dist = cp.linalg.norm(self.data[: None, :] - centers[None, :, :],axis = 2)
        new_pred = cp.argmin(dist, axis = 1)

        pred = new_pred

        c = cp.arange(self.kn)
        mask = pred == c[:,None]
        sums = cp.where(mask[:, :, None], self.data, 0).sum(axis = 1)
        counts = cp.count_nonzero(mask, axis = 1).reshape((self.kn,1))
        centers = sums/counts 

        return pred,centers   

    def fit(self):
        pred = cp.zeros(self.size)
        initial_center = cp.random.choice(self.size,self.kn,replace=False)
        centers = self.data[initial_center]

        FFMpegWriter = manimation.writers['ffmpeg']
        writer = FFMpegWriter(fps=10)

        fig = plt.figure()


        # NEED THIS TO BE 28x28 IMAGE OF THE MEANS
        with writer.saving(fig, "mnist_mask.mp4", 50):

            for i in tqdm(range(self.maxiter)):
                new_pred, new_centers = self.fit_step(pred,centers)

                if cp.all(new_pred == pred):
                    break
                pred = new_pred
                centers = new_centers

                # Run Iterations and record images
                if i % 1000 == 0:
                    for k in range(self.kn):
                        labels = self.data[pred == k]
                        img = plt.scatter(labels[:, 0], labels[:, 1], c=cp.random.rand(3))
                    img = plt.scatter(centers[:, 0], centers[:, 1], s=120, marker='s', facecolors='y',edgecolors='k')
                    writer.grab_frame()
                    img.remove()
                    
        
        plt.close('all')
    
        return pred, centers
    
def run(cluster):
    pred, center = cluster.fit()
    return

@click.command()
@click.option(
    '--kn','-k',
    default=10,
    show_default=True,
    help='Number of Clusters'
)
@click.option(
    '--maxiter','-m',
    default=10000,
    show_default=True,
    help='Max Iterations'
)

def main(kn,maxiter,):
    c1 = cluster(kn,maxiter)
    run(c1)

if __name__ == "__main__":
    plt.ion()
    main()