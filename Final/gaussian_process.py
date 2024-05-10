# Gaussian Process Problem 1

import os

import scipy.linalg

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import numpy as np
import matplotlib.pyplot as plt
import scipy

# Seed for Reproducible results
rng = np.random.default_rng(42)

# Paths, Points, Interval, mu, sigma
paths = 5
interval = [0,10]
dt = .1
points = int((interval[1]-interval[0])/dt)
mu = 0
sigma = 1

# Draw samples
Z = rng.normal(mu, sigma, (paths,points))

t_axis = np.linspace(interval[0], interval[1], points)

# Brownian Motion
W = np.zeros((paths,points))
for i in range(points-1):
    real_i = i+ 1
    W[:, real_i] = W[:, real_i-1] + np.sqrt(dt)*Z[:,i]

# Ornsetin-Uhlenbeck
Cov = np.identity(points)
for i in range(points-1):
    for j in range(i+1,points-1):
        dist = np.abs(i*dt-j*dt)
        # sum = np.abs(i*dt+j*dt)
        cv = np.exp(-5*dist)
        # sv = np.exp(-5*sum)
        Cov[i,j] = cv
A = scipy.linalg.cholesky(Cov)
x_process = A.dot(Z.T)
x_process = x_process.transpose()

fig, ax = plt.subplots(1, 1, figsize=(12, 8))
for path in range(paths):
    ax.plot(t_axis, W[path, :])
ax.set_title("Standard Brownian Motion sample paths")
ax.set_xlabel("Time")
ax.set_ylabel("Value")
plt.show()

fig, ax2 = plt.subplots(1, 1, figsize=(12, 8))
for path in range(paths):
    ax2.plot(t_axis, x_process[path, :])
ax2.set_title("Ornstein Uhlenbeck sample paths")
ax2.set_xlabel("Time")
ax2.set_ylabel("Value")
plt.show()