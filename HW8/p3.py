# Sampler definitions for problem 3 and its corresping results

from scipy import stats
import numpy as np

# Defining our discrete random variable X, using a discrete random variable
def discrete_3_sample(X_W,P_W):
    """
    Creating the defined random variable and using a discrete sampler for comparison.

    Parameters:
    X_W: list int: list of values w of W (Omega)
    P_W: list float: list of probabilites for each possible w
    
    Return:
    sample: int: value from the list of states chosen based on its corresponding probability
    """
    X = stats.rv_discrete(name = "X",values = (X_W, P_W))
    sample = X.rvs(1)-1
    return sample

# Sampling from the same X_W and P_W using a random uniform variable
def disc_samp(val,prob):
    """
    The function `disc_samp` performs discrete sampling based on provided values and probabilities.
    
    Parameters:
    val: list: int: list of values from which to sample
    prob: list: float: list of probabilities corresponding to the values in the `val` parameter

    Return: 
    The value returned is the element from the `val` list
    corresponding to the probability that was selected based on the random uniform value `u`.
    """
    u = np.random.uniform(0,1)
    for w in enumerate(val):
        u -= prob[w[0]]
        if u < 0:
            return w[1]

# For problem 3(a)
X_W = (1,2,3)
P_W = (1/2,1/3,1/6)

print(discrete_3_sample(X_W,P_W))
print(disc_samp(X_W,P_W))