import Data_generation as data
import numpy as np
import math
import matplotlib.pyplot as plt



def PCG(kernel_matrix, b, G, Lambda, max_iter = 100):

    # solve Ax=b using G^T @ A @ G = G^T @ b 
    nystrom = np.size(kernel_matrix, 1)
    iter = 0
    x = np.zeros(nystrom)
    resi = np.copy(b)
    
    while(max(abs(resi)) > 1.e-4 and iter < max_iter):
        iter += 1
        z = np.dot(G, np.dot(G.T, resi))
        if iter == 1:
            p = z
            rho = np.dot(resi, z)
        else:
            rho1 = rho
            rho = np.dot(resi, z)
            beta = rho / rho1
            p = z + beta * p

        w = np.dot(kernel_matrix.T, np.dot(kernel_matrix, p)) + Lambda * data.n_training * nystrom * p
        alpha = rho / np.dot(p, w)
        x = x + alpha * p
        resi = resi - alpha * w

    return (x, iter)


def predict(x, slope):
    x_1 = np.copy(x)
    for i in range(data.N):
        x_1[i, :] = x_1[i, :] / (i + 1) / math.sqrt(2)
    return x_1.T @ slope





    