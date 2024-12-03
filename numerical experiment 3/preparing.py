import numpy as np

# input: 100 * 701, output: 100
# 701 equally spaced frequencies with a spacing of 2 nm between 1100 and 2500 nm
# split 100 into 80 for training and 20 for testing
input_training_wheat = np.load("input_training_wheat.npy")
output_training = (np.load("output_training_wheat.npy"))
input_testing_wheat = np.load("input_testing_wheat.npy")
output_testing = (np.load("output_testing_wheat.npy"))


def rbf(x, y, sigma):
    return np.exp(- (x[:, None] - y) ** 2 / (2 * sigma ** 2))


def compute_kernel_matrix(input, nystrom_input, point_num, sigma):
    x = np.array([1100 + i * 1400/point_num for i in range(point_num)])
    k = rbf(x, x, sigma)
    return  input @ k @ nystrom_input.T * (1400/point_num) ** 2


def predict(nystrom_input, input_testing, alpha, point_num, sigma):
    x = np.array([1100 + i * 1400/point_num for i in range(point_num)])
    return input_testing @ rbf(x, x, sigma) @ nystrom_input.T @ alpha * (1400/point_num) ** 2


def PCG(kernel_matrix, b, G, Lambda, max_iter = 100):

    # solve Ax=b using G^T @ A @ G = G^T @ b 
    n = np.size(kernel_matrix, 0)
    m = np.size(kernel_matrix, 1)
    iter = 0
    x = np.zeros(m)
    resi = np.copy(b)
    
    while(max(abs(resi)) > 1 and iter < max_iter):
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

        w = np.dot(kernel_matrix.T, np.dot(kernel_matrix, p)) + Lambda * n * m * p
        alpha = rho / np.dot(p, w)
        x = x + alpha * p
        resi = resi - alpha * w

    return (x, iter)




