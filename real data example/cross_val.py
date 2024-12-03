import numpy as np
import random
import matplotlib.pyplot as plt

Lambda = 0.2
sigma = 0.3
nystrom = 64
point_num = 70

# result: 
# if point_num = 700, Lambda = 0.002
# if point_num = 70, Lambda = 0.3
# if point_num = 25, Lambda = 1.2


input_training_wheat = np.load("input_training_wheat.npy")
input_testing_wheat = np.load("input_testing_wheat.npy")


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

def run(input_training, output_training, input_testing, output_testing, total_runs=10):
    average_error = 0

    
    for i in range(total_runs):
        index = random.sample(range(0, np.size(input_training, 0)), nystrom)

        kernel_matrix = compute_kernel_matrix(input_training, input_training[index, :], point_num, sigma)
        kernel_matrix_cutoff = kernel_matrix[index, :]
        G = np.linalg.cholesky(kernel_matrix_cutoff @ kernel_matrix_cutoff\
                                  + Lambda * nystrom * nystrom * np.eye(nystrom))
        for j in range(0, nystrom):
            for i in range(j + 1, nystrom):
                G[i,j] /= -G[j,j]*G[j,j]
            G[j,j] = 1 / G[j,j]
        G = G.T


        b = kernel_matrix.T @ output_training
        (alpha, iter) = PCG(kernel_matrix, b, G, Lambda)
        output = predict(input_training[index, :], input_testing, alpha, point_num, sigma)
  
        average_error += np.dot(output_testing - output, output_testing - output) 
    average_error /= (total_runs * len(output_testing))

    return average_error


error = 0
for i in range(5):
    index1 = [j for j in range(80) if j < 16 * i or j >= 16 * (i+1)]
    index2 = [j for j in range(16 * i, 16 * (i+1))]
    input_training = input_training_wheat[index1, :-1 :round(700/point_num)]
    input_testing = input_training_wheat[index2, :-1 :round(700/point_num)]
    output_training = (np.load("output_training_wheat.npy"))[index1]
    output_testing = (np.load("output_training_wheat.npy"))[index2]
    error += run(input_training, output_training, input_testing, output_testing)
error /= 5


print("sigma:", sigma)
print("point_num:", point_num)
print("nystrom:", nystrom)
print("lambda:", Lambda)
print("error:", error)