import numpy as np
import time
import random
import matplotlib.pyplot as plt

N = 1000
n_training = 500
n_testing = 1000


# nystrom = 50, Lambda = 0.00001
# nystrom = 500, Lambda = 0.00001
Lambda = 0.00001
sigma = 0.33
nystrom = 5

x = (np.linspace(0, 1, N))[: -1]

def rbf(x, y, sigma):
    return np.exp(- (x[:, None] - y) ** 2 / (2 * sigma ** 2))

def compute_kernel_matrix(input, index, sigma):
    k = rbf(x, x, sigma)
    return  input @ k @ input[index, :].T / (N ** 2)

def predict(input_training, input_testing, alpha, sigma):
    return input_testing @ rbf(x, x, sigma) @ input_training.T @ alpha / (N ** 2)

def estimate(input_training, alpha, sigma):
    return rbf(x, x, sigma) @ input_training.T @ alpha / N

def PCG(matrix, b, G, Lambda, max_iter = 100):
    # solve Ax=b using G^T @ A @ G = G^T @ b 
    (n, nystrom) = matrix.shape
    iter = 0
    x = np.zeros(nystrom)
    resi = np.copy(b)
    
    while(max(abs(resi)) > 0.5 and iter < max_iter):
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

        w = np.dot(matrix.T, np.dot(matrix, p)) + Lambda * n * nystrom * p
        alpha = rho / np.dot(p, w)
        x = x + alpha * p
        resi = resi - alpha * w

    return (x, iter)


sigma = 0.33
temp = [4 * k** (-2.55) for k in range(1, 51)]
temp1 = np.diag([(-1) ** (k+1) * k ** (-0.55) if k == 1 else\
                    (-1) ** (k+1) * k ** (-0.55) * np.sqrt(2) for k in range(1, 51)])
temp2 = np.array([np.cos(k * np.pi * x) for k in range(50)])
# beta
temp3 = [4 * (-1) ** (k+1) * k** (-2) if k == 1 else\
            4 * (-1) ** (k+1) * k** (-2) * np.sqrt(2) for k in range(1, 51)]
beta = temp2.T @ temp3

def generate_data():
    input_training = np.random.uniform(-np.sqrt(3), np.sqrt(3), [n_training, 50])
    input_testing = np.random.uniform(-np.sqrt(3), np.sqrt(3), [n_testing, 50])

    # noise: N(0,sigma^2)
    noise_training = sigma * np.random.randn(n_training)
    noise_testing = sigma * np.random.randn(n_testing)
    noise = np.dot(noise_testing, noise_testing) / n_testing

    # output data
    output_training = noise_training + input_training @ temp
    output_testing = noise_testing + input_testing @ temp

    # discrete the input with N points
    return input_training @ temp1 @ temp2, input_testing @ temp1 @ temp2, output_training, output_testing, noise




def run(total_runs= 100):
    error_lst = [0 for i in range(total_runs)]
    time_lst = [0 for i in range(total_runs)]
    average_iterations = 0
    average_estimation_error = 0
    average_relative_error = 0
    
    for k in range(total_runs):
        input_training, input_testing, output_training, output_testing, noise = generate_data()

        time_1 = time.time()
        index = random.sample(range(0, n_training), nystrom)
        kernel_matrix = compute_kernel_matrix(input_training, index, sigma)
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
        output = predict(input_training[index, :], input_testing, alpha, sigma)
        time_2 = time.time()

        error_lst[k] = np.dot(output_testing - output, output_testing - output) / n_testing - noise
        time_lst[k] = time_2 - time_1
        average_iterations += iter     
        beta_estimated = estimate(input_training[index, :], alpha,  sigma)
        average_estimation_error += np.dot(beta_estimated - beta, beta_estimated - beta) / N
        average_relative_error += error_lst[k] / (np.dot(output_testing, output_testing) /  n_testing)

    average_iterations /= total_runs
    average_estimation_error /= total_runs
    average_relative_error /= total_runs
    # plt.plot(beta)
    # plt.plot(beta_estimated)
    # plt.show()

    print("lambda:", Lambda)
    print("total_runs:", total_runs)
    print("average_iterations:", average_iterations)
    print("average_estimation_error:", average_estimation_error)
    print("Time (std):", np.mean(time_lst), np.std(time_lst))
    print("Prediction error (std):", np.mean(error_lst), np.std(error_lst))
    print("Relative_error:", average_relative_error)
    return


run()