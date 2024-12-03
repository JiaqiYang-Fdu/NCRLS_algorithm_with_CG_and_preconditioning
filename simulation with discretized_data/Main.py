import numpy as np
import math
import random
import time
import matplotlib.pyplot as plt

# First run Data_generation.py once, the run Main.py
# N = 10, Lambda = 1e-8
# N = 100, Lambda = 1e-8
# N = 1000, Lambda = 1e-8
N = 10
nystrom = 100
Lambda = 1e-8


M = 50
n_training = 10000
n_testing = 1000
total_run = 25

def generate_data(N):
    gauss_training = np.random.normal(0, 1, [n_training, M])
    gauss_testing = np.random.normal(0, 1, [n_testing, M])

    # noise: N(0,sigma^2)
    sigma = 0.005
    noise_training = sigma * np.random.randn(n_training)
    noise_testing = sigma * np.random.randn(n_testing)
    # prepared for computing excess risk
    noise = np.dot(noise_testing, noise_testing) / len(noise_testing)

    # output data
    temp1 = [k**(-6) for k in range(1, M + 1)]
    output_training = noise_training + math.pi**(-4) * gauss_training @ temp1 / (math.sqrt(M))
    output_testing = noise_testing + math.pi**(-4) * gauss_testing @ temp1 / (math.sqrt(M))

    x = np.linspace(0, 1, N, endpoint = False)
    temp2 = np.zeros([M, M])
    for i in range(M):
        temp2[i, i] = 1 / (i+1)
    temp3 = np.zeros([M, N])
    for i in range(M):
        temp3[i] = np.cos((i+1)* np.pi * x)
    temp4 = np.zeros([M, M])
    for i in range(M):
        temp4[i, i] = 2 / ((i+1) * np.pi) ** 4
    kernel = temp3.T @ temp4 @ temp3    

    return gauss_training @ temp2 @ temp3, output_training, gauss_testing @ temp2 @ temp3, output_testing, noise, kernel

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

        w = np.dot(kernel_matrix.T, np.dot(kernel_matrix, p)) + Lambda * n_training * nystrom * p
        alpha = rho / np.dot(p, w)
        x = x + alpha * p
        resi = resi - alpha * w

    return (x, iter)



def run(total_runs = 25):
    average_iterations = 0
    time_lst = [0 for i in range(total_runs)]
    error_lst = [0 for i in range(total_runs)]   
    input_training, output_training, input_testing, output_testing, noise, kernel = generate_data(N)
    norm = np.dot(output_testing, output_testing) / len(output_testing)

    for k in range(total_runs):
        time_1 = time.time()

        index = random.sample(range(0, n_training), nystrom)
        kernel_matrix = input_training @ kernel @ input_training[index, :].T / (N ** 2)
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

        time_2 = time.time()
        time_lst[k] = time_2 - time_1
        average_iterations += iter

        slope = kernel @ input_training[index, :].T @ alpha / N
        res = input_testing @ slope / N
        error_lst[k] = np.dot(output_testing - res, output_testing - res) / len(output_testing) - noise

    average_iterations /= total_runs
    relative_error = np.abs(np.mean(error_lst) / norm)     

    print("N=", N)
    print("Total runs:", total_runs)
    print("Nystrom:", nystrom)
    print("Lambda:", Lambda)  
    print("Average iterations:", average_iterations)
    print("Computing time:", np.mean(time_lst), np.std(time_lst))
    print("Prediction error:", np.mean(error_lst), np.std(error_lst))
    print("Relative error:", relative_error)

    # plot the true slope function and the predicted one
    # mesh = np.linspace(0, 1, N, endpoint=False)
    # beta = np.zeros(len(mesh))
    # for i in range(1, M + 1):
    #     beta += math.sqrt(2) * math.pi**(-4) * i**(-5) / math.sqrt(M) * np.cos(i * math.pi * mesh)
    # beta_predicted = np.zeros(len(mesh))
    # for i in range(1, M + 1):
    #     beta_predicted += slope[i - 1] * np.cos(i * math.pi * mesh)
    # plt.plot(mesh, beta)
    # plt.plot(mesh, slope)
    # plt.show()

    return


run(total_run)





