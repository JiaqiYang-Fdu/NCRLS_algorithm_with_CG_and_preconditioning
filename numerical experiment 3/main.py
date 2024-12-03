import numpy as np
import random
import matplotlib.pyplot as plt
from preparing import *


Lambda = 1.2
sigma = 0.3
nystrom = 80


point_num = 25
input_training = input_training_wheat[:, :-1 :round(700/point_num)]
input_testing = input_testing_wheat[:, :-1 :round(700/point_num)]
n = np.size(input_training, 0)


def run(total_runs= 200):
    average_iterations = 0
    average_error = 0
    lst = np.zeros(total_runs)
    
    for k in range(total_runs):
        index = random.sample(range(0, n), nystrom)

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

        average_iterations += iter     
        lst[k] = np.dot(output_testing - output, output_testing - output) 
        average_error += lst[k] 
        
    average_iterations /= total_runs
    average_error /= (total_runs * len(output_testing))



    print("lambda:", Lambda)
    print("total_runs:", total_runs)
    print("average_iterations:", average_iterations)
    print("average_error:", average_error)
    print("standard error", np.std(lst))
    return


print("point_num:", point_num)
print("sigma:", sigma)
run()