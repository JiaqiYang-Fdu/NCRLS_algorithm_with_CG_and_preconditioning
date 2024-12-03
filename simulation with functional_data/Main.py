import PCG_algorithm as PCG
import numpy as np
import random
import time
import math
import matplotlib.pyplot as plt

# First run Data_generation.py once, the run Main.py

# load data
gauss_training = np.load('gauss_training.npy')
output_training = np.load('output_training.npy')
covariate_shift_1 = np.load('covariate_shift_1.npy')
covariate_shift_2 = np.load('covariate_shift_2.npy')

input_testing = np.load('gauss_testing.npy')
output_testing = np.load('output_testing.npy')
output_covariant_shift_1 = np.load('output_covariant_shift_1.npy')
output_covariant_shift_2 = np.load('output_covariant_shift_2.npy')

noise = np.load('noise.npy')


norm_1 = np.dot(output_testing, output_testing) / len(output_testing)
norm_2 = np.dot(output_covariant_shift_1, output_covariant_shift_1) / len(output_covariant_shift_1)
norm_3 = np.dot(output_covariant_shift_2, output_covariant_shift_2) / len(output_covariant_shift_2)

# nystrom = 100, Lambda = 1e-09 for the last distribution, and 1e-08 for the first two distribution
# nystrom = 1000, Lambda = 1e-09 for the last distribution, and 1e-08 for the first two distribution
# nystrom = 5000, Lambda = 1e-09 for the last distribution, and 1e-08 for the first two distribution
nystrom = 1000
Lambda = 1e-8


def run(total_runs = 5):
    average_iterations = 0
    time_lst = [0 for i in range(total_runs)]
    error_lst1 = [0 for i in range(total_runs)]
    error_lst2 = [0 for i in range(total_runs)]
    error_lst3 = [0 for i in range(total_runs)]    
    
    for k in range(total_runs):
        time_1 = time.time()

        index = random.sample(range(0, PCG.data.n_training), nystrom)
        temp = np.diag([k**(-6) * math.pi**(-4) for k in range(1, PCG.data.N + 1)])
        kernel_matrix = gauss_training.T @ temp @ gauss_training[:, index]
        kernel_matrix_cutoff = kernel_matrix[index, :]

        G = np.linalg.cholesky(kernel_matrix_cutoff @ kernel_matrix_cutoff\
                                  + Lambda * nystrom * nystrom * np.eye(nystrom))
        for j in range(0, nystrom):
            for i in range(j + 1, nystrom):
                G[i,j] /= -G[j,j]*G[j,j]
            G[j,j] = 1 / G[j,j]
        G = G.T


        b = kernel_matrix.T @ output_training
        (alpha, iter) = PCG.PCG(kernel_matrix, b, G, Lambda)

        time_2 = time.time()

        time_lst[k] = time_2 - time_1
        average_iterations += iter

        temp = np.diag([math.sqrt(2) * k**(-5) * math.pi**(-4) for k in range(1, PCG.data.N + 1)])
        slope = temp @ (gauss_training[:, index] @ alpha)
        res_1 = PCG.predict(input_testing, slope)       
        res_2 = PCG.predict(covariate_shift_1, slope)
        res_3 = PCG.predict(covariate_shift_2, slope)
        
        error_lst1[k] = np.dot(output_testing - res_1, output_testing - res_1) / len(output_testing) - noise
        error_lst2[k] = np.dot(output_covariant_shift_1 - res_2, output_covariant_shift_1 - res_2) / len(output_covariant_shift_1) - noise
        error_lst3[k] = np.dot(output_covariant_shift_2 - res_3, output_covariant_shift_2 - res_3) / len(output_covariant_shift_2) - noise

    average_iterations /= total_runs

    
    relative_error_1 = np.abs(np.mean(error_lst1) / norm_1)      
    relative_error_2 = np.abs(np.mean(error_lst2) / norm_2)        
    relative_error_3 = np.abs(np.mean(error_lst3) / norm_3)

    print("Total runs:", total_runs)
    print("Nystrom:", nystrom)
    print("Lambda:", Lambda)  
    print("Average iterations:", average_iterations)
    print("Computing time:", np.mean(time_lst), np.std(time_lst))
    print("Prediction error:", np.mean(error_lst1), np.std(error_lst1))
    print("Prediction error (covariate shift 1):", np.mean(error_lst2), np.std(error_lst2))
    print("Prediction error (covariate shift 2):", np.mean(error_lst3), np.std(error_lst3))
    print("Relative error:", relative_error_1)
    print("Relative error (covariate shift 1):", relative_error_2)
    print("Relative error (covariate shift 2):", relative_error_3)


    # plot the true slope function and the predicted one
    mesh = np.linspace(0, 1, 1001)
    beta = np.zeros(len(mesh))
    for i in range(1, PCG.data.N + 1):
        beta += math.sqrt(2) * math.pi**(-4) * i**(-5) / math.sqrt(PCG.data.N) * np.cos(i * math.pi * mesh)
    beta_predicted = np.zeros(len(mesh))
    for i in range(1, PCG.data.N + 1):
        beta_predicted += slope[i - 1] * np.cos(i * math.pi * mesh)
    plt.plot(mesh, beta)
    plt.plot(mesh, beta_predicted)
    plt.show()

    return


def run_and_show_each_step(total_steps = 10, total_runs = 5):
    error_lst1 = np.zeros(total_steps)
    error_lst2 = np.zeros(total_steps)
    error_lst3 = np.zeros(total_steps)  

    for k in range(total_runs):

        index = random.sample(range(0, PCG.data.n_training), nystrom)
        temp = np.diag([k**(-6) * math.pi**(-4) for k in range(1, PCG.data.N + 1)])
        kernel_matrix = gauss_training.T @ temp @ gauss_training[:, index]
        kernel_matrix_cutoff = kernel_matrix[index, :]

        G = np.linalg.cholesky(kernel_matrix_cutoff @ kernel_matrix_cutoff\
                                  + Lambda * nystrom * nystrom * np.eye(nystrom))
        for j in range(0, nystrom):
            for i in range(j + 1, nystrom):
                G[i,j] /= -G[j,j]*G[j,j]
            G[j,j] = 1 / G[j,j]
        G = G.T
        b = kernel_matrix.T @ output_training

        iter = 0
        alpha = np.zeros(nystrom)
        resi = np.copy(b)
        while(iter < total_steps):
            z = np.dot(G, np.dot(G.T, resi))
            if iter == 0:
                p = z
                rho = np.dot(resi, z)
            else:
                rho1 = rho
                rho = np.dot(resi, z)
                beta = rho / rho1
                p = z + beta * p

            w = np.dot(kernel_matrix.T, np.dot(kernel_matrix, p)) + Lambda * PCG.data.n_training * nystrom * p
            x = rho / np.dot(p, w)
            alpha = alpha + x * p
            resi = resi - x * w

            temp2 = np.diag([math.sqrt(2) * k**(-5) * math.pi**(-4) for k in range(1, PCG.data.N + 1)])
            slope = temp2 @ (gauss_training[:, index] @ alpha)
            res_1 = PCG.predict(input_testing, slope)       
            res_2 = PCG.predict(covariate_shift_1, slope)
            res_3 = PCG.predict(covariate_shift_2, slope)

            error_lst1[iter] += np.dot(output_testing - res_1, output_testing - res_1) / len(output_testing) - noise
            error_lst2[iter] += np.dot(output_covariant_shift_1 - res_2, output_covariant_shift_1 - res_2) / len(output_covariant_shift_1) - noise
            error_lst3[iter] += np.dot(output_covariant_shift_2 - res_3, output_covariant_shift_2 - res_3) / len(output_covariant_shift_2) - noise

            iter += 1


    error_lst1 = np.abs(error_lst1) / (total_runs * norm_1)
    error_lst2 = np.abs(error_lst2) / (total_runs * norm_2)
    error_lst3 = np.abs(error_lst3) / (total_runs * norm_3)




    # plot
    x_axis = [i for i in range(1, total_steps+1)]
    x_major_locator=plt.MultipleLocator(2)
    ax=plt.gca()
    plt.tight_layout()
    ax.xaxis.set_major_locator(x_major_locator)
    plt.plot(x_axis, error_lst1, label='distribution 1',marker='o', markersize=5, linestyle='-', color = 'red')
    plt.plot(x_axis, error_lst2, label='distribution 2',marker='s', markersize=5, linestyle='--', color = 'green')
    plt.plot(x_axis, error_lst3, label='distribution 3',marker='*', markersize=5, linestyle='-.', color = 'blue')
    plt.legend()
    # plt.title('')
    plt.ylabel('Relative error')
    plt.xlabel('Iterations (t)')
    plt.show()


    return error_lst1, error_lst2, error_lst3



# run()

error_lst1, error_lst2, error_lst3 = run_and_show_each_step(10)
print(error_lst1)
print(error_lst2)
print(error_lst3)




