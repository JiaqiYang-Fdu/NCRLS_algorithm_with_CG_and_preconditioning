import numpy as np
import math


N = 50
n_training = 1000000
n_testing = 10000

if __name__ == '__main__':
    gauss_training = np.random.normal(0, 1, [N, n_training])
    gauss_testing = np.random.normal(0, 1, [N, n_testing])
    covariate_shift_1 = np.random.normal(0, 0.9, [N, n_testing])
    covariate_shift_2 = np.random.uniform(-10, 10, [N, n_testing])


    # noise: N(0,sigma^2)
    sigma = 0.005
    noise_training = sigma * np.random.randn(n_training)
    noise_testing = sigma * np.random.randn(n_testing)
    # prepared for computing excess risk
    noise = np.dot(noise_testing, noise_testing) / len(noise_testing)

    # output data
    temp = [k**(-6) for k in range(1, N + 1)]
    output_training = noise_training + math.pi**(-4) * gauss_training.T @ temp / (math.sqrt(N))
    output_testing = noise_testing + math.pi**(-4) * gauss_testing.T @ temp / (math.sqrt(N))
    output_covariant_shift_1 = noise_testing + math.pi**(-4) * covariate_shift_1.T @ temp / (math.sqrt(N))
    output_covariant_shift_2 = noise_testing + math.pi**(-4) * covariate_shift_2.T @ temp / (math.sqrt(N))


    # save the data
    np.save('gauss_training.npy', gauss_training)
    np.save('gauss_testing.npy', gauss_testing)
    np.save('covariate_shift_1', covariate_shift_1)
    np.save('covariate_shift_2', covariate_shift_2)

    np.save('output_training.npy', output_training)
    np.save('output_testing.npy', output_testing)
    np.save('output_covariant_shift_1', output_covariant_shift_1)
    np.save('output_covariant_shift_2', output_covariant_shift_2)

    np.save('noise.npy',noise)
