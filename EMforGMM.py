import numpy as np
from scipy.stats import multivariate_normal
from random import random
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def logLikelihoodCalculation(data, K, N, means, covariances, mixing_coefficients):
    likelihood = np.zeros((N, K))
    for n in range(N):
        for k in range(K):
            likelihood[n,k] = multivariate_normal.pdf(data[n], means[k], covariances[k])
    log_likelihood = np.sum(np.log(likelihood.dot(mixing_coefficients)))
    return log_likelihood

def expectationStep(data, K, N, means, covariances, mixing_coefficients):
    gamma = np.zeros((N,K))
    for n in range(N):
        sum_denominator = 0
        for k in range(K):
            denominator = mixing_coefficients[k]*multivariate_normal.pdf(data[n], means[k], covariances[k])
            sum_denominator += denominator
        for k in range(K):
            numerator = mixing_coefficients[k]*multivariate_normal.pdf(data[n], means[k], covariances[k])
            gamma[n,k] = numerator/sum_denominator
    return gamma

def maximizationStep(data, K, N, D, gamma):
    Nk = np.zeros(K)
    means = np.zeros((K, D))
    covariances = np.zeros((K, D, D))
    mixing_coefficients = np.zeros(K)

    for k in range(K):
        sum = 0
        for n in range(N):
            sum = sum + gamma[n,k]
        Nk[k] = sum

        sum = 0
        for n in range(N):
            sum = sum + gamma[n,k]*data[n]
        means[k] = sum/Nk[k]

        sum = 0
        for n in range(N):
            sum = sum + gamma[n,k]*(data[n]-means[k])*(data[n]-means[k])[np.newaxis].T
        covariances[k] = sum/Nk[k]

        mixing_coefficients[k] = Nk[k]/N
    return means, covariances, mixing_coefficients


def prediction(data, K, N, means, covariances, mixing_coefficients):
    predicted_k = np.zeros(N, dtype=int)
    for n in range(N):
        probabilities = np.zeros(K)
        for k in range(K):
            probabilities[k] = mixing_coefficients[k]*multivariate_normal.pdf(data[n], means[k], covariances[k])
        predicted_k[n] = int(np.argmax(probabilities))
    return predicted_k


def plotting(i, data, predicted_k):

    data_frame = pd.DataFrame(data)
    data_frame['cluster'] = predicted_k
    data_frame.columns = ['x', 'y', 'cluster']
    sns.lmplot(data=data_frame, x='x', y='y', hue='cluster', fit_reg=False, legend=True, legend_out=True)
    plt.show()
    plt.savefig('EMforGM_iteration' + str(i) +".png")

def main():
    data = np.load('dataset.npy') # path of data file
    N = data.shape[0]           # number of rows
    D = data.shape[1]           # number of columns -> dimension of data
    K = 3                       # number of distributions/clusters
    max_iteration_number = 200  # number of maximum iteration
    convergence_value = 0.01    # negligible convergence value

    means = [[random(), random()], # mean of dimension 1, dimension 2 for distribution 1
             [random(), random()], # mean of dimension 1, dimension 2 for distribution 2
             [random(), random()]] # mean of dimension 1, dimension 2 for distribution 3

    covariances = [[[1., 0.],[0., 1.]], # covariance matrix of distribution 1
                   [[1., 0.],[0., 1.]], # covariance matrix of distribution 2
                   [[1., 0.],[0., 1.]]] # covariance matrix of distribution 3

    mixing_coefficients = np.zeros(K)
    mixing_coefficients[0] = 1/K # weight of distribution 1
    mixing_coefficients[1] = 1/K # weight of distribution 2
    mixing_coefficients[2] = 1/K # weight of distribution 3

    print("Initial Values: ")
    print("means = ", means)
    print("covariances = ", covariances)
    print("mixing_coefficients: ", mixing_coefficients)

    log_likelihoods = []
    current_log_likelihood = logLikelihoodCalculation(data, K, N, means, covariances, mixing_coefficients)
    log_likelihoods.append(current_log_likelihood)

    i=0
    while(i<max_iteration_number):
        gamma = expectationStep(data, K, N, means, covariances, mixing_coefficients)
        means, covariances, mixing_coefficients = maximizationStep(data, K, N, D, gamma)
        current_log_likelihood = logLikelihoodCalculation(data, K, N, means, covariances, mixing_coefficients)
        log_likelihoods.append(current_log_likelihood)
        print('Iteration: ', i, ' - Log likelihood value: ', current_log_likelihood)
        if abs(log_likelihoods[-1] - log_likelihoods[-2])<convergence_value:
            break

        # Use if plots for some intermediate iterations are needed
        # if i%5==0:
        #    predicted_k = prediction(data, K, N, means, covariances, mixing_coefficients)
        #    plotting(i, data, predicted_k)

        i = i + 1

    predicted_k = prediction(data, K, N, means, covariances, mixing_coefficients)
    plotting(i, data, predicted_k)

    print("Final Values: ")
    print("means = ", means)
    print("covariances = ", covariances)
    print("mixing_coefficients: ", mixing_coefficients)

if __name__ == '__main__':
    main()