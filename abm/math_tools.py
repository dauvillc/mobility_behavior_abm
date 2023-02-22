"""
Clément Dauvilliers - EPFL TRANSP-OR Lab - 30/11/2022
Defines mathematical tools such as the sigmoid function.
"""
import numpy as np


def sigmoid(x):
    """
    Usual sigmoid (logistic) function.
    """
    return 1 / (1 + np.exp(-x))


def compute_lognormal_params(mean, std):
    """
    Given the mean and standard deviation of a lognormal
    distribution, computes the mean and std of the underlying
    normal distribution.
    Parameters
    ----------
    mean: mean of the lognormal distrib.
    std: std of the lognormal distrib.

    Returns
    -------
    (mu, sigma): mu is the mean of the underlying normal distrib,
        sigma is its standard deviation.
    """
    mean_n  = np.log(mean ** 2 / np.sqrt(std ** 2 + mean ** 2))
    std_n = np.sqrt(np.log(std ** 2 / mean ** 2 + 1))
    return mean_n, std_n
