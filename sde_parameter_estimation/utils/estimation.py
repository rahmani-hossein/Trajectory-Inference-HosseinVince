import numpy as np
from scipy.spatial.distance import mahalanobis
from scipy.stats import chi2
from tqdm import tqdm
import math


def extract_marginal_samples(trajectories, shuffle=False):
    """
    Extract marginal distributions per time from a 3D trajectory array.

    Parameters:
        trajectories (numpy.ndarray): 3D array of trajectories (num_trajectories, num_steps, d).

    Returns:
        list of numpy.ndarray: Each element is an array containing samples from the marginal distribution at each time step.
    """
    num_trajectories, num_steps, d = trajectories.shape
    marginal_samples = []

    for t in range(num_steps):
        # Extract all samples at time t from each trajectory
        samples_at_t = trajectories[:, t, :]
        if shuffle:
            samples_at_t_copy = samples_at_t.copy()
            np.random.shuffle(samples_at_t_copy)
            marginal_samples.append(samples_at_t_copy)
        else:
            marginal_samples.append(samples_at_t)

    return marginal_samples

def flatten_trajectories_sequentially(trajectories):
    """
    Flatten the 3D trajectory array into a 2D array ordered sequentially by time steps.

    Parameters:
        trajectories (numpy.ndarray): 3D array of trajectories (num_trajectories, num_steps, d).

    Returns:
        numpy.ndarray: 2D array where rows are samples ordered first by time step, then by trajectory index.
    """
    num_trajectories, num_steps, d = trajectories.shape
    # Initialize the flattened array
    flattened = np.zeros((num_trajectories * num_steps, d))

    # Fill the flattened array
    for t in range(num_steps):
        flattened[t*num_trajectories:(t+1)*num_trajectories] = trajectories[:, t, :]

    return flattened

def set_probabilities(observations, num_trajectories, t, frac_other_time_samples):
    """
    Set probabilities for distributions a and b with special emphasis on time steps t and t+1.

    Parameters:
        observations (numpy.ndarray): Flattened array of observations sorted by time.
        num_trajectories (int): Number of trajectories per time step.
        t (int): Current time step of interest.
        frac_other_time_samples (float): Fraction of probability mass for other times.

    Returns:
        tuple: Two numpy arrays representing the probability distributions a and b.
    """
    num_samples = observations.shape[0]

    # Initialize probability distributions
    a = np.full(num_samples, frac_other_time_samples / (num_samples - num_trajectories))
    b = np.full(num_samples, frac_other_time_samples / (num_samples - num_trajectories))

    # Adjust probabilities for time step t and t+1
    a[t * num_trajectories:(t + 1) * num_trajectories] = (1 - frac_other_time_samples) / num_trajectories
    b[(t + 1) * num_trajectories:(t + 2) * num_trajectories] = (1 - frac_other_time_samples) / num_trajectories

    return a, b



def estimate_gaussian_marginal(X_t):
    """ Estimate Gaussian parameters from samples. """
    fit_mean = np.mean(X_t, axis=0)
    fit_cov = np.cov(X_t, rowvar=False)
    return fit_mean, fit_cov

def multiply_matrices(matrix_list):
    # Start with the first matrix
    result = matrix_list[0]

    # Sequentially multiply the matrices
    for matrix in matrix_list[1:]:
        result = np.matmul(result, matrix)
    return result



def gaussian_outer_product(fit_mean, fit_cov):
    """ Compute the outer product using Gaussian parameters. """
    d = len(fit_mean)
    outer_prod = np.outer(fit_mean, fit_mean) + fit_cov
    return outer_prod


def calculate_weights(X_t, fit_mean, fit_cov):
    """ Calculate weights for each sample based on Mahalanobis distance. """
    inv_covmat = np.linalg.inv(fit_cov)
    weights = np.array([np.exp(-0.5 * mahalanobis(x, fit_mean, inv_covmat) ** 2) for x in X_t])
    weights /= np.sum(weights)  # Normalize the weights
    return weights





