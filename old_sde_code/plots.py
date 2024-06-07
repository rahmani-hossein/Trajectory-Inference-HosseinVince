import matplotlib
import numpy as np

matplotlib.use('TkAgg')  # Use TkAgg backend for interactive display
import matplotlib.pyplot as plt

def plot_trajectories(X, T, dt):
    """
    Plot the trajectories of a multidimensional process.

    Parameters:
        X (numpy.ndarray): Array of trajectories.
        T (float): Total time period.
        dt (float): Time step size.
    """
    num_dimensions = X.shape[1]
    time_steps = np.linspace(0, T, X.shape[0])  # Generate time steps corresponding to [0, T]

    # Plot trajectories
    plt.figure(figsize=(12, 8))
    for d in range(num_dimensions):
        plt.plot(time_steps, X[:, d], label=f'X_{d+1}')


    # plt.title('Manten path dependent example', fontsize=20)
    plt.xlabel('Time', fontsize=16)
    plt.ylabel('Value', fontsize=16)
    plt.legend(fontsize=14)
    plt.grid(True)

    # Show plot
    plt.show()
def plot_covariance_functions(X, T, dt, A, G):
    """
    Plot the empirical and theoretical autocovariance functions for each dimension.

    Parameters:
        X (numpy.ndarray): Array of trajectories.
        T (float): Total time period.
        dt (float): Time step size.
        A (numpy.ndarray): Drift matrix.
        G (numpy.ndarray): Variance matrix.
    """
    num_dimensions = X.shape[1]
    n_steps = len(X)
    time_steps = np.arange(1, n_steps) * dt

    plt.figure(figsize=(12, 8))
    for d in range(num_dimensions):
        # Empirical Autocovariance
        autocovariance = [np.cov(X[:-i, d], X[i:, d], ddof=0, bias=True)[0, 1] for i in range(1, n_steps)]
        plt.plot(time_steps, autocovariance, label=f'Empirical Cov($X_{d+1}$, $X_{d+1}$)')

        # Theoretical Covariance using linearly increasing variance
        # theoretical_cov = [np.exp(A[d, d] * h) * 0.01 * h for h in time_steps]
        # plt.plot(time_steps, theoretical_cov, linestyle='--', label=f'Theoretical Cov($X_{d+1}$, $X_{d+1}$)')

    plt.title('Empirical vs. Theoretical Autocovariance Functions', fontsize=20)
    plt.xlabel('Time gap (h)', fontsize=16)
    plt.ylabel('Covariance', fontsize=16)
    plt.legend(fontsize=14)
    plt.grid(True)
    plt.show()