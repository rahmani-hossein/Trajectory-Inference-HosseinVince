import matplotlib
import numpy as np
import os

matplotlib.use('TkAgg')  # Use TkAgg backend for interactive display
import matplotlib.pyplot as plt


def plot_MSE(ablation_values, ablation_variable_name, list_mse_scores, list_std_errs, list_method_labels, d,
             experiment_name, save_plot = True):
    """
    Plot and save Mean Squared Error (MSE) results.

    Parameters:
    - ablation_values: List of values for the ablation variable.
    - ablation_variable_name: Name of the ablation variable (string).
    - list_mse_scores: List of lists containing MSE scores for each method.
    - list_std_errs: List of lists containing standard errors for each method.
    - list_method_labels: List of method names corresponding to the MSE scores.
    - d: Number of dimensions for the experiment.
    - experiment_name: Name of the experiment (string).
    - save_plot: Boolean indicating whether to save the plot or not. Defaults to True.
    """
    fmt_list = ['-^', '-o', '--', '-s', '-d', '-x', '-*']
    for method_name, mse_score, std_err, fmt in zip(list_method_labels, list_mse_scores, list_std_errs, fmt_list):
        plt.errorbar(ablation_values, mse_score, yerr=std_err, fmt=fmt, label=method_name)
    plt.xlabel(ablation_variable_name)
    plt.ylabel('Mean Squared Error (MSE)')
    plt.title(f'Parameter Estimation on {d}-dimensional Stationary Linear Additive Noise SDE')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    # Save the plot
    if save_plot:
        os.makedirs('../MSE_plots', exist_ok=True)
        plot_filename = f"mse_plot_{experiment_name}.png"
        filepath = os.path.join('../MSE_plots', plot_filename)
        plt.savefig(filepath)

    # Show plot
    plt.show()

def plot_trajectories(X, T, dt, save_file = False):
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
    plt.tight_layout()
    if save_file:
        os.makedirs('Raw_trajectory_figures', exist_ok=True)
        plot_filename = os.path.join('Raw_trajectory_figures', f"raw_trajectory_d-{num_dimensions}_stationary.png")
        plt.savefig(plot_filename)
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