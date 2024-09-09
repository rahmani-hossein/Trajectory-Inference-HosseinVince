import pickle
import numpy as np
import matplotlib.pyplot as plt
import math


def aggregate_results(results_data, ground_truth_A_list, ground_truth_D_list):
    """Aggregate estimated A and GGT values from a dictionary where the keys are iteration numbers.
       Averages results for each iteration across 10 experiment iterates and computes correlations."""
    num_iterations = 30

    # Lists to store the results
    A_mean_maes = []
    D_mean_maes = []
    A_mae_std_errs = []
    D_mae_std_errs = []

    # Lists to store correlations for each iteration
    A_correlations = []
    D_correlations = []

    for iteration in range(num_iterations):
        A_maes = []
        D_maes = []
        A_corrs = []
        D_corrs = []

        # Loop through the experiment replicates
        for key in sorted(results_data.keys()):
            ground_truth_A = ground_truth_A_list[key - 1]
            ground_truth_D = ground_truth_D_list[key - 1]

            # Retrieve the estimated values for A and D at the current iteration
            A = results_data[key]['est A values'][iteration]
            D = results_data[key]['est D values'][iteration]

            # Compute MAE
            A_maes.append(compute_mae(A, ground_truth_A))
            D_maes.append(compute_mae(D, ground_truth_D))

            # Compute Correlations
            A_corr = calculate_correlation(A, ground_truth_A)
            D_corr = calculate_correlation(D, ground_truth_D)
            A_corrs.append(A_corr)
            D_corrs.append(D_corr)

        # Compute the average MAE and correlations for the current iteration
        avg_A_mae = np.mean(A_maes)
        avg_D_mae = np.mean(D_maes)
        A_mean_maes.append(avg_A_mae)
        D_mean_maes.append(avg_D_mae)

        # Compute standard error
        A_mae_std_errs.append(np.std(A_maes) / np.sqrt(len(results_data.keys())))
        D_mae_std_errs.append(np.std(D_maes) / np.sqrt(len(results_data.keys())))

        # Compute average correlations
        avg_A_corr = np.mean(A_corrs)
        avg_D_corr = np.mean(D_corrs)
        A_correlations.append(avg_A_corr)
        D_correlations.append(avg_D_corr)

    return A_mean_maes, A_mae_std_errs, D_mean_maes, D_mae_std_errs, A_correlations, D_correlations


def compute_mae(estimated, ground_truth):
    """Compute Mean Absolute Percentage Error (MAE)"""
    mae = np.mean(np.abs((estimated - ground_truth)))
    return mae


def calculate_correlation(estimated_matrix, ground_truth_matrix):
    """Calculates the correlation between two matrices."""
    # Flatten the matrices to 1D arrays and calculate the Pearson correlation
    estimated_flat = estimated_matrix.flatten()
    ground_truth_flat = ground_truth_matrix.flatten()
    correlation = np.corrcoef(estimated_flat, ground_truth_flat)[0, 1]
    return correlation



def plot_mae_and_correlation_vs_iterations(results_data_version1, ground_truth_A1_list, ground_truth_GGT1_list, exp_title = None):
    # Aggregate the estimated A and GGT values from version 1
    A_mean_maes_1, A_mae_std_errs_1, D_mean_maes_1, D_mae_std_errs_1, A_correlations_1, D_correlations_1 = aggregate_results(
        results_data_version1,
        ground_truth_A1_list,
        ground_truth_GGT1_list)

    iterations = np.arange(1, len(A_mean_maes_1) + 1)

    # Plot the MAE for drift (A) and diffusion (GGT) for both versions with error bars
    plt.figure(figsize=(10, 6))

    # Version 1 (with error bars)
    plt.errorbar(iterations, A_mean_maes_1, yerr=A_mae_std_errs_1, label='MAE between estimated A and true A', color='black',
                 linestyle='-', marker='o')
    plt.errorbar(iterations, D_mean_maes_1, yerr=D_mae_std_errs_1, label='MAE between estimated H and true H', color='black',
                 linestyle=':', marker='o', markerfacecolor='none', markeredgecolor='black')

    # Customize the MAE plot
    plt.xlabel('Iteration')
    plt.ylabel('MAE')
    plt.legend()
    plt.grid(True)
    if exp_title is not None:
        plt.title(f'MAE of Estimated Parameters vs Iterations for {exp_title }')
    else:
        plt.title(f'MAE of Estimated Parameters vs Iterations')
    plt.show()

    # Create a second plot for correlations
    plt.figure(figsize=(10, 6))

    # Plot correlation for version 1
    plt.errorbar(iterations, A_correlations_1, label='Correlation between estimated A and true A', color='black', linestyle='-', marker='o')
    plt.errorbar(iterations, D_correlations_1, label='Correlation between estimated H and true H', color='black', linestyle=':', marker='o', markerfacecolor='none', markeredgecolor='black')


    # Customize the correlation plot
    plt.xlabel('Iteration')
    plt.ylabel('Correlation')
    plt.legend()
    plt.grid(True)
    if exp_title is not None:
        plt.title(f'Correlation of Estimated Parameters vs Iterations for {exp_title }')
    else:
        plt.title(f'Correlation of Estimated Parameters vs Iterations')
    plt.show()



def retrieve_true_A_D(exp_number, version):
    if exp_number == 1:
        if version == 1:
            A = np.array([[-1]])
            G = np.eye(1)
        else:
            A = np.array([[-10]])
            G = math.sqrt(10) * np.eye(1)
    elif exp_number == 2:
        d = 2
        if version == 1:
            A = np.zeros((d, d))
        else:
            A = np.array([[0, 1], [-1, 0]])
        G = np.eye(d)
    elif exp_number == 3:
        d = 2
        if version == 1:
            A = np.array([[1, 2], [1, 0]])
        else:
            A = np.array([[1 / 3, 4 / 3], [2 / 3, -1 / 3]])
        G = np.array([[1, 2], [-1, -2]])

    return A, np.matmul(G, G.T)


def plot_exp_results(exp_number, version=None, d=None, num_reps=10):
    results_data_global = {}
    ground_truth_A_list = []
    ground_truth_D_list = []
    for i in range(1, num_reps + 1):
        if exp_number != "random":
            filename = f'Results_experiment_{exp_number}/version-{version}_replicate-{i}.pkl'
        else:
            filename = f'Results_experiment_{exp_number}_{d}/replicate-{i}.pkl'
        with open(filename, 'rb') as f:
            results_data = pickle.load(f)
        if exp_number == 'random':
            ground_truth_A_list.append(results_data['true_A'])
            ground_truth_D_list.append(results_data['true_D'])

        results_data_global[i] = results_data

    if exp_number != 'random':
        ground_truth_A1, ground_truth_GGT1 = retrieve_true_A_D(exp_number, version)
        ground_truth_A_list = [ground_truth_A1] * num_reps
        ground_truth_D_list = [ground_truth_GGT1] * num_reps

    if exp_number == 'random':
        plot_mae_and_correlation_vs_iterations(results_data_global, ground_truth_A_list, ground_truth_D_list, exp_title=f'random SDEs of dimension {d}')
    else:
        plot_mae_and_correlation_vs_iterations(results_data_global, ground_truth_A_list, ground_truth_D_list,
                                               exp_title=f'SDE {version} from example {exp_number}')

# plot_exp_results(exp_number='random', d=50, num_reps=10)
plot_exp_results(exp_number = 2, version = 1, num_reps=10)
plot_exp_results(exp_number = 2, version = 2, num_reps=10)
# plot_exp_results(exp_number = 1, version = 2)
# plot_exp_results(exp_number = 3, version = 1)
# plot_exp_results(exp_number = 3, version = 2)
# plot_exp_results(exp_number = 3, version = 2)
