import pickle
import numpy as np
import matplotlib.pyplot as plt
import math


def aggregate_results(results_data, ground_truth_A, ground_truth_D):
    """Aggregate estimated A and GGT values from a dictionary where the keys are iteration numbers.
       Averages results for each iteration across 10 experiment iterates."""
    num_iterations = 30
    # d = results_data[1]['initial D'].shape[0]
    # est_A_list = [np.zeros((d,d))]
    # initial_GGT_values = []
    # for key in sorted(results_data.keys()):
    #     print(f'initial X0: {results_data[key]['X0 points']}')
    #     initial_GGT_values.append(results_data[key]['initial D'])
    #     print(results_data[key]['initial D'])
    # est_GGT_list = [np.mean(initial_GGT_values)]
    A_mean_maes = []
    D_mean_maes = []
    A_mae_std_errs = []
    D_mae_std_errs = []


    for iteration in range(num_iterations):
        A_maes = []
        D_maes = []

        # Loop through the experiment replicates
        for key in sorted(results_data.keys()):
            A = results_data[key]['est A values'][iteration]
            A_maes.append(compute_mae(A, ground_truth_A))
            D = results_data[key]['est D values'][iteration]
            D_maes.append(compute_mae(D, ground_truth_D))

            # print(results_data[4]['initial D'])
            # print(results_data[4]['X0 points'])

        # Compute the average for the current iteration
        avg_A_mae = np.mean(A_maes)
        avg_D_mae = np.mean(D_maes)
        print(f'average MAE for estimated A at iteration {iteration}: {avg_A_mae}')
        print(f'average MAE estimated D at iteration {iteration}: {avg_D_mae}')
        A_mean_maes.append(avg_A_mae)
        D_mean_maes.append(avg_D_mae)
        A_mae_std_errs.append( np.std(A_maes / np.sqrt(len(results_data.keys()))))
        D_mae_std_errs.append(np.std(D_maes / np.sqrt(len(results_data.keys()))))

    return A_mean_maes, A_mae_std_errs, D_mean_maes, D_mae_std_errs


def compute_mae(estimated, ground_truth):
    """Compute Mean Absolute Percentage Error (MAE)"""
    mae = np.mean(np.abs((estimated - ground_truth)))
    return mae

# def compute_mape(estimated, ground_truth):
#     """Compute the 90th Percentile of the Mean Absolute Percentage Error (MAPE)"""
#     absolute_percentage_errors = np.abs((estimated - ground_truth) / ground_truth) * 100
#     return np.percentile(absolute_percentage_errors, 10)

def plot_mae_vs_iterations(results_data_version1, ground_truth_A1, ground_truth_GGT1,
                            results_data_version2=None, ground_truth_A2=None, ground_truth_GGT2=None):
    # Aggregate the estimated A and GGT values from version 1
    A_mean_maes_1, A_mae_std_errs_1, D_mean_maes_1, D_mae_std_errs_1 = aggregate_results(results_data_version1,
                                                                                         ground_truth_A1,
                                                                                         ground_truth_GGT1)

    iterations = np.arange(1, len(A_mean_maes_1) + 1)

    # Plot the MAPE for drift (A) and diffusion (GGT) for both versions with error bars
    plt.figure(figsize=(10, 6))

    # Version 1 (with error bars)
    plt.errorbar(iterations, A_mean_maes_1, yerr=A_mae_std_errs_1, label='A1 (Version 1)', color='black', linestyle='-',
                 marker='o')
    plt.errorbar(iterations, D_mean_maes_1, yerr=D_mae_std_errs_1, label='G1 (Version 1)', color='black', linestyle=':',
                 marker='o')

    # Version 2 (if provided)
    if results_data_version2 is not None:
        # Aggregate results for version 2
        A_mean_maes_2, A_mae_std_errs_2, D_mean_maes_2, D_mae_std_errs_2 = aggregate_results(results_data_version2,
                                                                                             ground_truth_A2,
                                                                                             ground_truth_GGT2)

        # Version 2 (with error bars)
        plt.errorbar(iterations, A_mean_maes_2, yerr=A_mae_std_errs_2, label='A2 (Version 2)', color='red',
                     linestyle='-', marker='D')
        plt.errorbar(iterations, D_mean_maes_2, yerr=D_mae_std_errs_2, label='G2 (Version 2)', color='red',
                     linestyle=':', marker='D')

    # Customize the plot
    plt.xlabel('Iteration')
    plt.ylabel('MAE')
    # plt.yscale('log')  # Uncomment if you want a log scale
    plt.legend()
    plt.grid(True)
    plt.title('MAE of Estimated Parameters vs Iterations')
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
        d=2
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


def plot_exp_results(exp_number, version, num_reps=10):
    results_data_global = {}
    ground_truth_A1, ground_truth_GGT1 = retrieve_true_A_D(exp_number, version)
    for i in range(1, num_reps+1):
        filename = f'Results_experiment_{exp_number}/version-{version}_replicate-{i}.pkl'
        with open(filename, 'rb') as f:
            results_data = pickle.load(f)
        results_data_global[i]=results_data
    plot_mae_vs_iterations(results_data_global, ground_truth_A1, ground_truth_GGT1)

plot_exp_results(exp_number = 3, version = 1, num_reps=10)
# plot_exp_results(exp_number = 1, version = 2)
# plot_exp_results(exp_number = 3, version = 1)
# plot_exp_results(exp_number = 3, version = 2)
# plot_exp_results(exp_number = 3, version = 2)

