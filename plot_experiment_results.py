import pickle
import numpy as np
import matplotlib.pyplot as plt
import math


def aggregate_results(results_data):
    """Aggregate estimated A and GGT values from a dictionary where the keys are iteration numbers.
       Averages results for each iteration across 10 experiment iterates."""
    num_iterations = 30
    d = results_data[1]['initial D'].shape[0]
    est_A_list = [np.zeros((d,d))]
    initial_GGT_values = []
    for key in sorted(results_data.keys()):
        initial_GGT_values.append(results_data[key]['initial D'])
    est_GGT_list = [np.mean(initial_GGT_values)]


    for iteration in range(num_iterations):
        A_values = []
        GGT_values = []

        # Loop through the experiment iterates
        for key in sorted(results_data.keys()):

            # Collect the values for the current iteration across all experiments
            A_values.append(results_data[key]['est A values'][iteration])
            GGT_values.append(results_data[key]['est D values'][iteration])


            # print(results_data[4]['initial D'])
            # print(results_data[4]['X0 points'])

        # Compute the average for the current iteration
        est_A_list.append(np.mean(A_values))
        est_GGT_list.append(np.mean(GGT_values))

    return est_A_list, est_GGT_list


def compute_mape(estimated, ground_truth):
    """Compute Mean Absolute Percentage Error (MAE)"""
    return np.mean(np.abs((estimated - ground_truth)))

# def compute_mape(estimated, ground_truth):
#     """Compute the 90th Percentile of the Mean Absolute Percentage Error (MAPE)"""
#     absolute_percentage_errors = np.abs((estimated - ground_truth) / ground_truth) * 100
#     return np.percentile(absolute_percentage_errors, 10)

def plot_mape_vs_iterations(results_data_version1, ground_truth_A1, ground_truth_GGT1,
                            results_data_version2=None, ground_truth_A2=None, ground_truth_GGT2=None):
    # Aggregate the estimated A and GGT values from both versions
    est_A_list_v1, est_GGT_list_v1 = aggregate_results(results_data_version1)


    # Compute MAPE for each iteration for both versions
    mape_A_v1 = [compute_mape(est_A, ground_truth_A1) for est_A in est_A_list_v1]
    mape_GGT_v1 = [compute_mape(est_GGT, ground_truth_GGT1) for est_GGT in est_GGT_list_v1]
    iterations = np.arange(1, len(mape_A_v1) + 1)


    # Plot the MAPE for drift (A) and diffusion (GGT) for both versions
    plt.figure(figsize=(10, 6))

    # Version 1
    plt.plot(iterations, mape_A_v1, label='A1 (Version 1)', color='black', linestyle='-', marker='o')
    plt.plot(iterations, mape_GGT_v1, label='G1 (Version 1)', color='black', linestyle=':', marker='o')
    if results_data_version2 is not None:
        est_A_list_v2, est_GGT_list_v2 = aggregate_results(results_data_version2)
        mape_A_v2 = [compute_mape(est_A, ground_truth_A2) for est_A in est_A_list_v2]
        mape_GGT_v2 = [compute_mape(est_GGT, ground_truth_GGT2) for est_GGT in est_GGT_list_v2]
        # Version 2
        plt.plot(iterations, mape_A_v2, label='A2 (Version 2)', color='red', linestyle='-', marker='D')
        plt.plot(iterations, mape_GGT_v2, label='G2 (Version 2)', color='red', linestyle=':', marker='D')

    # Customize the plot
    plt.xlabel('Iteration')
    plt.ylabel('MAE')
    # plt.yscale('log')
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


def plot_exp_results(exp_number, version, num_reps=8):
    results_data_global = {}
    ground_truth_A1, ground_truth_GGT1 = retrieve_true_A_D(exp_number, version)
    for i in range(1, num_reps+1):
        filename = f'Results_experiment_{exp_number}/version-{version}_replicate-{i}.pkl'
        with open(filename, 'rb') as f:
            results_data = pickle.load(f)
        results_data_global[i]=results_data
    plot_mape_vs_iterations(results_data_global, ground_truth_A1, ground_truth_GGT1)

plot_exp_results(exp_number = 3, version = 1)

