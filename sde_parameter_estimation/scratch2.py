import numpy as np
import ot
from plots import *
import matplotlib.pyplot as plt
from utils import *
import pickle
import math
from parameter_estimation import *
from simulate_trajectories import *
import matplotlib.pyplot as plt
import os



def analyze_X_OT(marginal_samples, dt, entropy_reg, num_trajectories, d, use_raw_avg=True, filename=None):
    if filename is None:
        filename = f'N-{num_trajectories}_reg_{entropy_reg}_d-{d}_transport_map.npy'

    p = np.load(filename)

    num_time_steps = len(marginal_samples)
    X_OT = np.zeros((num_trajectories, num_time_steps, d))
    X_OT[:, 0, :] = marginal_samples[0]  # Initial condition

    for t in range(num_time_steps - 1):
        if t == 0 or not use_raw_avg:
            X_t = marginal_samples[t]
            X_t1 = marginal_samples[t + 1]
        else:
            X_t = X_t1_OT
            X_t1 = marginal_samples[t + 1]

        if use_raw_avg:
            p_normalized = normalize_rows(p)
            X_t1_OT = np.dot(p_normalized, X_t1)
            X_OT[:, t + 1, :] = X_t1_OT

    return X_OT

def compute_distances(real_traj, traj):
    return np.linalg.norm(real_traj - traj, axis=1)


def plot_comparison(X, X_GWOT, trajectory_index=0):
    """
    Plot the true trajectory vs. OT-predicted trajectories for different entropy regularizations.

    Parameters:
        X (numpy.ndarray): True trajectories.
        X_GWOT (numpy.ndarray): GWOT-predicted trajectories
        trajectory_index (int): Index of the trajectory to plot.
    """
    num_time_steps, d = X.shape[1], X.shape[2]

    plt.figure(figsize=(10, 6))
    for dim in range(d):
        plt.subplot(d, 1, dim + 1)
        plt.plot(np.arange(num_time_steps), X[trajectory_index, :, dim], 'k-',
                 label='True Trajectory' if dim == 0 else "")
        plt.plot(np.arange(num_time_steps), X_GWOT[trajectory_index, :, dim], 'r--',
                 label='GWOT Trajectory' if dim == 0 else "")
        plt.xlabel('Time Step')
        plt.ylabel(f'Trajectory Value (Dim {dim + 1})')
        plt.title(f'Trajectory {trajectory_index}, Dimension {dim + 1}')
        if dim == 0:
            plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    A = -1
    T = 1
    G = 1
    N_= 20
    X = np.load(f'X0-1_paths_nullgrowth_d-1_from_N-20_A-[[1]]_G-1_dt-0.01.npy')
    dt = T/X.shape[1]
    print(dt)
    for N in [500, 1000, 5000]:
        X_ = X[:N, :, :]
         #ou_process(T, dt, A_, G, X0)
        # plot_trajectories(X, T, dt)
        # plot_trajectories(X_[0], T, dt)
        for j in range(10):
            X0 = X_[j, 0, :]
            A_ = np.array([[A]])
            G_ = np.array([[G]])
            # X = multiple_ou_trajectories(N, 1, T, dt, A_, G_, X0=X0)
            # print(X.shape)
            # plot_comparison(X, X_, j)


        A_est = estimate_A_exp(X_, dt)
        print('GWOT estimated A:', A_est[0][0])
        # print('MSE:',  np.mean((A_est - A) ** 2))
        print(f'A bias for {N} GWOT trajectories:', A_est[0][0]-A)
        G_est = estimate_GGT(X_, T)
        print('GWOT estimated G^2:', G_est[0][0])
        # print(G_est)
        print(f'G bias for {N} GWOT trajectories:',G_est[0][0]-G**2)

    # # Example usage
    # dt = 0.02
    # entropy_reg = 0.01
    # num_trajectories = 100
    # d = 1
    # filename = f'seed-2_X0-intermediate_d-{d}_n_sdes-50_dt-0.02_N-1000_T-1.0'
    # A_trues, G_trues, maximal_X_measured_list, max_num_trajectories, max_T, min_dt = load_measurement_data(filename)
    # X = maximal_X_measured_list[0]
    # X = X[:num_trajectories, :, :]
    # marginal_samples = extract_marginal_samples(X, shuffle = False)
    # use_raw_avg = True
    # filename = f'N-{num_trajectories}_reg_0.01_d-{d}_transport_map.npy'
    # X_OT_reg = analyze_X_OT(marginal_samples, dt, entropy_reg, num_trajectories, d, use_raw_avg, filename)
    # filename = f'N-{num_trajectories}_d-{d}_transport_map.npy'
    # X_OT = analyze_X_OT(marginal_samples, dt, entropy_reg, num_trajectories, d, use_raw_avg, filename)
    # n_steps = 5
    # l=0
    # for i in range(n_steps):
    #     real_traj = X[l, i]
    #     OT_traj = X_OT[l, i]
    #     reg_OT_traj = X_OT_reg[l, i]
    #
    #     distance_OT = np.linalg.norm(real_traj - OT_traj)
    #     distance_reg_OT = np.linalg.norm(real_traj - reg_OT_traj)
    #
    #     print(f'Step {i}:')
    #     print('real:', real_traj)
    #     print('OT:', OT_traj)
    #     print('OT_reg:', reg_OT_traj)
    #     print('Distance to OT traj:', distance_OT)
    #     print('Distance to reg OT traj:', distance_reg_OT)

# p = np.load('N-25_reg_0.01_d-10_transport_map.npy')
    # for i in range(25):
    #     print(p[i].sum())

