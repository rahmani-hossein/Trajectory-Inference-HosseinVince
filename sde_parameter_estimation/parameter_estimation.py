import numpy as np
import ot
from scipy.linalg import expm
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt
import random


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


def estimate_params_compare_methods(X, dt, T, entropy_reg, methods, n_iterations=1, frac_other_time_samples = None):
    """
    Estimate A using various methods.

    Parameters:
    - X (numpy.ndarray): 3D array of trajectories (num_trajectories, num_steps, d).
    - dt: Time step.
    - entropy_reg: Entropy regularization parameter.
    - methods: List of method names to use for estimation.

    Returns:
    - Dictionary of estimated A matrices keyed by method name.
    - Dictionary of estimated G matrices keyed by method name.
    """
    A_estimations = {}
    G_estimations = {}
    # Define the estimation functions for each method
    for method in methods:
        if method == 'Trajectory':
            A_estimations[method] = estimate_linear_drift(X, dt, expectation=True, OT=False, entropy_reg=0, GGT=None)
            G_estimations[method] = estimate_GGT(X, T)
        elif method == 'OT':
            est_A, est_G = estimate_linear_drift(X, dt, expectation=True, OT=True, entropy_reg=0, GGT=None,
                                                          n_iterations=n_iterations, metric = 'euclidean', frac_other_time_samples = frac_other_time_samples)
            A_estimations[method] = est_A
            G_estimations[method] = est_G
        elif method == 'OT reg':
            est_A, est_G = estimate_linear_drift(X, dt, expectation=True, OT=True,
                                                          entropy_reg=entropy_reg * dt,
                                                          GGT=None, n_iterations=n_iterations,
                                                          metric = 'sqeuclidean', frac_other_time_samples = frac_other_time_samples)
            A_estimations[method] = est_A
            G_estimations[method] = est_G
        elif method == 'Classical':
            A_estimations[method] = estimate_linear_drift(X, dt, expectation=False, GGT=None)
            N = X.shape[0]
            est_G_list = [estimate_GGT(X, T) for _ in range(N)]
            G_estimations[method] = np.mean(est_G_list, axis=0)
        else:
            raise ValueError(f"Unsupported method: {method}")

    return A_estimations, G_estimations


# def estimate_G_compare_methods(X, T, entropy_reg, methods, n_iterations=1, frac_other_time_samples = None):
#     """
#     Estimate G using various methods.
#
#     Parameters:
#     - X (numpy.ndarray): 3D array of trajectories (num_trajectories, num_steps, d).
#     - T
#     - entropy_reg: Entropy regularization parameter.
#     - methods: List of method names to use for estimation.
#
#     Returns:
#     - Dictionary of estimated G matrices keyed by method name.
#     """
#
#     G_estimations = {}
#     # Define the estimation functions for each method
#     for method in methods:
#         if method == 'Trajectory':
#             G_estimations[method] = estimate_GGT(X, T)
#         elif method == 'OT':
#
#             G_estimations[method] =
#         elif method == 'OT reg':
#             G_estimations[method] = estimate_linear_drift(X, dt, expectation=True, OT=True,
#                                                           entropy_reg=entropy_reg * dt,
#                                                           GGT=None, n_iterations=n_iterations,
#                                                           metric = 'sqeuclidean', frac_other_time_samples = frac_other_time_samples)
#         elif method == 'Classical':
#             G_estimations[method] = estimate_linear_drift(X, dt, expectation=False, GGT=None)
#         else:
#             raise ValueError(f"Unsupported method: {method}")
#
#     return G_estimations


def estimate_linear_drift(X, dt, expectation=True, OT=True, entropy_reg=0, GGT=None, n_iterations=1, metric = 'euclidean', frac_other_time_samples = 0):
    '''
    we assume that the SDE is multivariable OU: dX_t = AX_tdt + GdW_t
    This function serves to estimate the drift A using a specified estimator
    :param X: 3D array where each slice corresponds to a single trajectory (num_trajectories, num_steps, d)
    :param dt: the time granularity of the observed time series, assumed to be equal
    :param expectation: boolean for whether to use expected values in the estimator
    :param OT:boolean for whether to use optimal transport to estimate the expected values
    :param entropy_reg: the epsilon value used for entropy regularization in the OT problem (0 by default)
    :param GGT: the (estimated) Gram matrix of the diffusion parameter, assuming additive noise
    :return:
    '''
    if expectation is True:
        # we estimate A using the closed form solution (with expectations)
        if OT is True:
            its = 1
            T = X.shape[1] * dt
            # initial estimate for A
            A_0, G_0  = estimate_A_exp_ot_with_traj(X, dt, T, entropy_reg = entropy_reg, metric = metric, frac_other_time_samples= frac_other_time_samples, estimate_G=True)

            #estimate_A_exp_ot_with_traj(marginals, dt, entropy_reg=entropy_reg, cur_est_A=None, metric = metric)
            # if entropy_reg == 0:
            #     print('classic OT traj:', X_OT[:, : 5, :])
            G = G_0
            A = A_0
            # print(f'estimated A for iteration 1:', A)
            while its < n_iterations:
                A, G = estimate_A_exp_ot_with_traj(X, dt, T, entropy_reg= entropy_reg, cur_est_A= A, metric = metric, frac_other_time_samples=frac_other_time_samples, estimate_G=True)
                #estimate_A_exp_ot_with_traj(marginals, dt, entropy_reg=entropy_reg, cur_est_A=A)                                         X0=None)
                its += 1
                # print(f'estimated A for iteration {its}:', A)

            return A, G
        else:
            # the expectations are taken over the set of all observed trajectories
            A = estimate_A_exp(X, dt)
    else:
        # we estimate A using the classical closed form solution (no expectations)
        A = estimate_A(X, dt, GGT=GGT)
    return A

def create_OT_traj(X, entropy_reg, dt, sinkhorn_log_thresh=0.0002, cur_est_A = None, metric = 'euclidean', frac_other_time_samples = 0):
    marginal_samples = extract_marginal_samples(X)
    num_time_steps = len(marginal_samples)
    d = marginal_samples[0].shape[1]
    num_trajectories = marginal_samples[0].shape[0]
    # transport plans
    ps = []
    for t in range(num_time_steps-1):
        if frac_other_time_samples == 0:
            # extract marginal samples
            X_t = marginal_samples[t]
            X_t1 = marginal_samples[t + 1]
            a = np.ones(len(X_t)) / len(X_t)
            b = np.ones(len(X_t1)) / len(X_t1)
        else:
            observations_sorted_by_time = flatten_trajectories_sequentially(X)
            X_t = observations_sorted_by_time
            X_t1 = observations_sorted_by_time
            a, b = set_probabilities(observations_sorted_by_time, num_trajectories, t, frac_other_time_samples)
        # create cost matrix
        if cur_est_A is None:
            # optimize over empirical marginal transition
            M = ot.dist(X_t, X_t1, metric='sqeuclidean')
            # if t == 1:
            #     print('cost matrix:', M)
        else:
            # optimize over empirical marginal transition given current estimated A
            mu = np.dot(X_t, expm(cur_est_A * dt))
            M = ot.dist(mu, X_t1, metric=metric)
            # M = ot.dist(X_t + np.dot(X_t, expm(cur_est_A * dt)), X_t1, metric=metric)

        if entropy_reg == 0:
            p = ot.emd(a= a, b= b, M=M)
        else:
            if cur_est_A is not None:
                D = entropy_reg / (2 * dt)
                cond_variance = D * np.linalg.pinv(cur_est_A) * (expm(2*cur_est_A * dt) - 1)
            else:
                cond_variance = entropy_reg
            if entropy_reg > sinkhorn_log_thresh:
                p = ot.sinkhorn(a=a, b=b, M=M/2,
                                reg=cond_variance, verbose=False)
            else:
                p = ot.sinkhorn(a=a, b=b, M=M/2,
                                reg=cond_variance, verbose=False, method='sinkhorn_log')
        ps.append(p)
    if entropy_reg == 0:
        N = num_trajectories
    else:
        N = max(5 * num_trajectories, 1000)
    X_OT = np.zeros(shape=(N, num_time_steps, d))
    OT_index_propagation = np.zeros(shape=(N, num_time_steps-1))
    if frac_other_time_samples == 0:
        indices = np.arange(num_trajectories)
    else:
        indices = np.arange(num_trajectories * num_time_steps)
    for _ in range(N):
        for t in range(num_time_steps-1):
            pt_normalized = normalize_rows(ps[t])
            if t == 0:
                if entropy_reg == 0:
                    k = _
                else:
                    k = np.random.randint(num_trajectories)
                X_OT[_, 0, :] = marginal_samples[0][k]
            else:
                # retrieve where _th observation at time 0 was projected to at time t
                k = int(OT_index_propagation[_, t-1])
            j = np.random.choice(indices, p=pt_normalized[k])
            OT_index_propagation[_, t] = int(j)
            if frac_other_time_samples == 0:
                X_OT[_, t + 1, :] = marginal_samples[t + 1][j]
            else:
                X_OT[_, t + 1, :] = observations_sorted_by_time[j]
    # if N < 10:
    #     print(X_OT[:, :3, :])
    return X_OT

def estimate_A_exp_ot_with_traj(X, dt, T=1, N_traj_sim = 1000, entropy_reg = 0.01, frac_other_time_samples = 0, sinkhorn_log_thresh=0.1, cur_est_A = None, metric = 'euclidean', estimate_G = False):
    X_OT = create_OT_traj(X, entropy_reg, dt, cur_est_A = cur_est_A, metric = metric, frac_other_time_samples = frac_other_time_samples)
    A_OT = estimate_A_exp(X_OT, dt)
    if estimate_G:
        G_OT = estimate_GGT(X_OT, T)
        return A_OT, G_OT
    else:
        return A_OT
def estimate_A_exp_ot(X, dt, entropy_reg=0.01, cur_est_A=None, use_raw_avg=True,
                      sinkhorn_log_thresh=0.1, return_OT_traj=False, pinv=False):
    """
    Estimate the drift matrix A using optimal transport between successive marginal distributions.

    Parameters:
        - X (numpy.ndarray): 3D array of trajectories (num_trajectories, num_steps, d).
        dt (float): Discretization time step.

    Returns:
        numpy.ndarray: Estimated drift matrix A
    """
    marginal_samples = extract_marginal_samples(X)
    num_time_steps = len(marginal_samples)
    d = marginal_samples[0].shape[1]
    num_trajectories = marginal_samples[0].shape[0]

    X_OT = np.zeros(shape=(num_trajectories, num_time_steps, d))
    X_OT[:, 0, :] = marginal_samples[0]  # Initial condition

    sum_Edxt_xtT = np.zeros((d, d))
    sum_Ext_xtT = np.zeros((d, d))

    for t in range(num_time_steps - 1):
        # extract random samples of the process
        if t == 0 or not use_raw_avg:
            X_t = marginal_samples[t]
            X_t1 = marginal_samples[t + 1]
        else:
            # if we want to build on the trajectory ordering that OT picks and treat them as
            # trajectories for raw averages
            X_t = X_t1_OT
            X_t1 = marginal_samples[t + 1]  # marginal_samples[t + 1]

        # Calculate the cost matrix
        if cur_est_A is None:
            # optimize over empirical marginal transition
            M = ot.dist(X_t, X_t1, metric='sqeuclidean')
        else:
            # optimize over empirical marginal transition given current estimated A
            M = ot.dist(X_t + np.dot(X_t, expm(cur_est_A * dt)), X_t1, metric='sqeuclidean')

        # Solve optimal transport problem
        if entropy_reg > 0:
            # max_iter = 10000  # Increase the maximum number of iterations
            # numItermax=max_iter, stopThr=thresh
            # thresh = 1e-6  # Decrease the threshold
            if entropy_reg > sinkhorn_log_thresh:
                p = ot.sinkhorn(a=np.ones(len(X_t)) / len(X_t), b=np.ones(len(X_t1)) / len(X_t1), M=M,
                                reg=entropy_reg, verbose=False)
            else:
                p = ot.sinkhorn(a=np.ones(len(X_t)) / len(X_t), b=np.ones(len(X_t1)) / len(X_t1), M=M,
                                reg=entropy_reg, verbose=False, method='sinkhorn_log')
        else:
            p = ot.emd(a=np.ones(len(X_t)) / len(X_t), b=np.ones(len(X_t1)) / len(X_t1), M=M)
        # use the optimal transport matched trajectories
        if use_raw_avg:
            # Normalize each row of the transport plan to sum to 1
            p_normalized = normalize_rows(p)
            # Calculate X_t1_OT predictions
            if entropy_reg == 0:
                X_t1_OT = np.dot(p_normalized, X_t1)
                X_OT[:, t + 1, :] = X_t1_OT
            else:
                indices = np.arange(num_trajectories)
                X_t1_OT = np.zeros(shape=(num_trajectories, d))  # set up time slice for time t+1
                for i in range(num_trajectories):
                    p_normalized_i = p_normalized[i]
                    j = np.random.choice(indices, p=p_normalized_i)
                    # print(f'for trajectory {i}: {X_t[i]}, we pick continuation {j}: {X_t1[j]}')
                    X_t1_OT[i] = X_t1[j]
                X_OT[:, t + 1, :] = X_t1_OT
            term1 = sum(np.outer(X_OT[i, t + 1, :] - X_OT[i, t, :], X_OT[i, t, :]) for i in
                        range(num_trajectories)) / num_trajectories
            term2 = sum(np.outer(X_OT[i, t, :], X_OT[i, t, :]) for i in range(num_trajectories)) / num_trajectories
        else:
            term1 = np.zeros((d, d))
            for i in range(num_trajectories):
                for j in range(num_trajectories):
                    term1[0, 0] += p[j, i] * (X_t1[i] - X_t[j]) * X_t[j]
            # term2 = sum(np.outer(X_t[i], X_t[i]) for i in range(num_trajectories)) / num_trajectories
            term2 = np.dot(X_t.T, X_t) / num_trajectories
        sum_Edxt_xtT += term1
        sum_Ext_xtT += term2
    if pinv:
        est_A = np.matmul(sum_Edxt_xtT, np.linalg.pinv(sum_Ext_xtT)) * (1 / dt)
    else:
        est_A = left_Var_Equation(sum_Ext_xtT, sum_Edxt_xtT * (1 / dt))
    if return_OT_traj:
        return est_A, X_OT
    else:
        return est_A


# def estimate_A_exp(trajectories, dt, GGT=None):
#     """
#     Calculate the closed form estimator A_hat using observed data from multiple trajectories using the expectation formulation.
#
#     Parameters:
#         trajectories (numpy.ndarray): 3D array where each slice corresponds to a single trajectory (num_trajectories, num_steps, d).
#         dt (float): Discretization time step.
#         GGT (optional): the Gram matrix of the diffusion matrix
#
#     Returns:
#         numpy.ndarray: Estimated drift matrix A given the set of trajectories
#     """
#     num_trajectories, num_steps, d = trajectories.shape
#     # A_hat = np.zeros((d, d))
#     #
#     # if GGT is None:
#     #     GGT = np.eye(d)  # Use identity if no GGT provided
#
#     # Initialize cumulative sums
#     sum_Edxt_Ext = np.zeros((d, d))
#     sum_Ext_ExtT = np.zeros((d, d))
#
#     for t in range(num_steps - 1):
#         sum_dxt_xt = np.zeros((d, d))
#         sum_xt_xt = np.zeros((d, d))
#         for trajectory in trajectories:
#             sum_dxt_xt += np.outer(trajectory[t + 1] - trajectory[t], trajectory[t])
#             sum_xt_xt += np.outer(trajectory[t], trajectory[t])
#         sum_Edxt_Ext += sum_dxt_xt / num_trajectories
#         sum_Ext_ExtT += sum_xt_xt / num_trajectories
#     return np.matmul(sum_Edxt_Ext, np.linalg.pinv(sum_Ext_ExtT)) * (1 / dt)


def estimate_A_exp(trajectories, dt, GGT=None, pinv=False):
    """
    Calculate the closed form estimator A_hat using observed data from multiple trajectories using the expectation formulation.

    Parameters:
        trajectories (numpy.ndarray): 3D array where each slice corresponds to a single trajectory (num_trajectories, num_steps, d).
        dt (float): Discretization time step.
        GGT (optional): the Gram matrix of the diffusion matrix

    Returns:
        numpy.ndarray: Estimated drift matrix A given the set of trajectories
    """
    num_trajectories, num_steps, d = trajectories.shape
    # A_hat = np.zeros((d, d))
    #
    # if GGT is None:
    #     GGT = np.eye(d)  # Use identity if no GGT provided

    # Initialize cumulative sums
    sum_Edxt_Ext = np.zeros((d, d))
    sum_Ext_ExtT = np.zeros((d, d))

    for t in range(num_steps - 1):
        sum_dxt_xt = np.zeros((d, d))
        sum_xt_xt = np.zeros((d, d))
        for trajectory in trajectories:
            sum_dxt_xt += np.outer(trajectory[t + 1] - trajectory[t], trajectory[t])
            sum_xt_xt += np.outer(trajectory[t], trajectory[t])
        sum_Edxt_Ext += sum_dxt_xt / num_trajectories
        sum_Ext_ExtT += sum_xt_xt / num_trajectories

    if pinv:
        return np.matmul(sum_Edxt_Ext, np.linalg.pinv(sum_Ext_ExtT)) * (1 / dt)
    else:
        return left_Var_Equation(sum_Ext_ExtT, sum_Edxt_Ext * (1 / dt))


def estimate_A(trajectories, dt, GGT=None):
    """
    Calculates the closed form estimator A_hat from each of the observed trajectories
    and averages the estimates at the end.


    Parameters:
        trajectories (numpy.ndarray): 3D array where each slice corresponds to a single trajectory (num_trajectories, num_steps, d).
        dt (float): Discretization time step.
        GGT (optional): the Gram matrix of the diffusion matrix

    Returns:
        numpy.ndarray: Estimated drift matrix A given the set of trajectories
    """
    num_trajectories, num_steps, d = trajectories.shape
    A_hat = np.zeros((d, d))
    if GGT is None:
        GGT = np.eye(d)
    for trajectory in trajectories:
        # perform the estimate
        sum_xt_dxt = np.zeros((d, d))
        sum_xt_xtT = np.zeros((d, d))
        for t in range(num_steps - 1):
            xt = trajectory[t]
            xt_next = trajectory[t + 1]
            dxt = xt_next - xt
            sum_xt_dxt += np.outer(dxt, xt)
            sum_xt_xtT += np.outer(xt, xt)
        # Accumulating the estimates from all trajectories
        GGT_inv = np.linalg.inv(GGT)
        temp1 = np.matmul(GGT, sum_xt_dxt)
        temp2 = np.matmul(GGT_inv, np.linalg.pinv(sum_xt_xtT))
        estimator_from_traj = np.matmul(temp1, temp2) * (1 / dt)
        A_hat += estimator_from_traj

    # Averaging over all trajectories
    A_hat /= num_trajectories

    return A_hat


def estimate_GGT(trajectories, T):
    """
    Estimate the matrix GG^T from multiple trajectories of a multidimensional
    Ornstein-Uhlenbeck process.

    Parameters:
        trajectories (numpy.ndarray): A 3D array where each "slice" (2D array) corresponds to a single trajectory.
        T (float): Total time period.

    Returns:
        numpy.ndarray: Estimated GG^T matrix.
    """
    num_trajectories, num_steps, d = trajectories.shape

    # Initialize the GG^T matrix
    GGT = np.zeros((d, d))

    # Compute increments ΔX for each trajectory
    increments = np.diff(trajectories, axis=1)

    # Sum up the products of increments for each dimension pair across all trajectories and steps
    for i in range(d):
        for j in range(d):
            GGT[i, j] = np.sum(increments[:, :, i] * increments[:, :, j])

    # Divide by total time T*num_trajectories to normalize
    GGT /= T * num_trajectories
    return GGT


def left_Var_Equation(A1, B1):
    """
    doing the stable version of np.matmul(sum_Edxt_Ext, np.linalg.pinv(sum_Ext_ExtT)) * (1 / dt)
    generally for solving XA = B you should transpose them and use least squares.
    A^T X^T = B^T
    """
    m = B1.shape[0]
    n = A1.shape[0]
    X = np.zeros((m, n))
    for i in range(m):
        X[i, :] = np.linalg.lstsq(np.transpose(A1), B1[i, :])[0]
    return X


def normalize_rows(matrix):
    """
    Normalize each row of the matrix to sum to 1.

    Parameters:
        matrix (numpy.ndarray): The matrix to normalize.

    Returns:
        numpy.ndarray: The row-normalized matrix.
    """
    row_sums = matrix.sum(axis=1, keepdims=True)
    return matrix / row_sums


def plot_comparison(X, X_OT, X_OT_reg, trajectory_index=0):
    """
    Plot the true trajectory vs. OT-predicted trajectories for different entropy regularizations.

    Parameters:
        X (numpy.ndarray): True trajectories.
        X_OT (numpy.ndarray): OT-predicted trajectories with no entropy regularization.
        X_OT_reg (numpy.ndarray): OT-predicted trajectories with entropy regularization.
        trajectory_index (int): Index of the trajectory to plot.
    """
    num_time_steps, d = X.shape[1], X.shape[2]

    plt.figure(figsize=(10, 6))
    for dim in range(d):
        plt.subplot(d, 1, dim + 1)
        plt.plot(np.arange(num_time_steps), X[trajectory_index, :, dim], 'k-',
                 label='True Trajectory' if dim == 0 else "")
        plt.plot(np.arange(num_time_steps), X_OT[trajectory_index, :, dim], 'r--',
                 label='OT Predicted (No Reg)' if dim == 0 else "")
        plt.plot(np.arange(num_time_steps), X_OT_reg[trajectory_index, :, dim], 'b-.',
                 label='OT Predicted (Reg)' if dim == 0 else "")
        plt.xlabel('Time Step')
        plt.ylabel(f'Trajectory Value (Dim {dim + 1})')
        plt.title(f'Trajectory {trajectory_index}, Dimension {dim + 1}')
        if dim == 0:
            plt.legend()
    plt.tight_layout()
    plt.show()


def plot_fuck(X, X_shuffled, trajectory_index=0):
    """
    Plot the true trajectory vs. OT-predicted trajectories for different entropy regularizations.

    Parameters:
        X (numpy.ndarray): True trajectories.
        X_OT (numpy.ndarray): OT-predicted trajectories with no entropy regularization.
        X_OT_reg (numpy.ndarray): OT-predicted trajectories with entropy regularization.
        trajectory_index (int): Index of the trajectory to plot.
    """
    num_time_steps, d = X.shape[1], X.shape[2]

    plt.figure(figsize=(10, 6))
    for dim in range(d):
        plt.subplot(d, 1, dim + 1)
        plt.plot(np.arange(num_time_steps), X[trajectory_index, :, dim], 'k-',
                 label='True Trajectory' if dim == 0 else "")
        plt.plot(np.arange(num_time_steps), X_shuffled[trajectory_index, :, dim], 'r--',
                 label='shuffled' if dim == 0 else "")
        plt.xlabel('Time Step')
        plt.ylabel(f'Trajectory Value (Dim {dim + 1})')
        plt.title(f'Trajectory {trajectory_index}, Dimension {dim + 1}')
        if dim == 0:
            plt.legend()
    plt.tight_layout()
    plt.show()
