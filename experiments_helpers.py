import math
import numpy as np
import ot
import os
import warnings
import matplotlib.pyplot as plt
from scipy.linalg import expm
import time
import pickle


def generate_independent_points(d, num_points, min_magnitude=0, max_magnitude=10, min_angle_degrees=30):
    points = []

    # Generate first random point
    point = np.random.uniform(-1, 1, d)
    point = point / np.linalg.norm(point)  # Normalize
    scale = np.random.uniform(min_magnitude, max_magnitude)  # Scale to desired magnitude
    point = point * scale
    points.append(point)

    def angle_between(v1, v2):
        """Returns the angle in radians between vectors 'v1' and 'v2'"""
        dot_product = np.dot(v1, v2)
        norms = np.linalg.norm(v1) * np.linalg.norm(v2)
        cos_angle = dot_product / norms
        # Clip to handle numerical precision issues that may push cos_angle slightly out of range
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        return np.arccos(cos_angle)

    # Convert minimum angle from degrees to radians
    min_angle_radians = np.radians(min_angle_degrees)

    # Generate remaining linearly independent points
    for _ in range(1, num_points):
        while True:
            candidate_point = np.random.uniform(-1, 1, d)
            candidate_point = candidate_point / np.linalg.norm(candidate_point)  # Normalize
            scale = np.random.uniform(min_magnitude, max_magnitude)
            candidate_point = candidate_point * scale

            # Check linear independence by ensuring angles between vectors are above threshold
            independent = True
            for existing_point in points:
                angle = angle_between(candidate_point, existing_point)
                if angle < min_angle_radians:
                    independent = False
                    break

            if independent:
                points.append(candidate_point)
                break

    return points


def linear_additive_noise_data(num_trajectories, d, T, dt_EM, dt, A, G, X0_dist=None, stationary=False, matrix_exponential=False):
    '''
    Args:
        num_trajectories: number of trajectories
        d: dimension of process
        T: Total time period
        dt_EM: discretization time step used for simulating the raw cell trajectories
        dt: discretization time step of the measurements
        A (numpy.ndarray): Drift matrix.
        G (numpy.ndarray): Variance matrix.
        X0_dist (list of tuples): List of tuples, each containing an initial value and its associated probability.
        stationary (bool): Whether to use the stationary distribution for initial conditions..
        exact (bool): Whether to use the exact solution or the Euler-Maruyama method.

    Returns:
        numpy.ndarray: Measured trajectories.
    '''
    n_measured_times = int(T / dt)
    X_measured = np.zeros((num_trajectories, n_measured_times, d))
    rate = int(dt / dt_EM)

    for n in range(num_trajectories):
        if stationary:
            drift_scale = np.linalg.norm(A)
            D = np.linalg.norm(np.matmul(G, np.transpose(G)))
            if drift_scale > 0:
                stationary_variance = D / (2 * d * drift_scale)
            else:
                stationary_variance = D / (2 * d * 1e-10)
                cov = np.identity(d) * stationary_variance
                X0_ = np.random.multivariate_normal(np.zeros(d), cov)
        else:
            if X0_dist is not None:
                # Sample X0 based on the provided probability distribution
                X0_ = X0_dist[np.random.choice(len(X0_dist), p=[prob for _, prob in X0_dist])][0]
            else:
                cov_matrix = np.eye(d)
                X0_ = np.random.multivariate_normal(np.zeros(d), cov_matrix)

        if matrix_exponential:
            X_true = ou_process_matrix_exponential(T, dt_EM, A, G, X0_)
        else:
            X_true = ou_process(T, dt_EM, A, G, X0_)

        for i in range(n_measured_times):
            X_measured[n, i, :] = X_true[i * rate, :]
    return X_measured

def ou_process_matrix_exponential(T, dt, A, G, X0, seed=None):
    """
    Simulate the Ornstein-Uhlenbeck process at a given time t using an exact solution.

    Parameters:
        T (float): Total time period.
        dt (float): Time step size for discretization of the integral.
        A (numpy.ndarray): Drift matrix.
        G (numpy.ndarray): Diffusion matrix.
        X0 (numpy.ndarray): Initial value.
        seed (int, optional): Random seed for reproducibility.

    Returns:
        numpy.ndarray: Array of simulated trajectories.
    """
    if seed is not None:
        np.random.seed(seed)

    num_steps = int(T / dt)
    d = len(X0)
    X = np.zeros((num_steps, d))
    X[0] = X0

    # Compute the deterministic part
    exp_At = expm(A * dt)

    for i in range(1, num_steps):
        s = i * dt

        # Deterministic part
        X_det = np.dot(exp_At, X[i - 1])

        # Stochastic part
        dWs = np.random.normal(0, np.sqrt(dt), size=G.shape[1])
        X_sto = np.dot(expm(A * (dt)), G @ dWs)

        # Sum deterministic and stochastic parts
        X[i] = X_det + X_sto

    return X

def ou_process(T, dt, A, G, X0, seed=None):
    """
    Simulate a single trajectory of a multidimensional Ornstein-Uhlenbeck process:
    dX_t = AX_tdt + GdW_t

    Parameters:
        T (float): Total time period.
        dt (float): Time step size.
        A (numpy.ndarray): Drift matrix.
        G (numpy.ndarray): Variance matrix.
        X0 (numpy.ndarray): Initial value.

    Returns:
        numpy.ndarray: Array of simulated trajectories.
    """
    if seed is not None:
        np.random.seed(seed)
    num_steps = int(T / dt)
    d = len(X0)
    m = G.shape[0]
    dW = np.sqrt(dt) * np.random.randn(num_steps, m)
    X = np.zeros((num_steps, d))
    X[0] = X0

    for t in range(1, num_steps):
        X[t] = X[t - 1] + dt * (A.dot(X[t - 1])) + G.dot(dW[t])

    return X

def plot_trajectories(X, T, dt, save_file = False, N_truncate = None):
    """
    Plot the trajectories of a multidimensional process.

    Parameters:
        X (numpy.ndarray): Array of trajectories.
        T (float): Total time period.
        dt (float): Time step size.
    """
    num_trajectories, num_steps, num_dimensions = X.shape
    if N_truncate is not None:
        num_trajectories = N_truncate

    time_steps = np.linspace(0, T, num_steps)  # Generate time steps corresponding to [0, T]

    # Plot trajectories
    plt.figure(figsize=(12, 8))


    for n in range(num_trajectories):
        for d in range(num_dimensions):
            plt.plot(time_steps, X[n, :, d ], label=f'{n}th trajectory dim {d}')


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

### MLE estimators for drift A and diffusion GGT

def estimate_A_exp(X, dt, GGT=None, pinv=False):
    """
    Calculate the closed form estimator A_hat using observed data from multiple trajectories using the expectation formulation.

    Parameters:
        trajectories (numpy.ndarray): 3D array where each slice corresponds to a single trajectory (num_trajectories, num_steps, d).
        dt (float): Discretization time step.
        GGT (optional): the Gram matrix of the diffusion matrix

    Returns:
        numpy.ndarray: Estimated drift matrix A given the set of trajectories
    """
    num_trajectories, num_steps, d = X.shape
    # Initialize cumulative sums
    sum_Edxt_Ext = np.zeros((d, d))
    sum_Ext_ExtT = np.zeros((d, d))

    for t in range(num_steps - 1):
        sum_dxt_xt = np.zeros((d, d))
        sum_xt_xt = np.zeros((d, d))
        for n in range(num_trajectories):
            xt = X[n, t, :]
            dxt = X[n, t + 1, :] - X[n, t, :]
            sum_dxt_xt += np.outer(dxt, xt)
            sum_xt_xt += np.outer(xt, xt)
        sum_Edxt_Ext += sum_dxt_xt / num_trajectories
        sum_Ext_ExtT += sum_xt_xt / num_trajectories

    if pinv:
        # return sum_Edxt_Ext / sum_Ext_ExtT * 1/ dt
        return np.matmul(sum_Edxt_Ext, np.linalg.pinv(sum_Ext_ExtT)) * (1 / dt)
    else:
        return left_Var_Equation(sum_Ext_ExtT, sum_Edxt_Ext * (1 / dt))

# def estimate_GGT(trajectories, T, est_A=None):
#     """
#     Estimate the matrix GG^T from multiple trajectories of a multidimensional
#     Ornstein-Uhlenbeck process.
#
#     Parameters:
#         trajectories (numpy.ndarray): A 3D array where each "slice" (2D array) corresponds to a single trajectory.
#         T (float): Total time period.
#         est_A (numpy.ndarray, optional): Estimated drift matrix A. If provided, the increments will be adjusted by the deterministic drift.
#
#     Returns:
#         numpy.ndarray: Estimated GG^T matrix.
#     """
#     num_trajectories, num_steps, d = trajectories.shape
#     dt = T / num_steps
#
#     # Initialize the GG^T matrix
#     GGT = np.zeros((d, d))
#
#     if est_A is None:
#         # Compute increments ΔX for each trajectory (no drift adjustment)
#         increments = np.diff(trajectories, axis=1)
#     else:
#         # Adjust increments by subtracting the deterministic drift: ΔX - A * X_t * dt
#         increments = np.diff(trajectories, axis=1) - dt*np.matmul(trajectories[:, :-1, :], est_A)
#
#     # Sum up the products of increments for each dimension pair across all trajectories and steps
#     for i in range(d):
#         for j in range(d):
#             GGT[i, j] = np.sum(increments[:, :, i] * increments[:, :, j])
#
#     # Divide by total time T*num_trajectories to normalize
#     GGT /= (T - dt) * num_trajectories
#     return GGT

def estimate_GGT(trajectories, T, est_A=None):
    """
    Estimate the matrix GG^T from multiple trajectories of a multidimensional
    Ornstein-Uhlenbeck process.

    Parameters:
        trajectories (numpy.ndarray): A 3D array where each "slice" (2D array) corresponds to a single trajectory.
        T (float): Total time period.
        est_A (numpy.ndarray, optional): Estimated drift matrix A. If provided, the increments will be adjusted by the deterministic drift.

    Returns:
        numpy.ndarray: Estimated GG^T matrix.
    """
    num_trajectories, num_steps, d = trajectories.shape
    dt = T / num_steps

    # Initialize the GG^T matrix
    GGT = np.zeros((d, d))

    if est_A is None:
        # Compute increments ΔX for each trajectory (no drift adjustment)
        increments = np.diff(trajectories, axis=1)
    else:
        # Adjust increments by subtracting the deterministic drift: ΔX - A * X_t * dt
        increments = np.diff(trajectories, axis=1) - dt * np.einsum('ij,nkj->nki', est_A, trajectories[:, :-1, :])

    # Sum up the products of increments for each dimension pair across all trajectories and steps
    for i in range(d):
        for j in range(d):
            GGT[i, j] = np.sum(increments[:, :, i] * increments[:, :, j])

    # Divide by total time T*num_trajectories to normalize
    GGT /= (T - dt) * num_trajectories
    return GGT

### Optimal Transport matching from only observed marginals


### Helper functions

def extract_marginal_samples(trajectories, shuffle=True):
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
        X[i, :] = np.linalg.lstsq(np.transpose(A1), B1[i, :], rcond=None)[0]
    return X


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
        flattened[t * num_trajectories:(t + 1) * num_trajectories] = trajectories[:, t, :]

    return flattened


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


def sinkhorn_multidimensional(a, b, K, maxiter=1000, stopThr = 1e-9):
    u = np.ones(K.shape[0])
    v = np.ones(K.shape[1])

    for _ in range(maxiter):
        u = a / (K @ v)
        v = b / (K.T @ u)
        tmp = np.diag(u) @ K @ np.diag(v)
        err = np.linalg.norm(tmp.sum(axis=1) - a)
        if err < stopThr:
            break
    return tmp


def create_OT_traj_md(X, D, dt, cur_est_A=None, frac_other_time_samples=0, linearization=True):
    marginal_samples = extract_marginal_samples(X)
    np.random.seed()
    num_time_steps = len(marginal_samples)
    d = marginal_samples[0].shape[1]
    if D is None:
        D = np.eye(d)
    num_trajectories = marginal_samples[0].shape[0]
    # transport plans
    ps = []
    sinkhorn_time = 0
    K_time = 0
    for t in range(num_time_steps - 1):
        if frac_other_time_samples == 0:
            # extract marginal samples
            X_t = marginal_samples[t]
            X_t1 = marginal_samples[t + 1]
            a = np.ones(len(X_t)) / len(X_t)
            b = np.ones(len(X_t1)) / len(X_t1)
            # if cur_est_A is not None:
            #     M = ot.dist(X_t + np.matmul(X_t, cur_est_A) * dt, X_t1, metric='sqeuclidean')
            # else:
            #     M = ot.dist(X_t, X_t1, metric = 'sqeuclidean')
        else:
            observations_sorted_by_time = flatten_trajectories_sequentially(X)
            X_t = observations_sorted_by_time
            X_t1 = observations_sorted_by_time
            a, b = set_probabilities(observations_sorted_by_time, num_trajectories, t, frac_other_time_samples)

        K = np.zeros((num_trajectories, num_trajectories))
        # Compute Cholesky factorization of D * dt
        if np.linalg.det(D) != 0:
            for i in range(d):
                D[i,i] += 1e-8
        inv_D_dt = np.linalg.lstsq(D.T * dt, np.eye(D.shape[0]), rcond=None)[0]#np.linalg.pinv(D * dt)
        for i in range(num_trajectories):
            for j in range(num_trajectories):
                t1 = time.time()
                if cur_est_A is None:
                    cur_est_A = np.zeros((d, d))  # set Brownian motion as reference SDE if no A inputted
                if linearization:
                    dX = X_t1[j] - X_t[i] - np.matmul(cur_est_A * dt, X_t[i])
                else:
                    dX = X_t1[j] - np.matmul(expm(cur_est_A * dt), X_t[i])
                dX = dX.reshape(-1, 1)  # Reshape to a column vector
                if np.linalg.det(D) != 0:
                    K[i, j] = (2*math.pi)**(-d/2)*np.linalg.det(D)**(-1/2)*np.exp(-0.5 * np.dot(dX.T, inv_D_dt @ dX).item())
                else:
                    K[i, j] = np.exp(-0.5 * np.dot(dX.T, inv_D_dt @ dX).item())
                t2 = time.time()
                K_time += t2 - t1
        t1 = time.time()
        # if True:
        p = sinkhorn_multidimensional(a=a, b=b, K=K)
        # else:
        #     p = ot.sinkhorn(a, b, M=M/2, reg=dt)
        t2 = time.time()
        sinkhorn_time += t2-t1
        ps.append(p)

    t1 = time.time()
    N = max(5 * num_trajectories, 1000)
    X_OT = np.zeros(shape=(N, num_time_steps, d))
    OT_index_propagation = np.zeros(shape=(N, num_time_steps - 1))
    if frac_other_time_samples == 0:
        indices = np.arange(num_trajectories)
    else:
        indices = np.arange(num_trajectories * num_time_steps)
    for _ in range(N):
        for t in range(num_time_steps - 1):
            pt_normalized = normalize_rows(ps[t])
            if t == 0:
                k = np.random.randint(num_trajectories)
                X_OT[_, 0, :] = marginal_samples[0][k]
            else:
                # retrieve where _th observation at time 0 was projected to at time t
                k = int(OT_index_propagation[_, t - 1])
            j = np.random.choice(indices, p=pt_normalized[k])
            OT_index_propagation[_, t] = int(j)
            if frac_other_time_samples == 0:
                X_OT[_, t + 1, :] = marginal_samples[t + 1][j]
            else:
                X_OT[_, t + 1, :] = observations_sorted_by_time[j]
    t2 = time.time()
    ot_traj_time = t2-t1
    print('Time setting up K:', K_time)
    print('Time doing Sinkhorn:', sinkhorn_time)
    print('Time creating trajectories', ot_traj_time)
    return X_OT


def estimate_A_exp_ot_with_traj(X, dt, T=1, frac_other_time_samples=0,
                                sinkhorn_log_thresh=0.001, cur_est_A=None, cur_est_D=None, estimate_G=True, linearization = True):
    X_OT = create_OT_traj_md(X, cur_est_D, dt, cur_est_A, frac_other_time_samples, linearization=linearization)
    t1 = time.time()
    A_OT = estimate_A_exp(X_OT, dt)
    t2 = time.time()
    print('Time estimating A:', t2-t1)
    if estimate_G:
        t1 = time.time()
        G_OT = estimate_GGT(X_OT, T, est_A=A_OT)
        t2 = time.time()
        print('Time estimating G:', t2 - t1)
        return A_OT, G_OT, X_OT
    else:
        return A_OT, X_OT
