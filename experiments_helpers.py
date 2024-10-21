import math
import ot
import os
from scipy.special import logsumexp
import matplotlib.pyplot as plt
from scipy.linalg import expm
import time
import pickle
from scipy.stats import multivariate_normal
import numpy as np


def angle_between(v1, v2):
    """
    Helper function to compute the angle between two vectors in radians.
    """
    dot_product = np.dot(v1, v2)
    norms = np.linalg.norm(v1) * np.linalg.norm(v2)
    cos_angle = dot_product / norms
    cos_angle = np.clip(cos_angle, -1.0, 1.0)  # Clip for numerical stability
    return np.arccos(cos_angle)


def generate_independent_points(d, num_points, min_magnitude=2, max_magnitude=10, min_angle_degrees=30, max_its=1000):
    '''
    Theorem 1 implies that a d-dimensional 0-mean linear additive noise is identifiable from X0 if X0 is supported on
    d linearly independent points. This function generates linearly independent points in d-dimensional space.
    Args:
        d: dimension
        num_points: number of points to generate (if > d, then additional points are generated without constraints)
        min_magnitude: minimum Euclidean norm for considered points
        max_magnitude: maximum Euclidean norm for considered points
        min_angle_degrees: minimum pairwise angle required between considered points in degrees
        max_its: maximum number of iterations to attempt to generate linearly independent points with min_angle_degrees
    Returns:
        list of numpy.ndarray: list of linearly independent points
    '''
    # Convert minimum angle from degrees to radians
    min_angle_radians = np.radians(min_angle_degrees)

    points = []
    # Generate first random point
    point = np.random.uniform(-1, 1, d)
    point = point / np.linalg.norm(point)  # Normalize
    scale = np.random.uniform(min_magnitude, max_magnitude)
    point = point * scale
    points.append(point)
    # Generate remaining linearly independent points with at least min_angle_radians between each pair
    for _ in range(1, min(num_points, d)):
        its = 0
        independent = False
        while not independent:
            its += 1
            # Generate candidate point
            candidate_point = np.random.uniform(-1, 1, d)
            candidate_point = candidate_point / np.linalg.norm(candidate_point)
            scale = np.random.uniform(min_magnitude, max_magnitude)
            candidate_point = candidate_point * scale

            # check for linear independence
            matrix = np.vstack(points + [candidate_point])
            rank = np.linalg.matrix_rank(matrix)
            if rank == len(points) + 1:
                # Check the angles with all existing points to also ensure that pairwise angles are sufficiently large
                independent = True
                for existing_point in points:
                    angle = angle_between(candidate_point, existing_point)
                    if angle < min_angle_radians:
                        independent = False
                        if its > max_its:
                            print(
                                f'max number of iterations {max_its} exceeded. Consider increasing max_its or '
                                f'decreasing min_angle_degrees')
                            break
                if independent:
                    points.append(candidate_point)

    # If num_points > d, generate additional points without constraints
    for _ in range(d, num_points):
        candidate_point = np.random.uniform(-1, 1, d)
        candidate_point = candidate_point / np.linalg.norm(candidate_point)  # Normalize
        scale = np.random.uniform(min_magnitude, max_magnitude)
        candidate_point = candidate_point * scale
        points.append(candidate_point)
    return points


def generate_random_matrix_with_eigenvalue_constraint(d, eigenvalue_threshold=1, sparsity_threshold=0,
                                                      epsilon=0, max_iterations=1e5):
    '''
    Args:
        d: dimension of square matrix to be generated
        eigenvalue_threshold: maximal real part of eigenvalue
        sparsity_threshold: fraction of elements to set to zero, used for causal discovery experiment
        epsilon: minimum magnitude for matrix entries (set to 0.5 for causal discovery experiment)
        max_iterations: maximum number of iterations to attempt to generate a matrix with eigenvalue constraint
    Returns:
        np.array: d x d random matrix with eigenvalue constraint
    '''
    for _ in range(int(max_iterations)):
        M = np.random.uniform(low=epsilon, high=5, size=(d, d))
        sign_matrix = np.random.choice([-1, 1], size=M.shape)
        M = M * sign_matrix

        # Introduce sparsity if applicable
        if sparsity_threshold > 0:
            mask = np.random.rand(d, d) < sparsity_threshold
            M = np.multiply(M, mask)

        # Eigenvalue check
        eigenvalues = np.linalg.eigvals(M)
        max_eigenvalue = np.max(eigenvalues.real)
        if max_eigenvalue < eigenvalue_threshold:
            return M  # Return the matrix if the condition is satisfied

    # Step 5: Raise an exception if no valid matrix was found within the iteration limit
    raise ValueError(
        f"Failed to generate a matrix of dimension {d} with max real eigenvalue {eigenvalue_threshold} after {max_iterations} iterations. Consider lowering eigenvalue threshold or increasing max_iterations")


def linear_additive_noise_data(num_trajectories, d, T, dt_EM, dt, A, G, X0_dist=None, stationary=False,
                               matrix_exponential=False):
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
        if X0_dist is not None:
            # Sample X0 based on the provided probability distribution
            X0_ = X0_dist[np.random.choice(len(X0_dist), p=[prob for _, prob in X0_dist])][0]
        else:
            cov_matrix = np.eye(d)
            X0_ = np.random.multivariate_normal(np.zeros(d), cov_matrix)

        if matrix_exponential:
            # print('This should not be happening')
            X_true = ou_process_matrix_exponential(T, dt_EM, A, G, X0_)
        else:
            X_true = ou_process(T, dt_EM, A, G, X0_)

        for i in range(n_measured_times):
            X_measured[n, i, :] = X_true[i * rate, :]
    return X_measured


def killed_linear_additive_noise_data(num_trajectories, d, T, dt_EM, dt, A, G, X0_dist=None, matrix_exponential=False,
                                      stationary=False):
    n_measured_times = int(T / dt)
    X_measured = np.zeros((num_trajectories, n_measured_times, d))
    for i in range(n_measured_times):
        for n in range(num_trajectories):
            if X0_dist is not None:
                # Sample X0 based on the provided probability distribution
                X0_ = X0_dist[np.random.choice(len(X0_dist), p=[prob for _, prob in X0_dist])][0]
            else:
                cov_matrix = np.eye(d)
                X0_ = np.random.multivariate_normal(np.zeros(d), cov_matrix)
            if i == 0:
                X_measured[n, 0, :] = X0_
            else:
                # cell trajectory will terminate at i*dt
                measured_T = i * dt
                # use consistent X0s across the measured times
                X_measured[n, i, :] = ou_process(measured_T, dt_EM, A, G, X0_)[-1]
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


def plot_trajectories(X, T, dt, save_file=False, N_truncate=None):
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
            plt.plot(time_steps, X[n, :, d], label=f'{n}th trajectory dim {d}')

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


def estimate_A_exact(X, dt):
    """
    Calculate the closed form estimator A_hat using observed data from multiple trajectories
    using the expectation formulation.

    Parameters:
        X (numpy.ndarray): 3D array where each slice corresponds to a single trajectory (num_trajectories, num_steps, d).
        dt (float): Discretization time step.

    Returns:
        numpy.ndarray: Estimated drift matrix A given the set of trajectories.
    """
    num_trajectories, num_steps, d = X.shape
    # Initialize cumulative sums
    sum_Xtp1_XtT = np.zeros((d, d))  # Sum of X_{t+1} * X_t^T
    sum_Xt_XtT = np.zeros((d, d))  # Sum of X_t * X_t^T

    for t in range(num_steps - 1):
        sum_Xtp1_Xt = np.zeros((d, d))
        sum_Xt_Xt = np.zeros((d, d))
        for n in range(num_trajectories):
            Xt = X[n, t, :]  # X_t for trajectory n
            Xtp1 = X[n, t + 1, :]  # X_{t+1} for trajectory n
            sum_Xtp1_Xt += np.outer(Xtp1, Xt)  # X_{t+1} * X_t^T
            sum_Xt_Xt += np.outer(Xt, Xt)  # X_t * X_t^T
        sum_Xtp1_XtT += sum_Xtp1_Xt / num_trajectories
        sum_Xt_XtT += sum_Xt_Xt / num_trajectories
    return (np.log(sum_Xtp1_XtT) - np.log(sum_Xt_XtT)) * (1 / dt)


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


# def estimate_GGT_exact(X, T, est_A = None):
#     """
#     Estimate the matrix GG^T from multiple trajectories of a multidimensional
#     Ornstein-Uhlenbeck process.
#
#     Parameters:
#         X (numpy.ndarray): A 3D array where each "slice" (2D array) corresponds to a single trajectory.
#         T (float): Total time period.
#
#     Returns:
#         numpy.ndarray: Estimated GG^T matrix.
#     """
#     num_trajectories, num_steps, d = X.shape
#
#     dt = T/num_steps
#
#     # Initialize the GG^T matrix
#     GGT = np.zeros((d, d))
#
#
#     # Sum up the products of increments for each dimension pair across all trajectories and steps
#     for t in range(num_steps-1):
#         for i in range(d):
#             for j in range(d):
#                 for n in range(num_trajectories):
#                     if est_A is None:
#                         dX_i = X[n, t+1, i] - X[n, t, i]
#                         dX_j = X[n, t+1, j] - X[n, t, j]
#                     else:
#                         dX = X[n, t+1, :] - np.matmul((X[n, t, :]), expm(est_A*dt))
#                         dX_i = dX[i]
#                         dX_j = dX[j]
#                     GGT[i, j] += dX_i * dX_j
#
#     GGT /= (T - dt) * num_trajectories
#     return GGT

def estimate_GGT_exact(X, T, est_A=None):
    """
    Estimate the matrix GG^T from multiple trajectories of a multidimensional
    Ornstein-Uhlenbeck process.

    Parameters:
        X (numpy.ndarray): A 3D array where each "slice" (2D array) corresponds to a single trajectory.
        T (float): Total time period.
        est_A (numpy.ndarray, optional): Estimated drift matrix A. If provided, the increments will be adjusted by the deterministic drift.

    Returns:
        numpy.ndarray: Estimated GG^T matrix.
    """
    num_trajectories, num_steps, d = X.shape
    dt = T / num_steps

    # Initialize the GG^T matrix
    GGT = np.zeros((d, d))

    if est_A is None:
        # Compute increments ΔX for each trajectory (no drift adjustment)
        increments = np.diff(X, axis=1)
    else:
        # Precompute exp(A * dt)
        exp_Adt = expm(est_A * dt)
        # Adjust increments: X_{t+1} - exp(A * dt) * X_t
        increments = X[:, 1:, :] - np.einsum('ij,nkj->nki', exp_Adt, X[:, :-1, :])

    # Efficient computation of GG^T using einsum
    GGT = np.einsum('nti,ntj->ij', increments, increments)

    # Normalize by total time and number of trajectories
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


#
#
# def sinkhorn_multidimensional(a, b, K, maxiter=1000, stopThr=1e-9, epsilon=1e-2, log_threshold=0):
#     '''
#     Oct 12 commented out
#     Args:
#         a:
#         b:
#         K:
#         maxiter:
#         stopThr:
#         epsilon:
#         log_threshold:
#
#     Returns:
#
#     '''
#     u = np.ones(K.shape[0])
#     v = np.ones(K.shape[1])
#
#     log_scale = np.min(K) < log_threshold  # Check if we should switch to log-scale computations
#     if log_scale:
#         log_K = np.log(K + 1e-300)  # Add a small value to prevent log(0) in case K has zeros
#         log_a = np.log(a + 1e-300)
#         log_b = np.log(b + 1e-300)
#         log_u = np.zeros(K.shape[0])
#         log_v = np.zeros(K.shape[1])
#     else:
#         log_K, log_a, log_b = None, None, None  # Placeholder in case we don't use log-scale
#
#     for _ in range(maxiter):
#         u_prev = u
#
#         if log_scale:
#             # Perform updates in the log domain
#             log_u = log_a - np.log(np.exp(log_K + log_v).sum(axis=1))
#             log_v = log_b - np.log(np.exp(log_K.T + log_u[:, np.newaxis]).sum(axis=0))
#             u = np.exp(log_u)
#             v = np.exp(log_v)
#         else:
#             # Perform standard Sinkhorn update
#             u = a / (K @ v)
#             v = b / (K.T @ u)
#
#         # Calculate the transport plan
#         if log_scale:
#             tmp = np.exp(log_u[:, np.newaxis] + log_K + log_v)
#         else:
#             tmp = np.diag(u) @ K @ np.diag(v)
#
#         # Check for convergence based on the error
#         err = np.linalg.norm(tmp.sum(axis=1) - a)
#         if np.linalg.norm(u_prev):
#             if err < stopThr or np.linalg.norm(u - u_prev) / np.linalg.norm(u_prev) < epsilon:
#                 break
#
#     return tmp
#

def sinkhorn_multidimensional(a, b, K, maxiter=1000, stopThr=1e-9, epsilon=1e-2, log_threshold=1e-10):
    '''
    Sept 13 version
    Args:
        a:
        b:
        K:
        maxiter:
        stopThr:
        epsilon:
        log_threshold:

    Returns:

    '''
    u = np.ones(K.shape[0])
    v = np.ones(K.shape[1])

    log_scale = np.min(K) < log_threshold  # Check if we should switch to log-scale computations
    if log_scale:
        log_K = np.log(K + 1e-300)  # Add a small value to prevent log(0) in case K has zeros
        log_a = np.log(a + 1e-300)
        log_b = np.log(b + 1e-300)
        log_u = np.zeros(K.shape[0])
        log_v = np.zeros(K.shape[1])
    else:
        log_K, log_a, log_b = None, None, None  # Placeholder in case we don't use log-scale

    for _ in range(maxiter):
        u_prev = u

        if log_scale:
            # Perform updates in the log domain
            log_u = log_a - np.log(np.exp(log_K + log_v).sum(axis=1) + 1e-300)
            log_v = log_b - np.log(np.exp(log_K.T + log_u[:, np.newaxis]).sum(axis=0) + 1e-300)
            u = np.exp(log_u)
            v = np.exp(log_v)
        else:
            # Perform standard Sinkhorn update
            u = a / (K @ v)
            v = b / (K.T @ u)

        # Calculate the transport plan
        if log_scale:
            tmp = np.exp(log_K + log_u[:, np.newaxis] + log_v)
        else:
            tmp = np.diag(u) @ K @ np.diag(v)

        # Check for convergence based on the error
        err = np.linalg.norm(tmp.sum(axis=1) - a)
        if err < stopThr or np.linalg.norm(u - u_prev) / np.linalg.norm(u_prev) < epsilon:
            break

    return tmp


def sinkhorn_multidimensional_new(a, b, K, maxiter=1000, stopThr=1e-9, epsilon=1e-2, log_threshold=0,
                                  reg_factor=1e-8, min_val=1e-300, return_both_plans=False):
    u = np.ones(K.shape[0])
    v = np.ones(K.shape[1])

    # Regularize K to avoid zeros and very small values
    K[K < min_val] = min_val

    # Decide whether to use log-scale based on the smallest value in K
    log_scale = np.min(K) < log_threshold
    if log_scale:
        print('joey')
        # Log-scale variables
        log_K = np.log(K)
        log_a = np.log(a + min_val)
        log_b = np.log(b + min_val)
        log_u = np.zeros(K.shape[0])
        log_v = np.zeros(K.shape[1])
    else:
        log_K = log_a = log_b = None

    for it in range(maxiter):
        if log_scale:
            log_u_prev = log_u.copy()

            # Perform log-domain updates
            log_u = log_a - logsumexp(log_K + log_v, axis=1)
            log_v = log_b - logsumexp(log_K.T + log_u[:, np.newaxis], axis=0)

            # Compute relative change for convergence
            rel_change = np.linalg.norm(log_u - log_u_prev)
            if rel_change < epsilon:
                break

        else:
            u_prev = u.copy()

            # Standard Sinkhorn updates
            u = a / (K @ v)
            v = b / (K.T @ u)

            # Compute relative change for convergence
            denom = np.linalg.norm(u_prev)
            if np.isfinite(denom) and denom > 0:
                rel_change = np.linalg.norm(u - u_prev) / denom
            else:
                rel_change = np.inf

            if rel_change < epsilon:
                break

    if log_scale:
        # Compute the transport plan using log-scale computations
        exponent = log_K + log_u[:, np.newaxis] + log_v
        max_exponent = np.max(exponent)
        exponent -= max_exponent  # Shift exponent for numerical stability
        pi_log = np.exp(exponent)
        pi_log[pi_log < min_val] = min_val  # Avoid underflow
        pi_log /= np.sum(pi_log)  # Normalize transport plan

        if return_both_plans:
            # Compute standard transport plan for comparison
            u_standard = np.exp(log_u)
            v_standard = np.exp(log_v)
            pi_standard = np.diag(u_standard) @ K @ np.diag(v_standard)
            pi_standard[pi_standard < min_val] = min_val  # Avoid underflow
            pi_standard /= np.sum(pi_standard)
            return pi_log, pi_standard
        else:
            return pi_log

    else:
        # Compute transport plan using standard (non-log-scale) method
        pi_standard = np.diag(u) @ K @ np.diag(v)
        pi_standard[pi_standard < min_val] = min_val  # Avoid underflow
        pi_standard /= np.sum(pi_standard)
        return pi_standard


# def sinkhorn_multidimensional(a, b, K, maxiter=1000, stopThr = 1e-9, epsilon=1e-2):
#     u = np.ones(K.shape[0])
#     v = np.ones(K.shape[1])
#
#     for _ in range(maxiter):
#         u_prev = u
#         u = a / (K @ v)
#         v = b / (K.T @ u)
#         tmp = np.diag(u) @ K @ np.diag(v)
#         err = np.linalg.norm(tmp.sum(axis=1) - a)
#         if err < stopThr or np.linalg.norm(u - u_prev) / np.linalg.norm(u_prev) < epsilon:
#             break
#     return tmp


def create_OT_traj_md(X, D, dt, cur_est_A=None, linearization=True, report_time_splits=False):
    marginal_samples = extract_marginal_samples(X)
    np.random.seed()
    num_time_steps = len(marginal_samples)
    d = marginal_samples[0].shape[1]
    if D is None:
        D = np.eye(d)
    num_trajectories = marginal_samples[0].shape[0]
    ps = []  # transport plans
    sinkhorn_time = 0
    K_time = 0
    epsilon = 1e-8
    for t in range(num_time_steps - 1):
        # extract marginal samples
        X_t = marginal_samples[t]
        X_t1 = marginal_samples[t + 1]
        a = np.ones(len(X_t)) / len(X_t)
        b = np.ones(len(X_t1)) / len(X_t1)

        # Precompute reusable terms for matrix A and regularize D
        if cur_est_A is None:
            cur_est_A = np.zeros((d, d))  # Set Brownian motion as reference SDE if no A inputted

        A_dt = cur_est_A * dt if linearization else expm(cur_est_A * dt)
        if linearization:
            A_X_t = np.matmul(A_dt, X_t.T)
            D_reg = D + np.eye(D.shape[0]) * epsilon
            cov_D_dt = D_reg * dt  # Precompute D * dt once to avoid repeated computation
        else:
            assert d == 1
            D_reg = D
            if cur_est_A[0][0] == 0:
                cov_D_dt = D_reg * dt
            else:
                cov_D_dt = (D_reg / (2 * abs(cur_est_A[0][0]))) * (1 - np.exp(-2 * abs(cur_est_A[0][0]) * dt))
                # cov_D_dt = D_reg * dt
                #
                # print('cov_wiki:', cov_D_dt)
                # print('cov_opposite_sign:', cov_D_dt_)

        K = np.zeros((num_trajectories, num_trajectories))

        # Loop over trajectories, vectorize inner calculations
        for i in range(num_trajectories):
            t1 = time.time()

            if linearization:
                # Vectorize the computation for all j's
                dX_ij = X_t1 - X_t[i] - A_X_t[:, i].T
            else:
                # Matrix multiply and vectorize
                dX_ij = X_t1 - np.matmul(A_dt, X_t[i])

            # Flatten the differences for all pairs (vectorized)
            dX_ij_flattened = dX_ij.reshape(num_trajectories, d)
            try:
                # Vectorized PDF computation for all j's
                K[i, :] = multivariate_normal.pdf(dX_ij_flattened, mean=np.zeros(d), cov=cov_D_dt)
            except np.linalg.LinAlgError:
                # If numerical issues, regularize again and compute PDFs
                print(f"Numerical issue in multivariate normal pdf at i={i}")
                cov_D_dt += np.eye(D.shape[0]) * epsilon  # Further regularize if needed
                K[i, :] = multivariate_normal.pdf(dX_ij_flattened, mean=np.zeros(d), cov=cov_D_dt)

            t2 = time.time()
            K_time += t2 - t1
        t1 = time.time()
        p = sinkhorn_multidimensional(a=a, b=b, K=K)
        t2 = time.time()
        sinkhorn_time += t2 - t1
        ps.append(p)

    t1 = time.time()
    N = 1000
    X_OT = np.zeros(shape=(N, num_time_steps, d))
    OT_index_propagation = np.zeros(shape=(N, num_time_steps - 1))
    # Precompute normalized probabilities once
    normalized_ps = np.array([normalize_rows(ps[t]) for t in range(num_time_steps - 1)])
    indices = np.arange(num_trajectories)
    for _ in range(N):
        for t in range(num_time_steps - 1):
            pt_normalized = normalized_ps[t]
            if t == 0:
                k = np.random.randint(num_trajectories)
                X_OT[_, 0, :] = marginal_samples[0][k]
            else:
                # retrieve where _th observation at time 0 was projected to at time t
                k = int(OT_index_propagation[_, t - 1])
            j = np.random.choice(indices, p=pt_normalized[k])
            OT_index_propagation[_, t] = int(j)
            X_OT[_, t + 1, :] = marginal_samples[t + 1][j]
    t2 = time.time()
    ot_traj_time = t2 - t1
    if report_time_splits:
        print('Time setting up K:', K_time)
        print('Time doing Sinkhorn:', sinkhorn_time)
        print('Time creating trajectories', ot_traj_time)
    return X_OT


def estimate_A_exp_ot_with_traj(X, dt, T=1, cur_est_A=None, cur_est_D=None, linearization=True,
                                report_time_splits=False):
    X_OT = create_OT_traj_md(X, cur_est_D, dt, cur_est_A, linearization=linearization,
                             report_time_splits=report_time_splits)
    if linearization:
        A_OT = estimate_A_exp(X_OT, dt)
        G_OT = estimate_GGT(X_OT, T, est_A=A_OT)
    else:
        # print('bibbona: this should not be happening')
        A_OT = estimate_A_exact(X_OT, dt)
        G_OT = estimate_GGT_exact(X_OT, T, est_A=A_OT)

    # A_OT, G_OT = estimate_A_GGT_ot_with_K(X, cur_est_D, dt, cur_est_A, linearization=linearization,
    #                                       report_time_splits=report_time_splits)
    return A_OT, G_OT


'''
craparino below
'''


# def estimate_A_GGT_ot_with_K(X, D, dt, cur_est_A=None, linearization=True, report_time_splits=False):
#     """
#     Estimate the drift matrix A and diffusion matrix GGT using optimal transport
#     between successive marginal distributions, utilizing the kernel matrix K and
#     multidimensional Sinkhorn algorithm without explicit trajectory creation.
#     Compute the optimal transport plan p once per time step and use it for both
#     A and GGT estimation. Report time splits if required.
#     """
#     import time
#
#     marginal_samples = extract_marginal_samples(X)
#     num_time_steps = len(marginal_samples)
#     d = marginal_samples[0].shape[1]
#     if D is None:
#         D = np.eye(d)
#     num_trajectories = marginal_samples[0].shape[0]
#     epsilon = 1e-8
#
#     # Initialize time measurements
#     K_time = 0
#     sinkhorn_time = 0
#
#     sum_Edxt_xtT = np.zeros((d, d))
#     sum_Ext_xtT = np.zeros((d, d))
#     ps = []
#
#     total_time = (num_time_steps -1 ) * dt
#
#     for t in range(num_time_steps - 1):
#         X_t = marginal_samples[t]
#         X_t1 = marginal_samples[t + 1]
#         a = np.ones(len(X_t)) / len(X_t)
#         b = np.ones(len(X_t1)) / len(X_t1)
#
#         # Use initial or zero matrix for A
#         if cur_est_A is None:
#             cur_est_A = np.zeros((d, d))
#
#         # Regularize D
#         D_reg = D + np.eye(D.shape[0]) * epsilon
#         cov_D_dt = D_reg * dt
#
#         # Compute A_dt and A_X_t consistent with create_OT_traj_md
#         if linearization:
#             A_dt = cur_est_A * dt
#             A_X_t = np.matmul(A_dt, X_t.T)  # Shape: (d, num_trajectories)
#         else:
#             A_dt = expm(cur_est_A * dt)
#             A_X_t = np.matmul(A_dt, X_t.T) - X_t.T
#
#         # Compute residuals: D_diff = X_t1[j] - X_t[i] - A_X_t[:, i].T
#         D_diff = X_t1[None, :, :] - X_t[:, None, :] - A_X_t.T[:, None, :]
#
#         # Compute the kernel matrix K
#         t1 = time.time()
#         K = np.zeros((num_trajectories, num_trajectories))
#         for i in range(num_trajectories):
#             dX_ij_flattened = D_diff[i, :, :]  # Shape (num_trajectories, d)
#             try:
#                 K[i, :] = multivariate_normal.pdf(dX_ij_flattened, mean=np.zeros(d), cov=cov_D_dt)
#             except np.linalg.LinAlgError:
#                 cov_D_dt += np.eye(D.shape[0]) * epsilon
#                 K[i, :] = multivariate_normal.pdf(dX_ij_flattened, mean=np.zeros(d), cov=cov_D_dt)
#         t2 = time.time()
#         K_time += t2 - t1
#
#         # Compute the optimal transport plan p
#         t1 = time.time()
#         p = sinkhorn_multidimensional(a=a, b=b, K=K)
#         ps.append(p)
#         t2 = time.time()
#         sinkhorn_time += t2 - t1
#
#         # Estimate A
#         term1 = np.zeros((d, d))
#         for i in range(num_trajectories):
#             for j in range(num_trajectories):
#                 term1 += p[i, j] * (X_t1[j] - X_t[i]) * X_t[i]
#         term2 = np.dot(X_t.T, X_t) / num_trajectories
#         sum_Edxt_xtT += term1
#         sum_Ext_xtT += term2
#
#     # Solve for A
#     est_A = 1/dt * sum_Edxt_xtT * np.linalg.pinv(sum_Ext_xtT)  #np.linalg.solve(sum_Ext_xtT, sum_Edxt_xtT * (1 / dt))
#
#     # Estimate H
#     sum_H = np.zeros((d, d))
#     for t in range(num_time_steps-1):
#         p = ps[t]
#         X_t = marginal_samples[t]
#         X_t1 = marginal_samples[t + 1]
#         for i in range(num_trajectories):
#             for j in range(num_trajectories):
#                 sum_H += p[i, j] * np.outer((X_t1[j] - X_t[i] - np.matmul(est_A, X_t[i])),
#                                             (X_t1[j] - X_t[i] - np.matmul(est_A, X_t[i])))
#
#     # Compute GGT without dividing by num_trajectories
#     est_GGT = sum_H / total_time
#
#     if report_time_splits:
#         print('Time setting up K:', K_time)
#         print('Time doing Sinkhorn:', sinkhorn_time)
#
#     return est_A, est_GGT


def estimate_A_GGT_ot_with_K(X, D, dt, cur_est_A=None, linearization=True, report_time_splits=False):
    """
    Estimate the drift matrix A and diffusion matrix GGT using optimal transport
    between successive marginal distributions, utilizing the kernel matrix K and
    multidimensional Sinkhorn algorithm without explicit trajectory creation.
    Compute the optimal transport plan p once per time step and use it for both
    A and GGT estimation. Report time splits if required.
    """
    import time

    marginal_samples = extract_marginal_samples(X)
    num_time_steps = len(marginal_samples)
    # print(num_time_steps)
    d = marginal_samples[0].shape[1]
    if D is None:
        D = np.eye(d)
    num_trajectories = marginal_samples[0].shape[0]
    epsilon = 1e-8

    # Initialize time measurements
    K_time = 0
    sinkhorn_time = 0

    sum_Edxt_xtT = np.zeros((d, d))
    sum_Ext_xtT = np.zeros((d, d))
    sum_H = np.zeros((d, d))
    ps = []
    total_time = (num_time_steps - 1) * dt

    for t in range(num_time_steps - 1):
        X_t = marginal_samples[t]
        X_t1 = marginal_samples[t + 1]
        a = np.ones(len(X_t)) / len(X_t)
        b = np.ones(len(X_t1)) / len(X_t1)

        # Use initial or zero matrix for A
        if cur_est_A is None:
            cur_est_A = np.zeros((d, d))

        # Regularize D
        D_reg = D + np.eye(D.shape[0]) * epsilon
        cov_D_dt = D_reg * dt

        # Compute A_dt and A_X_t consistent with create_OT_traj_md
        if linearization:
            A_dt = cur_est_A * dt
            A_X_t = np.matmul(A_dt, X_t.T)  # Shape: (d, num_trajectories)
        else:
            A_dt = expm(cur_est_A * dt)
            A_X_t = np.matmul(A_dt, X_t.T) - X_t.T

        # Compute residuals: D_diff = X_t1[j] - X_t[i] - A_X_t[:, i].T
        D_diff = X_t1[None, :, :] - X_t[:, None, :] - A_X_t.T[:, None, :]

        # Compute the kernel matrix K
        t1 = time.time()
        K = np.zeros((num_trajectories, num_trajectories))
        for i in range(num_trajectories):
            dX_ij_flattened = D_diff[i, :, :]  # Shape (num_trajectories, d)
            try:
                K[i, :] = multivariate_normal.pdf(dX_ij_flattened, mean=np.zeros(d), cov=cov_D_dt)
            except np.linalg.LinAlgError:
                cov_D_dt += np.eye(D.shape[0]) * epsilon
                K[i, :] = multivariate_normal.pdf(dX_ij_flattened, mean=np.zeros(d), cov=cov_D_dt)
        t2 = time.time()
        K_time += t2 - t1

        # Compute the optimal transport plan p
        t1 = time.time()
        p = sinkhorn_multidimensional(a=a, b=b, K=K)
        ps.append(p)
        t2 = time.time()
        sinkhorn_time += t2 - t1

        # Estimate A
        term1 = np.zeros((d, d))
        for i in range(num_trajectories):
            Xi = X_t[i]  # Shape (d,)
            # Compute weighted sum over j
            residuals_ij = D_diff[i, :, :]  # Shape (num_trajectories, d)
            weighted_residual = np.dot(p[i, :], residuals_ij)  # Shape (d,)
            term1 += np.outer(weighted_residual, Xi)
        # term2 = sum_{i} X_t[i] X_t[i]^T / num_trajectories
        term2 = np.dot(X_t.T, X_t) / num_trajectories

        sum_Edxt_xtT += term1
        sum_Ext_xtT += term2

        # Estimate GGT
        # H_term = sum_{i,j} p[i,j] * outer(residuals[i,j,:], residuals[i,j,:])

    # Solve for A
    est_A = np.linalg.solve(sum_Ext_xtT, sum_Edxt_xtT * (1 / dt))
    A_dt = est_A * dt if linearization else expm(est_A * dt)

    # Compute GGT without dividing by num_trajectories
    for t in range(num_time_steps - 1):
        p = ps[t]
        X_t = marginal_samples[t]
        X_t1 = marginal_samples[t + 1]
        A_X_t = np.matmul(A_dt, X_t.T)
        residuals = X_t1[None, :, :] - X_t[:, None, :] - A_X_t.T[:, None, :]
        H_term = np.einsum('ij,ijk,ijl->kl', p, residuals, residuals)
        sum_H += H_term
    est_GGT = sum_H / total_time

    if report_time_splits:
        print('Time setting up K:', K_time)
        print('Time doing Sinkhorn:', sinkhorn_time)

    return est_A, est_GGT


def save_with_unique_filename(filepath):
    """Ensures that the file is saved with a unique name if the file already exists."""
    base_filepath, extension = os.path.splitext(filepath)
    counter = 1
    unique_filepath = filepath

    # Keep checking and incrementing the counter until a unique filename is found
    while os.path.exists(unique_filepath):
        unique_filepath = f"{base_filepath}({counter}){extension}"
        counter += 1

    return unique_filepath


def compute_mae(estimated, ground_truth):
    """Compute Mean Absolute Percentage Error (MAE)"""
    mae = np.mean(np.abs((estimated - ground_truth)))
    return mae
