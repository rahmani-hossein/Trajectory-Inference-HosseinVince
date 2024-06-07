import numpy as np
import ot
from simulate_trajectories import extract_marginals
def estimate_drift_diffusion(X, T, dt, use_estimated_GGT = False, expectation = True, OT = True):
    GGT = estimate_GGT(X, T)
    if use_estimated_GGT is True:
        if expectation is True:
            # print('using expected values over trajectories')
            A = estimate_A_exp(X, dt)
        else:
            A = estimate_A(X, dt, GGT)
    else:
        if expectation is True:
            # print('using expected values over trajectories')
            if OT == True:
                # print('using optimal transport')
                marginals = extract_marginals(X)
                A = estimate_A_exp_ot(marginals, dt)
            else:
                A = estimate_A_exp(X, dt)
        else:
            A = estimate_A(X, dt)
    return A, GGT


def estimate_A_exp_ot(marginals, dt, entropy_reg = 0.01):
    """
    Estimate the drift matrix A using optimal transport between successive marginal distributions.

    Parameters:
        marginals (list of numpy.ndarray): List of arrays, each containing samples from the marginal distribution at each time step.
        dt (float): Discretization time step.

    Returns:
        numpy.ndarray: Estimated drift matrix A
    """
    num_time_steps = len(marginals)
    num_dimensions = marginals[0].shape[1]

    sum_Edxt_Ext = np.zeros((num_dimensions, num_dimensions))
    sum_Ext_ExtT = np.zeros((num_dimensions, num_dimensions))

    for t in range(num_time_steps - 1):
        X_t = marginals[t]
        X_t1 = marginals[t + 1]

        # Calculate the cost matrix (Euclidean distance squared between each pair)
        M = ot.dist(X_t, X_t1, metric='sqeuclidean')
        # Solve optimal transport problem
        if entropy_reg is not None:
            # Compute the regularized OT plan using the Sinkhorn algorithm
            p = ot.sinkhorn(a=np.ones(len(X_t)) / len(X_t), b=np.ones(len(X_t1)) / len(X_t1), M=M, reg=reg)
        else:
            p = ot.emd(a=np.ones(len(X_t)) / len(X_t), b=np.ones(len(X_t1)) / len(X_t1), M=M)

        # Use optimal transport distribution to estimate required terms
        sum_Edxt_Ext += (X_t1.T @ p @ X_t - X_t.T @ p @ X_t) / len(X_t)
        sum_Ext_ExtT += (X_t.T @ p @ X_t) / len(X_t)

    # Compute A_hat using the estimated expectations
    return np.matmul(sum_Edxt_Ext, np.linalg.pinv(sum_Ext_ExtT)) * (1 / dt)

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
    num_trajectories, num_steps, num_dimensions = trajectories.shape

    # Initialize the GG^T matrix
    GGT = np.zeros((num_dimensions, num_dimensions))

    # Compute increments ΔX for each trajectory
    increments = np.diff(trajectories, axis=1)


    # Sum up the products of increments for each dimension pair across all trajectories and steps
    for i in range(num_dimensions):
        for j in range(num_dimensions):
            GGT[i, j] = np.sum(increments[:, :, i] * increments[:, :, j])

    # Divide by total time T*num_trajectories to normalize
    GGT /= T*num_trajectories
    return GGT

def estimate_A(trajectories, dt, GGT = None):
    """
    Calculate the closed form estimator A_hat using observed data from multiple trajectories.

    Parameters:
        trajectories (numpy.ndarray): 3D array where each slice corresponds to a single trajectory (num_trajectories, num_steps, num_dimensions).
        dt (float): Discretization time step.
        GGT (optional): the Gram matrix of the diffusion matrix

    Returns:
        numpy.ndarray: Estimated drift matrix A given the set of trajectories
    """
    num_trajectories, num_steps, num_dimensions = trajectories.shape
    A_hat = np.zeros((num_dimensions, num_dimensions))
    if GGT is None:
        GGT = np.eye(num_dimensions)
    for trajectory in trajectories:
        # perform the estimate
        sum_xt_dxt = np.zeros((num_dimensions, num_dimensions))
        sum_xt_xtT = np.zeros((num_dimensions, num_dimensions))
        for t in range(num_steps-1):
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

def estimate_A_exp(trajectories, dt, GGT = None):
    """
    Calculate the closed form estimator A_hat using observed data from multiple trajectories using the expectation formulation.

    Parameters:
        trajectories (numpy.ndarray): 3D array where each slice corresponds to a single trajectory (num_trajectories, num_steps, num_dimensions).
        dt (float): Discretization time step.
        GGT (optional): the Gram matrix of the diffusion matrix

    Returns:
        numpy.ndarray: Estimated drift matrix A given the set of trajectories
    """
    num_trajectories, num_steps, num_dimensions = trajectories.shape
    A_hat = np.zeros((num_dimensions, num_dimensions))

    if GGT is None:
        GGT = np.eye(num_dimensions)  # Use identity if no GGT provided

    # Initialize cumulative sums
    sum_Edxt_Ext = np.zeros((num_dimensions, num_dimensions))
    sum_Ext_ExtT = np.zeros((num_dimensions, num_dimensions))

    for t in range(num_steps - 1):
        sum_dxt_xt = np.zeros((num_dimensions, num_dimensions))
        sum_xt_xt = np.zeros((num_dimensions, num_dimensions))
        for trajectory in trajectories:
            sum_dxt_xt += np.outer(trajectory[t+1] - trajectory[t], trajectory[t])
            sum_xt_xt += np.outer(trajectory[t], trajectory[t])
        sum_Edxt_Ext += sum_dxt_xt / num_trajectories
        sum_Ext_ExtT += sum_xt_xt / num_trajectories
    return np.matmul(sum_Edxt_Ext, np.linalg.pinv(sum_Ext_ExtT)) * (1/dt)


def estimate_A_1D(trajectories, dt):
    A_hat = 0
    num_trajectories, num_steps, num_dimensions = trajectories.shape
    for trajectory in trajectories:
        # perform the estimate
        sum_xt_dxt = 0
        sum_xt_xtT = 0
        for t in range(num_steps - 1):
            xt = trajectory[t]
            xt_next = trajectory[t + 1]
            dxt = xt_next - xt
            sum_xt_dxt += xt*dxt
            sum_xt_xtT += xt*xt*dt
        # Accumulating the estimates from all trajectories
        estimator_from_traj = sum_xt_dxt/sum_xt_xtT
        A_hat += estimator_from_traj

    # Averaging over all trajectories
    A_hat /= num_trajectories

    return A_hat


def estimate_A_(trajectories, dt, G):
    """
    Calculate the closed form estimator A_hat using discrete observed data from multiple trajectories.

    Parameters:
        trajectories (numpy.ndarray): 3D array where each slice corresponds to a single trajectory (num_trajectories, num_steps, num_dimensions).
        dt (float): Discretization time step.

    Returns:
        numpy.ndarray: Estimated drift matrix A given the set of trajectories
    """
    num_trajectories, num_steps, num_dimensions = trajectories.shape
    A_hat = np.zeros((num_dimensions, num_dimensions))
    sum_xt_dxt = np.zeros((num_dimensions, num_dimensions))
    sum_xt_xtT = np.zeros((num_dimensions, num_dimensions))

    for trajectory in trajectories:
        # perform the estimate
        for t in range(num_steps-1):
            xt = trajectory[t]
            xt_next = trajectory[t + 1]
            dxt = xt_next - xt
            sum_xt_dxt += np.outer(xt, dxt)
            sum_xt_xtT += np.outer(xt, xt)
    # Accumulating the estimates from all trajectories
    if np.shape(G)[0]>1:
        GGT = np.matmul(G, np.transpose(G))
        GGT_inv = np.linalg.inv(GGT)
    else:
        GGT = [1]
    temp1 = np.matmul(GGT, sum_xt_dxt)
    temp2 = np.matmul(GGT_inv, np.linalg.inv(sum_xt_xtT))
    estimator_from_traj = np.matmul(temp1, temp2) * (1 / dt)
    A_hat += estimator_from_traj
    #
    # # Averaging over all trajectories
    # A_hat /= num_trajectories

    return A_hat
