import numpy as np
import ot
from scipy.linalg import expm

def extract_marginal_samples(trajectories):
    """
    Extract marginal distributions from a 3D trajectory array.

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
        marginal_samples.append(samples_at_t)

    return marginal_samples
# def estimate_A_compare_methods(X, dt, entropy_reg = 0):
#     A_hat_traj = estimate_linear_drift(X, dt, expectation = True, OT = False, entropy_reg = 0, GGT = None)
#     A_hat_OT = estimate_linear_drift(X, dt, expectation = True, OT = True, entropy_reg = 0, GGT = None)
#     A_hat_OT_reg = estimate_linear_drift(X, dt, expectation=True, OT=True, entropy_reg=entropy_reg, GGT=None)
#     return  A_hat_traj, A_hat_OT, A_hat_OT_reg

def estimate_A_compare_methods(X, dt, entropy_reg, methods, n_iterations = 1):
    """
    Estimate A using various methods.

    Parameters:
    - X: The trajectories.
    - dt: Time step.
    - entropy_reg: Entropy regularization parameter.
    - methods: List of method names to use for estimation.

    Returns:
    - Dictionary of estimated A matrices keyed by method name.
    """
    A_estimations = {}

    # Define the estimation functions for each method
    for method in methods:
        if method == 'Trajectory':
            A_estimations[method] = estimate_linear_drift(X, dt, expectation=True, OT=False, entropy_reg=0, GGT=None)
        elif method == 'OT':
            A_estimations[method] = estimate_linear_drift(X, dt, expectation=True, OT=True, entropy_reg=0, GGT=None, n_iterations=n_iterations)
        elif method == 'OT reg':
            A_estimations[method] = estimate_linear_drift(X, dt, expectation=True, OT=True, entropy_reg=entropy_reg, GGT=None, n_iterations=n_iterations)
        elif method == 'Classical':
            A_estimations[method] = estimate_linear_drift(X, dt, expectation = False, GGT = None)
        else:
            raise ValueError(f"Unsupported method: {method}")

    return A_estimations

def estimate_linear_drift(X, dt, expectation = True, OT = True, entropy_reg = 0, GGT = None, n_iterations = 1):
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
            # extract the marginal samples first
            marginals = extract_marginal_samples(X)
            its = 1
            # initial estimate for A
            # the expectations are estimated using conditional densities from OT
            A_0 = estimate_A_exp_ot(marginals, dt, entropy_reg=entropy_reg, cur_est_A=None)
            A = A_0
            # print(f'estimated A for iteration 1:', A)
            while its < n_iterations:
                A = estimate_A_exp_ot(marginals, dt, entropy_reg = entropy_reg, cur_est_A = A)
                its += 1
                # print(f'estimated A for iteration {its}:', A)
        else:
            # the expectations are taken over the set of all observed trajectories
            A = estimate_A_exp(X, dt)
    else:
        # we estimate A using the classical closed form solution (no expectations)
        A = estimate_A(X, dt, GGT = GGT)
    return A

def estimate_A_exp_ot(marginal_samples, dt, entropy_reg = 0.01, cur_est_A = None, use_raw_avg = True):
    """
    Estimate the drift matrix A using optimal transport between successive marginal distributions.

    Parameters:
        marginals (list of numpy.ndarray): List of arrays, each containing samples from the marginal distribution at each time step.
        dt (float): Discretization time step.

    Returns:
        numpy.ndarray: Estimated drift matrix A
    """
    num_time_steps = len(marginal_samples)
    d = marginal_samples[0].shape[1]
    num_trajectories = marginal_samples[0].shape[0]

    sum_Edxt_xtT = np.zeros((d, d))
    sum_Ext_xtT = np.zeros((d, d))

    for t in range(num_time_steps - 1):
        # extract the samples of the process, taken from time t and t+1
        X_t = marginal_samples[t]
        X_t1 = marginal_samples[t + 1]
        # Calculate the cost matrix
        if cur_est_A is None:
            # optimize over empirical marginal transition
            M = ot.dist(X_t, X_t1, metric='sqeuclidean')
        else:
            # optimize over empirical marginal transition given current estimated A
            # delta_Xt = np.zeros_like(X_t)
            # # Iterate over each trajectory
            # for i in range(num_trajectories):
            #
            #     delta_Xt[i, :] = np.dot(cur_est_A, X_t[i, :])*dt
            # delta_Xt = X_t @ expm(cur_est_A * dt)
            # Compute the cost matrix with the correctly aligned dimensions
            # delta_Xt = np.dot(X_t, cur_est_A.T) * dt
            # M = ot.dist(np.dot(X_t, cur_est_A.T) * dt, X_t1-X_t, metric='sqeuclidean')
            # M = ot.dist(X_t + np.dot(X_t, cur_est_A.T) * dt, X_t1, metric='sqeuclidean')
            M = ot.dist(X_t + np.dot(X_t, expm(cur_est_A * dt)), X_t1, metric='sqeuclidean')


        # Solve optimal transport problem
        if entropy_reg > 0:
            # Compute the entropy-regularized OT plan using the Sinkhorn algorithm.
            # We normalize the cost matrix with respect to its largest entry for numerical stability
            p = ot.sinkhorn(a=np.ones(len(X_t)) / len(X_t), b=np.ones(len(X_t1)) / len(X_t1), M=M/M.max(), reg=entropy_reg)
        else:
            p = ot.emd(a=np.ones(len(X_t)) / len(X_t), b=np.ones(len(X_t1)) / len(X_t1), M=M)
        #
        # print('optimal transport plan dimensions:', p.shape)
        # print('plan:', p)
        # Use optimal transport distribution to estimate required terms
        sum_Edxt_xtT += (X_t1.T @ p @ X_t - X_t.T @ p @ X_t)
        if use_raw_avg:
            term = sum(np.outer(X_t[i], X_t[i]) for i in range(num_trajectories))/num_trajectories
        else:
            term = (X_t.T @ p @ X_t)   #sum(np.outer(X_t[i], X_t[i]) for i in range(num_trajectories))/num_trajectories
        sum_Ext_xtT += term
        #(X_t.T @ np.eye(len(X_t)) @ X_t)  / len(X_t)
        #(X_t.T @ np.eye(len(X_t)) @ X_t)  / len(X_t))
        #(X_t.T @ p @ X_t) / len(X_t) #(X_t.T @ np.eye(len(X_t)) @ X_t) / (len(X_t)**2) # (X_t.T @ p @ X_t) / len(X_t)

    # Compute A_hat using the estimated expectations
    return np.matmul(sum_Edxt_xtT, np.linalg.pinv(sum_Ext_xtT)) * (1 / dt)

def estimate_A(trajectories, dt, GGT = None):
    """
    Calculate the closed form estimator A_hat using observed data from multiple trajectories.

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
        trajectories (numpy.ndarray): 3D array where each slice corresponds to a single trajectory (num_trajectories, num_steps, d).
        dt (float): Discretization time step.
        GGT (optional): the Gram matrix of the diffusion matrix

    Returns:
        numpy.ndarray: Estimated drift matrix A given the set of trajectories
    """
    num_trajectories, num_steps, d = trajectories.shape
    A_hat = np.zeros((d, d))

    if GGT is None:
        GGT = np.eye(d)  # Use identity if no GGT provided

    # Initialize cumulative sums
    sum_Edxt_Ext = np.zeros((d, d))
    sum_Ext_ExtT = np.zeros((d, d))

    for t in range(num_steps - 1):
        sum_dxt_xt = np.zeros((d, d))
        sum_xt_xt = np.zeros((d, d))
        for trajectory in trajectories:
            sum_dxt_xt += np.outer(trajectory[t+1] - trajectory[t], trajectory[t])
            sum_xt_xt += np.outer(trajectory[t], trajectory[t])
        sum_Edxt_Ext += sum_dxt_xt / num_trajectories
        sum_Ext_ExtT += sum_xt_xt / num_trajectories
    return np.matmul(sum_Edxt_Ext, np.linalg.pinv(sum_Ext_ExtT)) * (1/dt)

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
    GGT /= T*num_trajectories
    return GGT




def estimate_A_exp_alt(trajectories, dt, GGT = None):
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
    A_hat = np.zeros((d, d))

    if GGT is None:
        GGT = np.eye(d)  # Use identity if no GGT provided

    # Initialize cumulative sums
    sum_Edxt_Ext = np.zeros((d, d))
    sum_Ext_ExtT = np.zeros((d, d))

    for t in range(num_steps - 1):
        sum_dxt_xt = np.zeros((d, d))
        sum_xt = np.zeros((d, 1))
        sum_xtT = np.zeros((1, d))
        for trajectory in trajectories:
            sum_dxt_xt += np.outer(trajectory[t+1] - trajectory[t], trajectory[t])
            sum_xt += np.reshape(trajectory[t], (d, 1))
            sum_xtT += np.transpose(trajectory[t])
        Ext = sum_xt /num_trajectories
        ExtT = sum_xtT / num_trajectories
        sum_Edxt_Ext += sum_dxt_xt / num_trajectories
        sum_Ext_ExtT += np.matmul(Ext, ExtT) / num_trajectories
    return np.matmul(sum_Edxt_Ext, np.linalg.pinv(sum_Ext_ExtT)) * (1/dt)

def estimate_A_1D(trajectories, dt):
    A_hat = 0
    num_trajectories, num_steps, d = trajectories.shape
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
        trajectories (numpy.ndarray): 3D array where each slice corresponds to a single trajectory (num_trajectories, num_steps, d).
        dt (float): Discretization time step.

    Returns:
        numpy.ndarray: Estimated drift matrix A given the set of trajectories
    """
    num_trajectories, num_steps, d = trajectories.shape
    A_hat = np.zeros((d, d))
    sum_xt_dxt = np.zeros((d, d))
    sum_xt_xtT = np.zeros((d, d))

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
