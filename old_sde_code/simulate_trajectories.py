import numpy as np


# def simulate_trajectories(drift_type, noise_type, num_trajectories, T, dt, A, G, X0):
#     """
#     Generate multiple trajectories of a multidimensional Ornstein-Uhlenbeck process.
#
#     Parameters:
#         drift_type (str): The functional type of the drift for the SDE ex. linear
#         noise_type (
#         num_trajectories (int): Number of trajectories to simulate.
#         T (float): Total time period.
#         dt (float): Time step size.
#         A (numpy.ndarray): Drift matrix.
#         G (numpy.ndarray): Variance matrix (for case of additive noise)
#                          : List of variance matrices, one for each dimension (for case of multiplicative noise)
#         X0 (numpy.ndarray): Initial value for each trajectory.
#
#     Returns:
#         numpy.ndarray: 3D array where each "slice" corresponds to a single trajectory.
#     """
#     if noise_type == 'additive':
#
#
#     return

def ou_process(T, dt, A, G, X0):
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
    num_steps = int(T / dt)
    d = len(X0)
    m = G.shape[0]
    dW = np.sqrt(dt) * np.random.randn(num_steps, m)
    X = np.zeros((num_steps, d))
    X[0] = X0

    for t in range(1, num_steps):
        X[t] = X[t - 1] + dt * (A.dot(X[t - 1])) + G.dot(dW[t])

    return X

def multiple_ou_trajectories(num_trajectories, T, dt, A, G, X0):
    """
    Generate multiple trajectories of a multidimensional Ornstein-Uhlenbeck process.

    Parameters:
        num_trajectories (int): Number of trajectories to simulate.
        T (float): Total time period.
        dt (float): Time step size.
        A (numpy.ndarray): Drift matrix.
        G (numpy.ndarray): Variance matrix.
        X0 (numpy.ndarray): Initial value for each trajectory.


    Returns:
        numpy.ndarray: 3D array where each "slice" corresponds to a single trajectory.
    """
    num_steps = int(T / dt)
    num_dimensions = len(X0)
    # Create a 3D array to store all trajectories
    trajectories = np.zeros((num_trajectories, num_steps, num_dimensions))

    for i in range(num_trajectories):
        trajectories[i] = ou_process(T, dt, A, G, X0)

    return trajectories

def multiplicative_noise_process(T, dt, A, G, X0):
    """
    Simulate a single trajectory of a multidimensional linear SDE with multiplicative noise:
    dX_t = A X_t dt + G(X_t) dW_t

    Parameters:
        T (float): Total time period.
        dt (float): Time step size.
        A (numpy.ndarray): Drift matrix.
        G (list of numpy.ndarray): List of variance matrices, one for each dimension.
        X0 (numpy.ndarray): Initial value.

    Returns:
        numpy.ndarray: Array of simulated trajectories.
    """
    num_steps = int(T / dt)
    num_dimensions = len(X0)
    dW = np.sqrt(dt) * np.random.randn(num_steps, num_dimensions)
    X = np.zeros((num_steps, num_dimensions))
    X[0] = X0

    for t in range(1, num_steps):
        GXt_dW = np.zeros(num_dimensions)
        for i in range(num_dimensions):
            GXt_dW += G[i].dot(X[t-1]) * dW[t, i]
        X[t] = X[t-1] + dt * (A.dot(X[t-1])) + GXt_dW

    return X

def multiple_multiplicative_noise_trajectories(num_trajectories, T, dt, A, G, X0):
    """
    Generate multiple trajectories of a multidimensional Ornstein-Uhlenbeck process with multiplicative noise.

    Parameters:
        num_trajectories (int): Number of trajectories to simulate.
        T (float): Total time period.
        dt (float): Time step size.
        A (numpy.ndarray): Drift matrix.
        G (list of numpy.ndarray): List of variance matrices, one for each dimension.
        X0 (numpy.ndarray): Initial value for each trajectory.

    Returns:
        numpy.ndarray: 3D array where each "slice" corresponds to a single trajectory.
    """
    num_steps = int(T / dt)
    num_dimensions = len(X0)
    trajectories = np.zeros((num_trajectories, num_steps, num_dimensions))

    for i in range(num_trajectories):
        trajectories[i] = multiplicative_noise_process(T, dt, A, G, X0)

    return trajectories


def select_drift_matrix(desired_eigenvalues):
    """
    Select a diagonal drift matrix A with diagonal entries from desired eigenvalues

    Parameters:
        desired_eigenvalues (list or numpy.ndarray): Desired eigenvalues for stability.

    Returns:
        numpy.ndarray: Drift matrix A.
    """
    # Construct A using desired eigenvalues
    A = np.diag(desired_eigenvalues)

    return A


def generate_negative_eigenvalue_matrix(dimension, magnitude=1):
    """
    Generate a matrix with all negative eigenvalues.

    Parameters:
        dimension (int): The dimension of the square matrix.
        magnitude (int): The magnitude factor for the diagonal dominance.

    Returns:
        numpy.ndarray: A square matrix with all negative eigenvalues, validated by eigenvalue check.
    """
    max_attempts = 100  # Limit the number of attempts to prevent infinite loops
    for attempt in range(max_attempts):
        # Generate a random matrix
        A = np.random.randn(dimension, dimension)

        # Adjust the diagonal entries to ensure negative eigenvalues
        for i in range(dimension):
            # Making diagonal entries negatively dominant
            # Subtract the sum of absolute values of row elements (excluding diagonal) and add a magnitude
            A[i, i] = -np.abs(A[i, i]) - np.sum(np.abs(A[i, :])) + np.abs(A[i, i]) - magnitude

        # Check if all eigenvalues are negative
        eigenvalues = np.linalg.eigvals(A)
        if np.all(eigenvalues < 0):
            return A

    raise ValueError("Failed to generate a matrix with all negative eigenvalues after {} attempts.".format(max_attempts))

def generate_random_matrix(dimension, low=-1, high=1):
    """
    Generate a matrix with entries uniformly sampled from a specified interval.

    Parameters:
        dimension (int): The dimension of the square matrix.
        low (float): The lower bound of the uniform distribution interval.
        high (float): The upper bound of the uniform distribution interval.

    Returns:
        numpy.ndarray: A matrix with randomly sampled entries.
    """
    # Generate a random matrix with entries uniformly sampled from the interval [low, high]
    A = np.random.uniform(low, high, (dimension, dimension))

    return A


def extract_marginals(trajectories):
    """
    Extract marginal distributions from a 3D trajectory array.

    Parameters:
        trajectories (numpy.ndarray): 3D array of trajectories (num_trajectories, num_steps, num_dimensions).

    Returns:
        list of numpy.ndarray: Each element is an array containing samples from the marginal distribution at each time step.
    """
    num_trajectories, num_steps, num_dimensions = trajectories.shape
    marginals = []

    for t in range(num_steps):
        # Extract all samples at time t from each trajectory
        marginal_at_t = trajectories[:, t, :]
        marginals.append(marginal_at_t)

    return marginals