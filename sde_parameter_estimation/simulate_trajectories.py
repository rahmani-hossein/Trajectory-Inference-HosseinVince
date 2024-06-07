import numpy as np

def initialize_drift(d, initialization_type = 'uniform_random_entries', high= 1, low = -1, magnitude = 1, sparsity = None):
    if initialization_type == 'uniform_random_entries':
        A = generate_random_matrix(d, high = high, low = low)
    elif initialization_type == 'negative_eigenvalue':
        A = generate_negative_eigenvalue_matrix(d, magnitude = magnitude)
    return A

def initialize_diffusion(d, initialization_type = 'scaled_identity', diffusion_scale = 0.1, high= 1, low = -1, sparsity = None):
    if initialization_type == 'scaled_identity':
        G = diffusion_scale*np.eye(d)
    elif initialization_type == 'uniform_random_entries':
        G = generate_random_matrix(d, high = high, low = low)
    return G


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

def ou_step(X_prev, dt, A, G):
    """
    Perform a single step simulation of an Ornstein-Uhlenbeck process.

    Parameters:
        X_prev (numpy.ndarray): Previous state of the process.
        dt (float): Time step size.
        A (numpy.ndarray): Drift matrix.
        G (numpy.ndarray): Variance matrix.

    Returns:
        numpy.ndarray: Next state of the process.
    """
    d = len(X_prev)
    dW = np.random.normal(scale=np.sqrt(dt), size=d)
    return X_prev + A @ X_prev * dt + np.sqrt(G) @ dW

def multiple_ou_trajectories_cell_branching(num_trajectories, d, T, dt, A, G, init_population, pd=0.5, tau=0.02, X0=None):
    num_steps = int(T / dt)
    all_trajectories = np.zeros((num_trajectories, num_steps, d))
    population_trajectories = [np.random.randn(d) if X0 is None else X0.copy() for _ in range(init_population)]

    # Simulate through time
    for step in range(num_steps):
        new_population = []
        if len(population_trajectories) < num_trajectories:
            raise ValueError("Not enough remaining population to sample the required number of trajectories.")

        # Process each cell in the current population
        for X_prev in population_trajectories:
            X_next = ou_step(X_prev, dt, A, G)
            event_occurred = np.random.rand() < 1 - np.exp(
                -dt / tau)  # Probability that the clock rings in this time step

            if event_occurred:  # Check if the clock rings
                if np.random.rand() < pd:  # Cell dies with probability pd
                    continue  # Skip adding this cell to the new population, effectively "killing" it
                else:  # Cell divides with probability 1-pd
                    new_population.append(X_next.copy())  # Add the original cell to the new population
                    new_population.append(X_next.copy())  # Add the divided new cell to the new population
            else:
                new_population.append(X_next)  # No event, cell continues to the next time step

        # Update population trajectories for the next step
        population_trajectories = new_population
        # Randomly sample trajectories to output
        sampled_indices = np.random.choice(len(population_trajectories), size=num_trajectories, replace=False)
        for i, idx in enumerate(sampled_indices):
            all_trajectories[i, step, :] = population_trajectories[idx]

    return all_trajectories

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

def multiple_ou_trajectories_cell(num_trajectories, d, T, dt, A, G, init_population, X0=None):
    num_steps = int(T / dt)
    all_trajectories = np.zeros((num_trajectories, num_steps, d))

    # Initialize trajectories for initial population
    population_trajectories = [ou_process(T, dt, A, G, np.random.randn(d) if X0 is None else X0) for _ in range(init_population)]
    # print('num_traj:', num_trajectories)
    # print('num_steps:', num_steps)
    # Sample trajectories at each time step
    for step in range(num_steps):
        # print('current population:', len(population_trajectories))
        if len(population_trajectories) < num_trajectories:
            raise ValueError("Not enough remaining population to sample the required number of trajectories.")

        sampled_indices = np.random.choice(len(population_trajectories), size=num_trajectories, replace=False)
        for i, idx in enumerate(sampled_indices):
            all_trajectories[i, step:] = population_trajectories[idx][step:]

        # Remove sampled trajectories to simulate cell death
        population_trajectories = [pop for j, pop in enumerate(population_trajectories) if j not in sampled_indices]

    return all_trajectories
def multiple_ou_trajectories(num_trajectories, d, T, dt, A, G, X0 = None):
    """
    Generate multiple trajectories of a multidimensional Ornstein-Uhlenbeck process.

    Parameters:
        num_trajectories (int): Number of trajectories to simulate.
        d (int): Dimension of process
        T (float): Total time period.
        dt (float): Time step size.
        A (numpy.ndarray): Drift matrix.
        G (numpy.ndarray): Variance matrix.
        X0 (numpy.ndarray): Fixed initial value for each trajectory. If None, then sample a random X0 for each trajectory


    Returns:
        numpy.ndarray: 3D array where each "slice" corresponds to a single trajectory.
    """
    num_steps = int(T / dt)
    # Create a 3D array to store all trajectories
    trajectories = np.zeros((num_trajectories, num_steps, d))

    for i in range(num_trajectories):
        if X0 is None:
            X0 = np.random.randn(d)
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


