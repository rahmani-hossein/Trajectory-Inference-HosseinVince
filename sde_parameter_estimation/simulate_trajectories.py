import numpy as np
from plots import *
import pickle
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm.auto import tqdm
import utils

def create_measurement_data(args, base_params, ablation_param):
    if 'T' in base_params:
        max_T = base_params['T']
    else:
        max_T = max(ablation_param['T'])
    if 'dt' in base_params:
        min_dt = base_params['dt']
    else:
        min_dt = min(ablation_param['dt'])
    if 'num_trajectories' in base_params:
        max_num_trajectories = base_params['num_trajectories']
    else:
        max_num_trajectories = max(ablation_param['num_trajectories'])
    with ProcessPoolExecutor() as executor:
        with ProcessPoolExecutor() as executor:
            futures = {executor.submit(generate_sde_data_cell_measurement, i, max_num_trajectories, max_T, min_dt,
                                       base_params): i for i in range(base_params['n_sdes'])}
            results = []
            # Create a tqdm progress bar for the futures as they complete
            for future in tqdm(as_completed(futures), total=base_params['n_sdes']):
                results.append(future.result())
    A_trues, G_trues, maximal_X_measured_list = zip(*[(res[0], res[1], res[2]) for res in results])
    if args.save_measurements:
        filename = f'measured_seed-{args.master_seed}_ablate_{args.ablation_variable_name}_d-{args.d}_N-{max_num_trajectories}_dt-{min_dt}_T-{max_T}'
        utils.save_measurement_data(filename, base_params, ablation_param, A_trues, G_trues, maximal_X_measured_list)
    return A_trues, G_trues, maximal_X_measured_list

def generate_sde_data_cell_measurement(i, max_num_trajectories, max_T, min_dt, base_params):
    np.random.seed(base_params['master_seed'] + i)
    A = utils.initialize_drift(base_params['d'], initialization_type=base_params['drift_initialization'])
    G = utils.initialize_diffusion(base_params['d'], initialization_type=base_params['diffusion_initialization'], diffusion_scale=base_params['diffusion_scale'])
    maximal_X_measured = generate_maximal_dataset_cell_measurement_death(max_num_trajectories, max_T, min_dt, base_params['d'], base_params['dt_EM'], A, G, base_params['X0'])
    print(f'A for {i}th:', A)
    return A, G, maximal_X_measured
def generate_maximal_dataset_cell_measurement_death(max_num_trajectories, max_T, min_dt, d, dt_EM, A, G, X0=None):
    n_measured_times = int(max_T / min_dt)
    X_measured = np.zeros((max_num_trajectories, n_measured_times, d))
    if X0 is None:
        X0 = np.random.randn(d)

    for i in range(n_measured_times):
        for n in range(max_num_trajectories):
            if i == 0:
                X_measured[n, 0, :] = X0
            else:
                measured_T = i * min_dt
                # cell trajectory terminating at i*dt
                X_measured[n, i, :] = ou_process(measured_T, dt_EM, A, G, X0)[-1]
    return X_measured

def multiple_ou_trajectories_cell_measurement_death(num_trajectories, d, T, dt_EM, dt, A, G, X0=None):
    '''
    Models measurements of cell trajectories, such that cells die when measured
    Cell trajectories are modeled by taking samples of the discretized SDE with time granularity dt_EM
    We assume that exactly num_trajectories cells are measured at each measured time, which are equally
    spaced according to time granularity dt (dt >= dt_EM)
    To efficiently simulate the measurements, we generate num_trajectories cell trajectories up until each
    time point dt*i
    Args:
        num_trajectories: number of trajectories
        d: dimension of process
        T: Total time period
        dt_EM: discretization time step used for simulating the raw cell trajectories
        dt: discretization time step of the measurements
        A (numpy.ndarray): Drift matrix.
        G (numpy.ndarray): Variance matrix.
        X0 (numpy.ndarray): Initial value.

    Returns:
    X_measured

    '''
    n_measured_times = int(T / dt)
    X_measured = np.zeros((num_trajectories, n_measured_times, d))
    if X0 is None:
        X0 = np.random.randn(d)
    for i in range(n_measured_times):
        for n in range(num_trajectories):
            if i == 0:
                X_measured[n, 0, :] = X0
            else:
                measured_T = i * dt
                # cell trajectory terminating at i*dt
                X_measured[n, i, :] = ou_process(measured_T, dt_EM, A, G, X0)[-1]
    return X_measured

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




### old
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



# def generate_sde_data_cell_measurement(i, d, drift_initialization, diffusion_initialization, diffusion_scale, num_trajectories, T, dt_EM, dt, X0, master_seed):
#     np.random.seed(master_seed+ i)  # Set seed for reproducibility
#     A = initialize_drift(d, initialization_type=drift_initialization)
#     G = initialize_diffusion(d, initialization_type=diffusion_initialization, diffusion_scale=diffusion_scale)
#     X_measured = multiple_ou_trajectories_cell_measurement_death(num_trajectories, d, T, dt_EM, dt, A, G, X0)
#     return A, G, X_measured






