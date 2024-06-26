import numpy as np
from plots import *
import pickle
from scipy.linalg import expm
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
        max_num_trajectories = int(base_params['num_trajectories'])
    else:
        max_num_trajectories = int(max(ablation_param['num_trajectories']))
    # filepath = os.path.join('Measurement_data', filename)
    # if os.path.exists(filepath):
    #     print(f"There is already saved measurement data under {filename}. ")
    #     return filename
    filepath = utils.find_existing_data(args, max_num_trajectories, max_T, min_dt, args.simulation_mode)
    if filepath:
        print(f"There is already saved measurement data under {filepath}. ")
        return filepath
    if args.simulation_mode == 'cell_death':
        print('generating trajectories from killed cells')
        with ProcessPoolExecutor(max_workers=5) as executor:
            futures = {executor.submit(generate_sde_data_cell_measurement, i, max_num_trajectories, max_T, min_dt,
                                       base_params): i for i in range(base_params['n_sdes'])}
            results = []
            # Create a tqdm progress bar for the futures as they complete
            for future in tqdm(as_completed(futures), total=base_params['n_sdes']):
                results.append(future.result())
        A_trues, G_trues, maximal_X_measured_list = zip(*[(res[0], res[1], res[2]) for res in results])
        filename = f'seed-{args.master_seed}_X0-{args.fixed_X0}_d-{args.d}_n_sdes-{args.n_sdes}_dt-{min_dt}_N-{max_num_trajectories}_T-{max_T}'
    elif args.simulation_mode == 'unkilled':
        print('generating unkilled trajectories')
        maximal_X_measured_list, A_trues, G_trues = [], [], []
        for i in tqdm(range(base_params['n_sdes'])):
            A_trues.append(utils.initialize_drift(base_params['d'], initialization_type=base_params['drift_initialization']))
            G_trues.append(utils.initialize_diffusion(base_params['d'], initialization_type=base_params['diffusion_initialization'],
                                           diffusion_scale=base_params['diffusion_scale']))
            maximal_X_measured_list.append(true_multi_ou_process(max_num_trajectories,  base_params['d'], max_T, base_params['dt_EM'], min_dt, A_trues[-1], G_trues[-1], X0 = base_params['X0']))
            # maximal_X_measured_list.append(multiple_ou_trajectories(max_num_trajectories, base_params['d'], max_T, min_dt, A_trues[-1], G_trues[-1], X0 = base_params['X0']))
            filename = f'unkilled_seed-{args.master_seed}_X0-{args.fixed_X0}_d-{args.d}_n_sdes-{args.n_sdes}_dt-{min_dt}_N-{max_num_trajectories}_T-{max_T}'

    utils.save_measurement_data(filename, base_params, ablation_param, A_trues, G_trues, maximal_X_measured_list, max_num_trajectories, max_T, min_dt)
    return A_trues, G_trues, maximal_X_measured_list, filename, max_num_trajectories, max_T, min_dt

def generate_sde_data_cell_measurement(i, max_num_trajectories, max_T, min_dt, base_params):
    np.random.seed(base_params['master_seed'] + i)
    A = utils.initialize_drift(base_params['d'], initialization_type=base_params['drift_initialization'])
    G = utils.initialize_diffusion(base_params['d'], initialization_type=base_params['diffusion_initialization'], diffusion_scale=base_params['diffusion_scale'])
    maximal_X_measured = generate_maximal_dataset_cell_measurement_death(max_num_trajectories, max_T, min_dt, base_params['d'], base_params['dt_EM'], A, G, base_params['X0'])
    print(f'A for SDE {i}:', A)
    return A, G, maximal_X_measured
def generate_maximal_dataset_cell_measurement_death(max_num_trajectories, max_T, min_dt, d, dt_EM, A, G, X0=None):
    n_measured_times = int(max_T / min_dt)
    X_measured = np.zeros((max_num_trajectories, n_measured_times, d))

    for i in range(n_measured_times):
        for n in range(max_num_trajectories):
            if isinstance(X0, np.ndarray):
                X0_ = X0
            if X0 is None or X0 == 'intermediate' and i == 0:
                X0_ = np.random.randn(d)
            if i == 0:
                X_measured[n, 0, :] = X0_
            else:
                # cell trajectory will terminate at i*dt
                measured_T = i * min_dt
                # use consistent X0s across the measured times
                if X0 == 'intermediate':
                    X_measured[n, i, :] = ou_process(measured_T, dt_EM, A, G, X_measured[n, 0, :])[-1]
                else:
                    # cell trajectory terminating at i*dt
                    X_measured[n, i, :] = ou_process(measured_T, dt_EM, A, G, X0_)[-1]
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

def true_multi_ou_process(num_trajectories, d, T, dt_EM, dt, A, G, X0=None):
    '''
    Models measurements of cell trajectories, such that cells don't die.
    We construct the ou_process by dt_EM and then just pick points from that where they are at i * dt position (T/ dt) of them.
    Trajectories are generated independently from each other. dt should be some number * dt-EM.
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
    rate = int(dt/ dt_EM)
    for n in range(num_trajectories):
        if X0 is None:
            # Generate a new random initial value for each trajectory
            X0_ = np.random.randn(d)
        else:
            X0_ = X0
        X_true = ou_process(T, dt_EM, A,G, X0_)

        for i in range(n_measured_times):
            X_measured[n, i, :] = X_true[i* rate, :]
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
    for i in range(n_measured_times):
        for n in range(num_trajectories):
            if i == 0:
                if X0 is None:
                    X0_ = np.random.randn(d)
                else:
                    X0_= X0
                X_measured[n, 0, :] = X0_
            else:
                measured_T = i * dt
                # cell trajectory terminating at i*dt
                X_measured[n, i, :] = ou_process(measured_T, dt_EM, A, G, X0)[-1]
    return X_measured




def ou_process_exact(T, dt, A, G, X0):
    """
    Simulate the Ornstein-Uhlenbeck process at a given time t.

    Parameters:
        A (numpy.ndarray): Drift matrix.
        G (numpy.ndarray): Diffusion matrix.
        X0 (numpy.ndarray): Initial value.
        t (float): Time at which to sample.
        dt (float): Time step size for discretization of the integral.

    Returns:
        numpy.ndarray: Sample from the process at time t.
    """
    # Compute the deterministic part
    exp_At = expm(A * T)
    deterministic_part = np.dot(exp_At, X0)

    # Compute the stochastic part
    num_steps = int(T / dt)
    stochastic_part = np.zeros_like(X0)
    for i in range(num_steps):
        s = i * dt
        exp_Ats = expm(A * (T - s))
        dWs = np.random.normal(0, np.sqrt(dt), size=G.shape[1])
        stochastic_part += np.dot(exp_Ats, G @ dWs)

    # Compute the final value
    Xt = deterministic_part + stochastic_part

    return Xt
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
            # Generate a new random initial value for each trajectory
            X0_ = np.random.randn(d)
        else:
            X0_ = X0
        trajectories[i] = ou_process(T, dt, A, G, X0_)

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




### old
# def multiple_ou_trajectories_cell_branching(num_trajectories, d, T, dt, A, G, init_population, pd=0.5, tau=0.02, X0=None):
#     num_steps = int(T / dt)
#     all_trajectories = np.zeros((num_trajectories, num_steps, d))
#     population_trajectories = [np.random.randn(d) if X0 is None else X0.copy() for _ in range(init_population)]
#
#     # Simulate through time
#     for step in range(num_steps):
#         new_population = []
#         if len(population_trajectories) < num_trajectories:
#             raise ValueError("Not enough remaining population to sample the required number of trajectories.")
#
#         # Process each cell in the current population
#         for X_prev in population_trajectories:
#             X_next = ou_step(X_prev, dt, A, G)
#             event_occurred = np.random.rand() < 1 - np.exp(
#                 -dt / tau)  # Probability that the clock rings in this time step
#
#             if event_occurred:  # Check if the clock rings
#                 if np.random.rand() < pd:  # Cell dies with probability pd
#                     continue  # Skip adding this cell to the new population, effectively "killing" it
#                 else:  # Cell divides with probability 1-pd
#                     new_population.append(X_next.copy())  # Add the original cell to the new population
#                     new_population.append(X_next.copy())  # Add the divided new cell to the new population
#             else:
#                 new_population.append(X_next)  # No event, cell continues to the next time step
#
#         # Update population trajectories for the next step
#         population_trajectories = new_population
#         # Randomly sample trajectories to output
#         sampled_indices = np.random.choice(len(population_trajectories), size=num_trajectories, replace=False)
#         for i, idx in enumerate(sampled_indices):
#             all_trajectories[i, step, :] = population_trajectories[idx]
#
#     return all_trajectories

# def multiple_ou_trajectories_cell(num_trajectories, d, T, dt, A, G, init_population, X0=None):
#     num_steps = int(T / dt)
#     all_trajectories = np.zeros((num_trajectories, num_steps, d))
#
#     # Initialize trajectories for initial population
#     population_trajectories = [ou_process(T, dt, A, G, np.random.randn(d) if X0 is None else X0) for _ in range(init_population)]
#     # print('num_traj:', num_trajectories)
#     # print('num_steps:', num_steps)
#     # Sample trajectories at each time step
#     for step in range(num_steps):
#         # print('current population:', len(population_trajectories))
#
#
#         if len(population_trajectories) < num_trajectories:
#             raise ValueError("Not enough remaining population to sample the required number of trajectories.")
#
#         sampled_indices = np.random.choice(len(population_trajectories), size=num_trajectories, replace=False)
#         for i, idx in enumerate(sampled_indices):
#             all_trajectories[i, step:] = population_trajectories[idx][step:]
#
#         # Remove sampled trajectories to simulate cell death
#         population_trajectories = [pop for j, pop in enumerate(population_trajectories) if j not in sampled_indices]
#
#     return all_trajectories






