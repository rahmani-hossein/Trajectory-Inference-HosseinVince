from plots import *
from scipy.linalg import expm
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm.auto import tqdm
import math
import utils.simulation
from utils.load_save import *


def create_measurement_data(args, base_params, ablation_param):
    D = base_params['D']
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
    filepath = find_existing_data(args, max_num_trajectories, max_T, min_dt, D, args.simulation_mode)
    if filepath:
        print(f"There is already saved measurement data under {filepath}. ")
        return filepath
    if args.simulation_mode == 'killed':
        print('generating trajectories from killed cells')
        with ProcessPoolExecutor(max_workers=5) as executor:
            futures = {
                executor.submit(generate_sde_data_cell_measurement, i, max_num_trajectories, max_T, min_dt, base_params,
                                args.saved_drifts_diffusions_file, i): i for i in range(base_params['n_sdes'])}
            results = []
            for future in tqdm(as_completed(futures), total=base_params['n_sdes']):
                results.append(future.result())
        A_trues, G_trues, maximal_X_measured_list = zip(*[(res[0], res[1], res[2]) for res in results])
        filename = f'seed-{args.master_seed}_X0-{args.fixed_X0}_d-{args.d}_n_sdes-{args.n_sdes}_dt-{min_dt}_N-{max_num_trajectories}_T-{max_T}_D-{D}'
    elif args.simulation_mode == 'unkilled':
        print('generating unkilled trajectories')
        maximal_X_measured_list, A_trues, G_trues = [], [], []
        for i in tqdm(range(base_params['n_sdes'])):
            A_trues.append(utils.simulation.initialize_drift(base_params['d'],
                                                             initialization_type=base_params['drift_initialization'],
                                                             saved_drifts_diffusions_file=args.saved_drifts_diffusions_file,
                                                             save_index=i))
            G_trues.append(utils.simulation.initialize_diffusion(base_params['d'], initialization_type=base_params[
                'diffusion_initialization'],
                                                                 diffusion_scale=base_params['D'],
                                                                 saved_drifts_diffusions_file=args.saved_drifts_diffusions_file,
                                                                 save_index=i))
            maximal_X_measured_list.append(
                true_multi_ou_process(max_num_trajectories, base_params['d'], max_T, base_params['dt_EM'], min_dt,
                                      A_trues[-1], G_trues[-1], X0=base_params['X0']))
            # maximal_X_measured_list.append(multiple_ou_trajectories(max_num_trajectories, base_params['d'], max_T, min_dt, A_trues[-1], G_trues[-1], X0 = base_params['X0']))
            filename = f'unkilled_seed-{args.master_seed}_X0-{args.fixed_X0}_d-{args.d}_n_sdes-{args.n_sdes}_dt-{min_dt}_N-{max_num_trajectories}_T-{max_T}_D-{D}'

    save_measurement_data(filename, base_params, ablation_param, A_trues, G_trues, maximal_X_measured_list,
                          max_num_trajectories, max_T, min_dt)
    return A_trues, G_trues, maximal_X_measured_list, filename, max_num_trajectories, max_T, min_dt


def generate_sde_data_cell_measurement(i, max_num_trajectories, max_T, min_dt, base_params,
                                       saved_drifts_diffusions_file=None, save_idx=None):
    np.random.seed(base_params['master_seed'] + i)
    A = utils.simulation.initialize_drift(base_params['d'], initialization_type=base_params['drift_initialization'],
                                          saved_drifts_diffusions_file=saved_drifts_diffusions_file,
                                          save_index=save_idx)
    G = utils.simulation.initialize_diffusion(base_params['d'],
                                              initialization_type=base_params['diffusion_initialization'],
                                              diffusion_scale=base_params['D'],
                                              saved_drifts_diffusions_file=saved_drifts_diffusions_file,
                                              save_index=save_idx)
    maximal_X_measured = generate_maximal_dataset_cell_measurement_death(max_num_trajectories, max_T, min_dt,
                                                                         base_params['d'], base_params['dt_EM'], A, G,
                                                                         base_params['X0'])
    print(f'A for SDE {i}:', A)
    return A, G, maximal_X_measured


def generate_maximal_dataset_cell_measurement_death(max_num_trajectories, max_T, min_dt, d, dt_EM, A, G, X0=None,
                                                    stationary=False):
    n_measured_times = int(max_T / min_dt)
    X_measured = np.zeros((max_num_trajectories, n_measured_times, d))
    if stationary:
        drift_scale = np.linalg.norm(A)
        D = np.linalg.norm(np.matmul(G, np.transpose(G)))
        if drift_scale > 0:
            stationary_variance = D / (2 * d * drift_scale)
        else:
            stationary_variance = D / (2 * d * 1e-10)

    for i in range(n_measured_times):
        for n in range(max_num_trajectories):
            if isinstance(X0, np.ndarray):
                X0_ = X0
            elif X0 is None or X0 == 'intermediate' and i == 0:
                X0_ = np.random.randn(d)
            if i == 0 and stationary == False:
                X_measured[n, 0, :] = X0_
            elif i == 0 and stationary == True:
                cov = np.identity(d) * stationary_variance
                X_measured[n, 0, :] = np.random.multivariate_normal(np.zeros(d), cov)
            else:
                # cell trajectory will terminate at i*dt
                measured_T = i * min_dt
                # use consistent X0s across the measured times
                if isinstance(X0, np.ndarray):
                    X_measured[n, i, :] = ou_process(measured_T, dt_EM, A, G, X0_)[-1]
                else:
                    if X0 == 'intermediate':
                        X_measured[n, i, :] = ou_process(measured_T, dt_EM, A, G, X_measured[n, 0, :])[-1]
                    else:
                        # cell trajectory terminating at i*dt
                        if stationary:
                            X_measured[n, i, :] = ou_process(measured_T, dt_EM, A, G, np.array(
                                [np.random.randn() * math.sqrt(stationary_variance) for i in range(d)]))[-1]
                        else:
                            X_measured[n, i, :] = ou_process(measured_T, dt_EM, A, G, X0_)[-1]
    return X_measured


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


def true_multi_ou_process(num_trajectories, d, T, dt_EM, dt, A, G, X0=None, X0_prob_dist=None, stationary=False,
                          beps=True, exact=False):
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
        X0 (numpy.ndarray): Initial value or distribution.
        X0_prob_dist (list of tuples): List of tuples, each containing an initial value and its associated probability.
        stationary (bool): Whether to use the stationary distribution for initial conditions.
        beps (bool): Boolean flag for specific covariance structure.
        exact (bool): Whether to use the exact solution or the Euler-Maruyama method.

    Returns:
        numpy.ndarray: Measured trajectories.
    '''
    n_measured_times = int(T / dt)
    X_measured = np.zeros((num_trajectories, n_measured_times, d))
    rate = int(dt / dt_EM)

    for n in range(num_trajectories):
        if X0_prob_dist is not None:
            # Sample X0 based on the provided probability distribution
            X0_ = X0_prob_dist[np.random.choice(len(X0_prob_dist), p=[prob for _, prob in X0_prob_dist])][0]
        elif X0 is None:
            # if beps:
            #     cov_matrix = [[1, 0], [0, 1 / 1]]
            # else:
            #     cov_matrix = [[1, 0], [0, 1 / 2]]
            cov_matrix = np.eye(d)
            X0_ = np.random.multivariate_normal(np.zeros(d), cov_matrix)
        elif stationary:
            drift_scale = np.linalg.norm(A)
            D = np.linalg.norm(np.matmul(G, np.transpose(G)))
            if drift_scale > 0:
                stationary_variance = D / (2 * d * drift_scale)
            else:
                stationary_variance = D / (2 * d * 1e-10)
            cov = np.identity(d) * stationary_variance
            X0_ = np.random.multivariate_normal(np.zeros(d), cov)
        else:
            X0_ = X0

        if exact:
            X_true = ou_process_exact(T, dt_EM, A, G, X0_)
        else:
            X_true = ou_process(T, dt_EM, A, G, X0_)

        for i in range(n_measured_times):
            X_measured[n, i, :] = X_true[i * rate, :]

    return X_measured




# def true_multi_ou_process(num_trajectories, d, T, dt_EM, dt, A, G, X0=None, stationary = False, beps = True, exact = False):
#     '''
#     Models measurements of cell trajectories, such that cells don't die.
#     We construct the ou_process by dt_EM and then just pick points from that where they are at i * dt position (T/ dt) of them.
#     Trajectories are generated independently from each other. dt should be some number * dt-EM.
#     Args:
#         num_trajectories: number of trajectories
#         d: dimension of process
#         T: Total time period
#         dt_EM: discretization time step used for simulating the raw cell trajectories
#         dt: discretization time step of the measurements
#         A (numpy.ndarray): Drift matrix.
#         G (numpy.ndarray): Variance matrix.
#         X0 (numpy.ndarray): Initial value.
#
#     Returns:
#     X_measured
#
#     '''
#     n_measured_times = int(T / dt)
#     X_measured = np.zeros((num_trajectories, n_measured_times, d))
#     rate = int(dt/ dt_EM)
#     for n in range(num_trajectories):
#
#         if X0 is None:
#             # Generate a new random initial value for each trajectory
#             # X0_ = np.random.randn(d)
#             # x = np.random.uniform(-2,2)
#             # sign = np.random.uniform(-1,1)
#             # if sign < 0:
#             #     sign = -1
#             # else:
#             #     sign = 1
#             # y = math.sqrt((4-x**2)/2) * sign
#             # X0_ = np.array([x,y])
#
#             # sigma_y = 1  # You can set this to any value you'd like
#             if beps:
#                 cov_matrix = [[1, 0], [0, 1 / 1]]
#             else:
#                 cov_matrix = [[1, 0], [0, 1 / 2]]
#             # cov_matrix = [[math.sqrt(2)*sigma_y ** 2, 0], [0, sigma_y ** 2]]
#             X0_=np.random.multivariate_normal([0,0], cov_matrix)
#             # Generate X0_ as either (1, 0) or (-1, 0) with 50% probability each
#             sign = np.random.uniform(-1,1)
#             if sign > -5:
#                 X0_ = np.random.choice([1, -1], p=[0.5, 0.5]) * np.array([1, 0])
#             else:
#                 X0_ = np.random.choice([1, -1], p=[0.5, 0.5]) * np.array([0, 1])
#             # X0_ = np.random.choice([1, -1], p=[0.5, 0.5]) * np.array([1, 0])
#         elif stationary:
#             drift_scale = np.linalg.norm(A)
#             D = np.linalg.norm(np.matmul(G, np.transpose(G)))
#             if drift_scale> 0:
#                 stationary_variance = D / (2 * d * drift_scale)
#             else:
#                 stationary_variance = D / (2 * d * 1e-10)
#             cov = np.identity(d) * stationary_variance
#             X0_ = np.random.multivariate_normal(np.zeros(d), cov)
#         else:
#             X0_ = X0
#         if exact:
#             X_true = ou_process_exact(T, dt_EM, A, G, X0_)
#         else:
#             X_true = ou_process(T, dt_EM, A,G, X0_)
#
#         for i in range(n_measured_times):
#             X_measured[n, i, :] = X_true[i* rate, :]
#     return X_measured


import numpy as np
from scipy.linalg import expm


def ou_process_exact(T, dt, A, G, X0, seed=None):
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


# def ou_process_exact(T, dt, A, G, X0):
#     """
#     Simulate the Ornstein-Uhlenbeck process at a given time t.
#
#     Parameters:
#         A (numpy.ndarray): Drift matrix.
#         G (numpy.ndarray): Diffusion matrix.
#         X0 (numpy.ndarray): Initial value.
#         t (float): Time at which to sample.
#         dt (float): Time step size for discretization of the integral.
#
#     Returns:
#         numpy.ndarray: Sample from the process at time t.
#     """
#     # Compute the deterministic part
#     exp_At = expm(A * T)
#     deterministic_part = np.dot(exp_At, X0)
#     num_steps = int(T / dt)
#     X = np.zeros((num_steps, d))
#     X[0] = X0
#     # Compute the stochastic part
#     num_steps = int(T / dt)
#     stochastic_part = np.zeros_like(X0)
#     for i in range(num_steps):
#         s = i * dt
#         exp_Ats = expm(A * (T - s))
#         dWs = np.random.normal(0, np.sqrt(dt), size=G.shape[1])
#         stochastic_part += np.dot(exp_Ats, G @ dWs)
#
#
#     # Compute the final value
#     Xt = deterministic_part + stochastic_part
#
#     return Xt

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
            GXt_dW += G[i].dot(X[t - 1]) * dW[t, i]
        X[t] = X[t - 1] + dt * (A.dot(X[t - 1])) + GXt_dW

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
