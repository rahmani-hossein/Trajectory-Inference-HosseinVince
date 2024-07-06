import os
import json
import numpy as np
import pickle
from scipy.spatial.distance import mahalanobis
from scipy.stats import chi2
from tqdm import tqdm


# from parameter_estimation import estimate_A_compare_methods


def extract_measurement_parameters(args):
    """
    Extract parameters from the args object and categorize them into base and ablation parameters.

    Args:
        args: An object containing simulation and measurement parameters.

    Returns:
        A tuple of two dictionaries:
        - base_params: Parameters that do not vary across simulations.
        - ablation_param: Parameter that may vary
    """
    # Initialize the random seed for reproducibility
    master_seed = args.master_seed
    np.random.seed(master_seed)
    # Base parameters
    base_params = {
        'master_seed': int(master_seed),
        'n_sdes': int(args.n_sdes),
        'd': int(args.d),
        'drift_initialization': args.drift_initialization,
        'diffusion_initialization': args.diffusion_initialization,
        'diffusion_scale': float(args.diffusion_scale),
        'T': float(args.T),
        'dt_EM': float(args.dt_em),
        'dt': float(args.dt),
        'num_trajectories': int(args.num_trajectories),
        'X0': initialize_X0(args.fixed_X0, args.d)
    }
    # print(f'Measured data comprises {args.num_trajectories} observations from {int(args.T/args.dt)} time points (T={args.T}, dt={args.dt})')
    simulation_measurement_variables = ['T', 'dt_EM', 'dt', 'num_trajectories']
    if args.ablation_variable_name in simulation_measurement_variables:
        base_params.pop(args.ablation_variable_name)
    ablation_values = [float(item) for item in args.ablation_values.split(',')]
    ablation_param = {
        args.ablation_variable_name: ablation_values
    }
    print(f'Our experiment considers the variable {args.ablation_variable_name} across the values {ablation_values}')
    print(f'The evaluated parameter estimation methods are {args.methods}')
    return base_params, ablation_param


def extract_estimation_parameters(args):
    estimation_params = {
        'entropy_reg': args.entropy_reg,
        'n_iterations': args.n_iterations
    }
    parameter_estimation_variables = ['n_iterations', 'entropy_reg']
    if args.ablation_variable_name in parameter_estimation_variables:
        estimation_params.pop(args.ablation_variable_name)
    return estimation_params


def initialize_X0(fixed_X0, d):
    if fixed_X0 == 'none':
        print('we will have random X0 for each trajectory')
        return None
    if fixed_X0 == 'intermediate':
        print('the trajectories measured at each time will start at the same random set of X0s ')
        return 'intermediate'
    elif fixed_X0 == 'ones':
        print('each X0 will start at (1,...,1)')
        return np.ones(d)
    elif fixed_X0 == 'zeros':
        return np.zeros(d)
        print('each X0 will start at (0,...,0)')
    else:
        raise ValueError(f"Unsupported X0 initialization: {fixed_X0}")


def initialize_drift(d, initialization_type='uniform_random_entries', high=1, low=-1, magnitude=1, sparsity=None):
    if initialization_type == 'uniform_random_entries':
        A = generate_random_matrix(d, high=high, low=low)
    elif initialization_type == 'negative_eigenvalue':
        A = generate_negative_eigenvalue_matrix(d, magnitude=magnitude)
    return A


def initialize_diffusion(d, initialization_type='scaled_identity', diffusion_scale=0.1, high=1, low=-1, sparsity=None):
    if initialization_type == 'scaled_identity':
        G = diffusion_scale * np.eye(d)
    elif initialization_type == 'uniform_random_entries':
        G = generate_random_matrix(d, high=high, low=low)
    return G


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

    raise ValueError(
        "Failed to generate a matrix with all negative eigenvalues after {} attempts.".format(max_attempts))


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


def estimate_gaussian_marginal(X_t):
    """ Estimate Gaussian parameters from samples. """
    fit_mean = np.mean(X_t, axis=0)
    fit_cov = np.cov(X_t, rowvar=False)
    return fit_mean, fit_cov


def gaussian_outer_product(fit_mean, fit_cov):
    """ Compute the outer product using Gaussian parameters. """
    d = len(fit_mean)
    outer_prod = np.outer(fit_mean, fit_mean) + fit_cov
    return outer_prod


def calculate_weights(X_t, fit_mean, fit_cov):
    """ Calculate weights for each sample based on Mahalanobis distance. """
    inv_covmat = np.linalg.inv(fit_cov)
    weights = np.array([np.exp(-0.5 * mahalanobis(x, fit_mean, inv_covmat) ** 2) for x in X_t])
    weights /= np.sum(weights)  # Normalize the weights
    return weights


def preprocess_measured_data(maximal_X_measured_list, ablation_values, max_num_trajectories, max_T, min_dt, args, measurement_ablation):
    if measurement_ablation:
        X_measured_ablation_dict = {}
        if args.ablation_variable_name == 'num_trajectories':
            for num_trajectories in ablation_values:
                step_ratio = int(args.dt / min_dt)
                num_steps = int(args.T / min_dt)
                X_measured_ablation_dict[num_trajectories] = [maximal_X_measured[: int(num_trajectories), :num_steps:step_ratio, :] for maximal_X_measured in
                                               maximal_X_measured_list]
        if args.ablation_variable_name == 'T':
            for T in ablation_values:
                num_steps = int(T / args.dt)
                step_ratio = int(args.dt / min_dt)
                X_measured_ablation_dict[T] = [maximal_X_measured[: args.num_trajectories, :num_steps:step_ratio, :] for maximal_X_measured in
                                               maximal_X_measured_list]
        if args.ablation_variable_name == 'dt':
            # loop over each ablation value
            for dt in ablation_values:
                num_steps = int(args.T / min_dt)
                step_ratio = int(dt / min_dt)
                X_measured_ablation_dict[dt] = [maximal_X_measured[:args.num_trajectories, :num_steps:step_ratio, :] for maximal_X_measured in
                                                maximal_X_measured_list]
        return X_measured_ablation_dict
    else:
        step_ratio = int(args.dt / min_dt)
        num_steps = int(args.T / min_dt)
        X_measured_list = [maximal_X_measured[: args.num_trajectories, :num_steps:step_ratio, :]
                                                      for maximal_X_measured in
                                                      maximal_X_measured_list]
        return X_measured_list

def find_existing_data(args, max_num_trajectories, max_T, min_dt, simulation_mode = 'cell_death'):
    directory = 'Measurement_data'
    if simulation_mode == 'cell_death':
        pattern = f'seed-{args.master_seed}_X0-{args.fixed_X0}'
        existing_files = [f for f in os.listdir(directory) if f.startswith(pattern)]
        offset = 0
    elif simulation_mode == 'unkilled':
        pattern = f'unkilled_seed-{args.master_seed}_X0-{args.fixed_X0}'
        existing_files = [f for f in os.listdir(directory) if f.startswith(pattern)]
        offset = 1
    # Check each file to see if it meets the conditions
    for filename in existing_files:
        # Extract parts from the filename, assuming a specific naming convention
        parts = filename.replace('.pkl', '').split('_')
        d = int(parts[2 + offset].split('-')[1])
        n_sdes = int(parts[4 + offset].split('-')[1])
        dt = float(parts[5 + offset].split('-')[1])
        # Example filename: "seed-0_d-3_n_sdes-10_dt-0.02_N-50_T-1.0"
        num_trajectories = int(parts[6 + offset].split('-')[1])
        T = float(parts[7 + offset].split('-')[1])
        if d == args.d and n_sdes >= args.n_sdes and num_trajectories >= max_num_trajectories and T >= max_T and dt <= min_dt:
            print('dt from filename:', dt )
            print('min dt:', min_dt)
            exit
            return os.path.join(filename)

    return None


def save_measurement_data(filename, base_params, ablation_param, A_trues, G_trues, maximal_X_measured_list,
                          max_num_trajectories, max_T, min_dt):
    '''
    Args:
        filename:
        base_params: dictionary of non-abalation parameters related to simulation and measurement
        ablation_param: dictionary storing the values of the ablation parameter (T, dt, num_trajectories, or None)
        A_trues: list of true drift matrices A for each SDE
        G_trues: list of true diffusion matrices G for each SDE
        maximal_X_measured_list: list of the measured data for each SDE
        max_num_trajectories:
        max_T:
        min_dt:


    Returns:
        saves the measurements and true SDE parameters
    '''
    os.makedirs('Measurement_data', exist_ok=True)
    filepath = os.path.join('Measurement_data', filename)
    data = {
        'ablation': ablation_param,
        'base': base_params,
        'A_trues': A_trues,
        'G_trues': G_trues,
        'max_num_trajectories': max_num_trajectories,
        'max_T': max_T,
        'min_dt': min_dt,
        'maximal_X_measured': maximal_X_measured_list
    }
    with open(filepath, 'wb') as f:
        pickle.dump(data, f)
    # save_dict = base_params
    # save_dict.update(ablation_param)
    # save_dict.update({
    #         'A_trues': A_trues,
    #         'G_trues': G_trues,
    #         'maximal_X_measured': maximal_X_measured_list,
    #         })
    # with open(filepath, 'wb') as f:
    #     pickle.dump(save_dict, f)
    print(f"Data generation complete and saved in {filename}.")


def save_detailed_experiment_data(filename, data):
    os.makedirs('../MSE_detailed_logs', exist_ok=True)
    filepath = os.path.join('../MSE_detailed_logs', filename)
    with open(filepath, 'wb') as f:
        pickle.dump(data, f)


def load_measurement_data(filename, verbose = False):
    '''
    Load measurement data and parameters from a file.

    Args:
        filename: The name of the file from which to load the data.

    Returns:
        A dictionary containing all the saved parameters and data including:
        - base_params: dictionary of non-ablation parameters related to simulation and measurement
        - ablation_param: dictionary storing the values of the ablation parameter (T, dt, num_trajectories, or None)
        - A_trues: list of true drift matrices A for each SDE
        - G_trues: list of true diffusion matrices G for each SDE
        - maximal_X_measured: list of the measured data for each SDE under the maximal
    '''
    filepath = os.path.join('Measurement_data', filename)
    with open(filepath, 'rb') as f:
        data = pickle.load(f)
    base_params = data['base']
    max_num_trajectories = data['max_num_trajectories']
    max_T = data['max_T']
    min_dt = data['min_dt']
    print('Base parameters of saved data')
    print(base_params)
    A_trues = data.get('A_trues', [])
    if verbose:
        i = 0
        for A in A_trues:
            print(f'A from SDE {i}: ', A)
            i += 1
    G_trues = data.get('G_trues', [])
    maximal_X_measured_list = data.get('maximal_X_measured', [])
    return A_trues, G_trues, maximal_X_measured_list, max_num_trajectories, max_T, min_dt


def save_experiment_results(filename, variables, results):
    os.makedirs('../MSE_logs', exist_ok=True)
    filepath = os.path.join('../MSE_logs', filename)
    data = {
        'variables': variables,
        'results': results
    }
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=4)


def save_experiment_results_args(filename, base_params, ablation_param, estimation_params, A_trues, G_trues, results):
    os.makedirs('../MSE_logs', exist_ok=True)
    filepath = os.path.join('../MSE_logs', filename)
    base_params.pop('X0')

    data = {
        'base': base_params,
        'ablation': ablation_param,
        'estimation': estimation_params,
        'results': results
    }
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=4)
    print(f'log of experiment results saved in {filepath}')


def load_experiment_results(filename):
    filepath = os.path.join('../MSE_logs', filename)
    with open(filepath, 'r') as f:
        data = json.load(f)
    print(data['base'])
    print(data['ablation'])
    print(data['estimation'])
    print(data['results'])
    return data
