import os
import json
import numpy as np
import pickle
from tqdm import tqdm
from parameter_estimation import estimate_A_compare_methods


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
    simulation_measurement_variables = ['T', 'dt_EM', 'dt', 'num_trajectories']
    if args.ablation_variable_name in simulation_measurement_variables:
        base_params.pop(args.ablation_variable_name)
    ablation_values = [float(item) for item in args.ablation_values.split(',')]
    ablation_param = {
        args.ablation_variable_name: ablation_values
    }
    return base_params, ablation_param


def initialize_X0(fixed_X0, d):
    if fixed_X0 == 'none':
        return None
    elif fixed_X0 == 'ones':
        return np.ones(d)
    elif fixed_X0 == 'zeros':
        return np.zeros(d)
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


def compute_mse_across_methods(X_measured_list, dt, A_trues, ablation_value, args, measurement_ablation=True):
    '''
    Args:
        X_measured_list: list of measured data indexed by list of SDEs
        dt: time granularity for measurements
        A_trues: list of true drift matrices A indexed by list of SDEs
        args:
    Returns:

    '''
    std_errs = {}
    mean_mse_scores = {}
    temp_mse_scores = {method: [] for method in args.methods}

    # first iterate over each SDE and collect results
    for i in tqdm(range(int(args.n_sdes))):
        if measurement_ablation:
            A_estimations = estimate_A_compare_methods(X_measured_list[i], dt, args.entropy_reg, args.methods,
                                                       n_iterations=args.n_iterations)
        else:
            if args.ablation_variable_name == 'n_iterations':
                A_estimations = estimate_A_compare_methods(X_measured_list[i], dt, args.entropy_reg, args.methods,
                                                           n_iterations=ablation_value)
            elif args.ablation_variable_name == 'entropy_reg':
                A_estimations = estimate_A_compare_methods(X_measured_list[i], dt, methods=args.methods,
                                                           entropy_reg=ablation_value,
                                                           n_iterations=args.n_iterations)
        for method, A_hat in A_estimations.items():
            temp_mse_scores[method].append(np.mean((A_hat - A_trues[i]) ** 2))

    # Compute mean MSEs and standard errors for the current ablation value
    for method in args.methods:
        mean_mse = np.mean(temp_mse_scores[method])
        mean_mse_scores[method] = mean_mse
        std_error = np.std(temp_mse_scores[method]) / np.sqrt(args.n_sdes)
        std_errs[method] = std_error
        print(
            f'Mean MSE ({method}) for {args.ablation_variable_name} = {ablation_value}: {mean_mse}, Standard Error: {std_error}')
    return mean_mse_scores, std_errs


def save_measurement_data(filename, base_params, ablation_param, A_trues, G_trues, maximal_X_measured_list):
    '''
    Args:
        filename:
        base_params: dictionary of non-abalation parameters related to simulation and measurement
        ablation_param: dictionary storing the values of the ablation parameter (T, dt, num_trajectories, or None)
        A_trues: list of true drift matrices A for each SDE
        G_trues: list of true diffusion matrices G for each SDE
        maximal_X_measured_list: list of the measured data for each SDE under the maximal

    Returns:
        saves the measurements and true SDE parameters
    '''
    os.makedirs('Measurement_data', exist_ok=True)
    filepath = os.path.join('Measurement_data', filename)
    data = {
        'base': base_params,
        'ablation': ablation_param,
        'A_trues': A_trues,
        'G_trues': G_trues,
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


def load_measurement_data(filename):
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
    print('Base parameters of saved data')
    print(base_params)
    ablation_param = data['ablation']
    print('Ablated parameter for measurement')
    print(ablation_param)
    A_trues = data.get('A_trues', [])
    G_trues = data.get('G_trues', [])
    maximal_X_measured_list = data.get('maximal_X_measured', [])
    return A_trues, G_trues, maximal_X_measured_list


def save_experiment_results(filename, variables, results):
    os.makedirs('MSE_logs', exist_ok=True)
    filepath = os.path.join('MSE_logs', filename)
    data = {
        'variables': variables,
        'results': results
    }
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=4)


def save_experiment_results_args(filename, base_params, ablation_param, A_trues, G_trues, results):
    os.makedirs('MSE_logs', exist_ok=True)
    filepath = os.path.join('MSE_logs', filename)

    data = {
        'base': base_params,
        'ablation': ablation_param,
        'results': results
    }
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=4)


def load_experiment_results(filename):
    filepath = os.path.join('MSE_logs', filename)
    with open(filepath, 'r') as f:
        data = json.load(f)
    return data['variables'], data['results']
