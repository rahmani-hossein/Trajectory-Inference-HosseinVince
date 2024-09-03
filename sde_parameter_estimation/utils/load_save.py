import json
import os
import pickle
import numpy as np
from .simulation import initialize_X0



def find_existing_data(args, max_num_trajectories, max_T, min_dt, specified_D, simulation_mode = 'killed'):
    '''
    Looks for compatible existing measurement data from the "Measurement_data" directory
    Args:
        args:
        max_num_trajectories:
        max_T:
        min_dt:
        simulation_mode:

    Returns:

    '''
    directory = 'Measurement_data'
    if simulation_mode == 'killed':
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
        # Example filename: "seed-0_d-3_n_sdes-10_dt-0.02_N-50_T-1.0_D-1.0"
        num_trajectories = int(parts[6 + offset].split('-')[1])
        T = float(parts[7 + offset].split('-')[1])
        D = float(parts[8 + offset].split('-')[1])
        if d == args.d and n_sdes >= args.n_sdes and num_trajectories >= max_num_trajectories and T >= max_T and dt <= min_dt and D == specified_D:
            print('dt from filename:', dt)
            print('min dt:', min_dt)
            return os.path.join(filename)
    return None

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
    print(f"Data generation complete and saved in {filename}.")

def save_drifts_diffusions(filename, As, Gs):
    os.makedirs('../Saved_drifts_diffusions', exist_ok=True)
    filepath = os.path.join('../Saved_drifts_diffusions', filename)
    data = {
        'A_trues': As,
        'G_trues': Gs,
    }
    with open(filepath, 'wb') as f:
        pickle.dump(data, f)



def save_detailed_experiment_data(filename, data):
    os.makedirs('../../MSE_detailed_logs', exist_ok=True)
    filepath = os.path.join('../../MSE_detailed_logs', filename)
    with open(filepath, 'wb') as f:
        pickle.dump(data, f)


def load_measurement_data(filename, verbose = True):
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
    G_trues = data.get('G_trues', [])
    if verbose:
        i = 0
        for A in A_trues:
            print(f'A from SDE {i}: ', A)
            i += 1
        i = 0
        for G in G_trues:
            print(f'G from SDE {i}: ', G)
            i += 1

    maximal_X_measured_list = data.get('maximal_X_measured', [])
    return A_trues, G_trues, maximal_X_measured_list, max_num_trajectories, max_T, min_dt

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
        'D': float(args.D),
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
        'n_iterations': args.n_iterations
    }
    parameter_estimation_variables = ['n_iterations']
    if args.ablation_variable_name in parameter_estimation_variables:
        estimation_params.pop(args.ablation_variable_name)
    return estimation_params

def save_experiment_results(filename, variables, results):
    os.makedirs('../../MSE_logs', exist_ok=True)
    filepath = os.path.join('../../MSE_logs', filename)
    data = {
        'variables': variables,
        'results': results
    }
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=4)


def save_experiment_results_args(filename, base_params, ablation_param, estimation_params, A_trues, G_trues, results):
    os.makedirs('../../MSE_logs', exist_ok=True)
    filepath = os.path.join('../../MSE_logs', filename)
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
    filepath = os.path.join('../../MSE_logs', filename)
    with open(filepath, 'r') as f:
        data = json.load(f)
    print(data['base'])
    print(data['ablation'])
    print(data['estimation'])
    print(data['results'])
    return data