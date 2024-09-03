from simulate_trajectories import *
from parameter_estimation import estimate_params_compare_methods
import plots
import argparse
import datetime
import utils

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def get_parser():
    parser = argparse.ArgumentParser(description='Run SDE parameter estimation experiments.')
    # parameters for simulation
    parser.add_argument('--measurement_load_file', default=None, help='Load file for SDE measurements')
    parser.add_argument('--saved_drifts_diffusions_file', default=None, help='Load file for presaved A and G matrices')
    parser.add_argument('--master_seed', type=int, default=2, help='Seed for reproducibility')
    parser.add_argument('--d', default=1, type=int, help='Dimension of the process.')
    parser.add_argument('--simulation_mode', default='killed', help='How should the data be simulated? Options: cell_death, unkilled')
    parser.add_argument('--dt_em', default=0.001, type=float, help='Simulation time step.')
    parser.add_argument('--n_sdes', default=1, type=int, help='Number of SDEs to simulate per setting.')
    parser.add_argument('--fixed_X0', default='none',
                        help='How is each trajectory for a given SDE initialized? Options: none, zero, ones')
    parser.add_argument('--drift_initialization', default='negative_eigenvalue',
                        help='Method to initialize drift matrix.')
    parser.add_argument('--diffusion_initialization', default='scaled_identity',
                        help='Method to initialize diffusion matrix.')
    parser.add_argument('--D', default=1, type=float,
                        help='Scale factor for diffusion matrix initialization.')
    # parameters for measurement
    parser.add_argument('--dt', default=0.01, type=float, help='Observation time step.')
    parser.add_argument('--num_trajectories', type = int,  default=1000,
                        help='Number of trajectories per SDE (observations per time step).')
    parser.add_argument('--T', default=1.0, type=float, help='Total length of observation.')
    # parameters for estimation
    parser.add_argument('--entropy_reg', default=0.01, type=float,
                        help='Entropy regularization parameter for OT solver.')
    parser.add_argument('--true_A', default=False, type=str2bool,
                        help='Whether or not to give estimator true A')
    parser.add_argument('--true_G', default=True, type=str2bool,
                        help='Whether or not to give estimator true diffusivity GGT')
    parser.add_argument('--frac_other_time_samples', default=0, type=float,
                        help='Fraction of probability reserved for samples observed outside the given time')
    parser.add_argument('--n_iterations', default=1, type=int, help='Number of iterations for "iterative" approach.')
    # experiment parameters
    parser.add_argument('--ablation_variable_name', default='dt', help='name of ablation variable')
    parser.add_argument('--ablation_values', default=' 0.1, 0.05, 0.02',
                        help='Comma-separated values for the ablation study.')
    parser.add_argument('--methods', nargs='+', default=['OT reg'], help='List of parameter estimation methods to try')
    parser.add_argument('--display_plot', default=True, type=str2bool, help='whether or not to display the MSE plots')
    parser.add_argument('--save_results', default=False, type=str2bool, help='whether or not to save parameter estimation results')
    return parser


def main(args):
    base_params, ablation_param = utils.load_save.extract_measurement_parameters(args)
    # create measurement data along with true SDE parameters (or load them from a pre-existing file)
    if args.measurement_load_file is None:
        result = create_measurement_data(args, base_params, ablation_param)
        if isinstance(result, str):
            # if we have detected a pre-existing file with the specified parameters, we load data from that file
            measurement_filename = result
            A_trues, G_trues, maximal_X_measured_list, max_num_trajectories, max_T, min_dt = utils.load_save.load_measurement_data(measurement_filename)
            print(f'Retrieved previously saved measurements from {measurement_filename}')
        else:
            A_trues, G_trues, maximal_X_measured_list, measurement_filename, max_num_trajectories, max_T, min_dt = result
            print('Finished generating the data samples')
    else:
        measurement_filename = args.measurement_load_file
        A_trues, G_trues, maximal_X_measured_list, max_num_trajectories, max_T, min_dt = utils.load_save.load_measurement_data(measurement_filename)
        print(f'Retrieved previously saved measurements from {measurement_filename}')
    # Get the current date
    current_date = datetime.datetime.now().strftime('%Y-%m-%d')
    experiment_name = f'{current_date}_{args.ablation_variable_name}_from_{measurement_filename}'
    measurement_variables = ['T', 'dt', 'num_trajectories']
    parameter_estimation_variables = ['n_iterations', 'frac_other_time_samples']

    if args.ablation_variable_name in measurement_variables:
        mse_scores_ablation_A, mse_scores_ablation_G = run_ablation(ablation_param, A_trues, G_trues, maximal_X_measured_list, max_num_trajectories, max_T, min_dt, args,
                                           measurement_ablation=True)
    elif args.ablation_variable_name in parameter_estimation_variables:
        mse_scores_ablation_A, mse_scores_ablation_G = run_ablation(ablation_param, A_trues, G_trues, maximal_X_measured_list, max_num_trajectories, max_T, min_dt, args,
                                           measurement_ablation=False)
    else:
        raise ValueError(f"Unsupported ablation variable: {args.ablation_variable_name}")

    std_errs_A = {}
    mean_mse_scores_A = {}
    std_errs_G = {}
    mean_mse_scores_G = {}

    for method in args.methods:
        mean_mse_scores_A[method] = []
        std_errs_A[method] = []
        mean_mse_scores_G[method] = []
        std_errs_G[method] = []
        for ablation_value in mse_scores_ablation_A:
            mean_mse_A = np.mean(mse_scores_ablation_A[ablation_value][method])
            mean_mse_scores_A[method].append(mean_mse_A)
            std_error_A = np.std(mse_scores_ablation_A[ablation_value][method]) / np.sqrt(args.n_sdes)
            std_errs_A[method].append(std_error_A)
            mean_mse_G = np.mean(mse_scores_ablation_G[ablation_value][method])
            mean_mse_scores_G[method].append(mean_mse_G)
            std_error_G = np.std(mse_scores_ablation_G[ablation_value][method]) / np.sqrt(args.n_sdes)
            std_errs_G[method].append(std_error_G)
    results = {
        'mse_scores_A': mean_mse_scores_A,
        'std_errs_A': std_errs_A,
        'mse_scores_G': mean_mse_scores_G,
        'std_errs_G': std_errs_G
    }
    # plot and save results
    results_filename = f"results_{experiment_name}.json"
    ablation_values = ablation_param[args.ablation_variable_name]
    if args.ablation_variable_name == 'n_iterations':
        ablation_values = [i for i in range(int(max(ablation_values)))]
    plots.plot_MSE(ablation_values, args.ablation_variable_name,
                   list(mean_mse_scores_A.values()), list(std_errs_A.values()), args.methods, args.d, experiment_name, parameter_name='A', display_plot=args.display_plot, save_plot=args.save_results)
    plots.plot_MSE(ablation_values, args.ablation_variable_name,
                   list(mean_mse_scores_G.values()), list(std_errs_G.values()), args.methods, args.d, experiment_name, parameter_name='G', display_plot=args.display_plot, save_plot=args.save_results)
    if args.save_results:
        utils.load_save.save_detailed_experiment_data(results_filename, mse_scores_ablation_A)
        estimation_params = utils.load_save.extract_estimation_parameters(args)
        utils.load_save.save_experiment_results_args(results_filename, base_params, ablation_param, estimation_params, A_trues,
                                           G_trues, results)


def run_ablation(ablation_param, A_trues, G_trues, maximal_X_measured_list, max_num_trajectories, max_T, min_dt, args, measurement_ablation=True):
    ablation_values = ablation_param[args.ablation_variable_name]
    D_trues = [np.matmul(G_true,G_true.T) for G_true in G_trues]
    # data preprocessing before running experiment
    if measurement_ablation:
        # if the ablated variable pertains to measurement (dt, T, num_trajectories), then each ablated variable (key) corresponds to a different set of measurements (value)
        X_measured_ablation_dict = utils.load_save.preprocess_measured_data(maximal_X_measured_list, ablation_values,  max_num_trajectories, max_T, min_dt, args,
                                       measurement_ablation)

    else:
        # if the ablated variable does not pertain to measurement, then we may use the same measurements (one per SDE) across experiments
        X_measured_list = utils.load_save.preprocess_measured_data(maximal_X_measured_list, ablation_values, max_num_trajectories, max_T, min_dt, args, measurement_ablation)
    # perform parameter estimation for each ablation value
    mse_scores_ablation_A = {}  # this will be a dictionary of dictionaries, with keys given by ablation value
    mse_scores_ablation_G = {}
    if args.ablation_variable_name == 'n_iterations':
        # this case is treated separately to avoid unneeded computations
        max_iterations = int(max(ablation_values))
        n_iteration = 1
        # first estimates
        A_estimations_list, G_estimations_list = estimate_A_G(X_measured_list, float(args.dt), float(args.T), A_trues,
                                                              D_trues, 1, args, False, iterative=False, true_Ds = D_trues)
        # use first estimates as priors for next estimates
        prev_A_estimates_list = [A_estimations['OT reg'] for A_estimations in A_estimations_list]
        prev_G_estimates_list = [G_estimations['OT reg'] for G_estimations in G_estimations_list] # these are for GGT
        mse_scores_A, mse_scores_G = compute_mse_across_methods(A_trues, D_trues, A_estimations_list,
                                                                G_estimations_list,
                                                                int(args.n_sdes))
        mse_scores_ablation_A[n_iteration] = mse_scores_A
        mse_scores_ablation_G[n_iteration] = mse_scores_G
        for method in args.methods:
            print(
                f'A estimation mean MSE ({method}) for {args.ablation_variable_name} = {n_iteration}: {np.mean(mse_scores_ablation_A[n_iteration][method])}, Standard Error: {np.std(mse_scores_ablation_A[n_iteration][method]) / np.sqrt(args.n_sdes)}')
            print(
                f'GGT estimation mean MSE ({method}) for {args.ablation_variable_name} = {n_iteration}: {np.mean(mse_scores_ablation_G[n_iteration][method])}, Standard Error: {np.std(mse_scores_ablation_G[n_iteration][method]) / np.sqrt(args.n_sdes)}')

        while n_iteration < max_iterations:
            # use previous estimates as priors for current estimate
            A_estimations_list, G_estimations_list = estimate_A_G(X_measured_list, float(args.dt), float(args.T),
                                                                  prev_A_estimates_list,
                                                                  prev_G_estimates_list, 1, args, False, iterative = True, true_Ds=D_trues)
            n_iteration += 1
            mse_scores_A, mse_scores_G = compute_mse_across_methods(A_trues, D_trues, A_estimations_list,
                                                                    G_estimations_list,
                                                                    int(args.n_sdes))
            mse_scores_ablation_A[n_iteration] = mse_scores_A
            mse_scores_ablation_G[n_iteration] = mse_scores_G
            for method in args.methods:
                print(
                    f'A estimation mean MSE ({method}) for {args.ablation_variable_name} = {n_iteration}: {np.mean(mse_scores_ablation_A[n_iteration][method])}, Standard Error: {np.std(mse_scores_ablation_A[n_iteration][method]) / np.sqrt(args.n_sdes)}')
                print(
                    f'GGT estimation mean MSE ({method}) for {args.ablation_variable_name} = {n_iteration}: {np.mean(mse_scores_ablation_G[n_iteration][method])}, Standard Error: {np.std(mse_scores_ablation_G[n_iteration][method]) / np.sqrt(args.n_sdes)}')
            # update priors for next iteration
            prev_A_estimates_list = [A_estimations['OT reg'] for A_estimations in A_estimations_list]
            prev_G_estimates_list = [G_estimations['OT reg'] for G_estimations in
                                     G_estimations_list]  # these are for GGT
    else:
        for ablation_value in ablation_values:
            if args.ablation_variable_name == 'dt':
                dt = ablation_value
            else:
                dt = float(args.dt)
            if args.ablation_variable_name == 'T':
                T = ablation_value
            else:
                T = float(args.dt)

            if measurement_ablation:
                print('sanity check for shape of measured data:', X_measured_ablation_dict[ablation_value][0].shape)

                A_estimations_list, G_estimations_list = estimate_A_G(X_measured_ablation_dict[ablation_value], dt, T, A_trues, D_trues, ablation_value, args, measurement_ablation)
            else:
                print('sanity check for shape of measured data:', X_measured_list[0].shape)
                A_estimations_list, G_estimations_list = estimate_A_G(X_measured_list, dt, T, A_trues,
                                                            D_trues, ablation_value, args, measurement_ablation)
            mse_scores_A, mse_scores_G = compute_mse_across_methods(A_trues, D_trues, A_estimations_list, G_estimations_list,
                                                                    int(args.n_sdes))
            mse_scores_ablation_A[ablation_value] = mse_scores_A
            mse_scores_ablation_G[ablation_value] = mse_scores_G
            for method in args.methods:
                print(
                    f'A estimation mean MSE ({method}) for {args.ablation_variable_name} = {ablation_value}: {np.mean(mse_scores_ablation_A[ablation_value][method])}, Standard Error: {np.std(mse_scores_ablation_A[ablation_value][method]) / np.sqrt(args.n_sdes)}')
                print(
                    f'GGT estimation mean MSE ({method}) for {args.ablation_variable_name} = {ablation_value}: {np.mean(mse_scores_ablation_G[ablation_value][method])}, Standard Error: {np.std(mse_scores_ablation_G[ablation_value][method]) / np.sqrt(args.n_sdes)}')
    return mse_scores_ablation_A, mse_scores_ablation_G


def estimate_A_G(X_measured_list, dt, T, A_priors, D_priors, ablation_value, args, measurement_ablation=True, iterative=False, true_Ds = None):
    A_estimations_list, G_estimations_list = [], []
    for i in tqdm(range(int(args.n_sdes))):
        if iterative:
            prior_A = A_priors[i]
            prior_D = D_priors[i]
            print('using current estimated A:', prior_A)
            print('using current estimated D:', prior_D)
        else:
            prior_A = None
            prior_D = None

        if measurement_ablation:
            A_estimations, G_estimations = estimate_params_compare_methods(X_measured_list[i], dt, T,
                                                                           args.methods,
                                                                           n_iterations=args.n_iterations,
                                                                           frac_other_time_samples=args.frac_other_time_samples,
                                                                           A=prior_A)
        else:
            if args.ablation_variable_name == 'n_iterations':
                A_estimations, G_estimations = estimate_params_compare_methods(X_measured_list[i], dt, T,
                                                                               args.methods,
                                                                               n_iterations=1,
                                                                               frac_other_time_samples=args.frac_other_time_samples,
                                                                               A=prior_A, GGT = prior_D)
            elif args.ablation_variable_name == 'frac_other_time_samples':
                A_estimations, G_estimations = estimate_params_compare_methods(X_measured_list[i], dt, T,
                                                                               methods=args.methods,
                                                                               n_iterations=args.n_iterations,
                                                                               frac_other_time_samples=ablation_value, true_A=prior_A)
            else:
                print('Error: unsupported ablation variable')
                return
        A_estimations_list.append(A_estimations)
        G_estimations_list.append(G_estimations)
    return A_estimations_list, G_estimations_list

def compute_mse_across_methods(A_trues, D_trues, A_estimations_list, G_estimations_list, n_sdes, verbose=True):
    '''
    Args:
        A_trues: list of true drift matrices A
        D_trues: list of true diffusion matrices GGT
        A_estimations_list: List of dictionaries of estimated A matrices keyed by method name.
        G_estimations_list: List of dictionaries of estimated G matrices keyed by method name.
    Returns:

    '''

    mse_scores_A = {method: [] for method in args.methods}
    mse_scores_G = {method: [] for method in args.methods}

    for i in range(n_sdes):
        A_estimations = A_estimations_list[i]
        G_estimations = G_estimations_list[i]
        for method, A_hat in A_estimations.items():
            if verbose:
                print('true A:', A_trues[i])
                print(f'estimated A from {method}:', A_hat)
            mse_scores_A[method].append(np.mean((A_hat - A_trues[i]) ** 2))
        for method, G_hat in G_estimations.items():
            if verbose:
                print('true GGT:', D_trues[i])
                print(f'estimated GGT from {method}:', G_hat)
            mse_scores_G[method].append(np.mean((G_hat - D_trues[i])** 2))
        # plot_true_vs_estimated(A_trues, A_estimations) need to fix this plotting
    return mse_scores_A, mse_scores_G


if __name__ == "__main__":
    args = get_parser().parse_args()
    main(args)

