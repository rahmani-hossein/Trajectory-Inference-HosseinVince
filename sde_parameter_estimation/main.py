import numpy as np
import sys
from simulate_trajectories import *
from parameter_estimation import estimate_A_compare_methods
import utils
import plots
import argparse
import matplotlib.pyplot as plt
import torch
from torch.autograd import grad, Variable
import autograd
import copy
import scipy as sp
from scipy import stats
from sklearn import metrics
import sys
import ot
import gwot
from gwot import models, sim, ts, util
import gwot.bridgesampling as bs
import parameter_estimation, simulate_trajectories, utils


def get_parser():
    parser = argparse.ArgumentParser(description='Run SDE parameter estimation experiments.')
    # parameters for simulation
    parser.add_argument('--measurement_load_file', default=None, help='Load file for SDE measurements')
    parser.add_argument('--master_seed', type=int, default=2, help='Seed for reproducibility')
    parser.add_argument('--d', default=1, type=int, help='Dimension of the process.')
    parser.add_argument('--simulation_mode', default='cell_death', help='How should the data be simulated? Options: cell_death, unkilled')
    parser.add_argument('--dt_em', default=0.001, type=float, help='Simulation time step.')
    parser.add_argument('--n_sdes', default=1, type=int, help='Number of SDEs to simulate per setting.')
    parser.add_argument('--fixed_X0', default='none',
                        help='How is each trajectory for a given SDE initialized? Options: none, zero, ones')
    parser.add_argument('--drift_initialization', default='negative_eigenvalue',
                        help='Method to initialize drift matrix.')
    parser.add_argument('--diffusion_initialization', default='scaled_identity',
                        help='Method to initialize diffusion matrix.')
    parser.add_argument('--diffusion_scale', default=0.1, type=float,
                        help='Scale factor for diffusion matrix initialization.')
    # parameters for measurement
    parser.add_argument('--dt', default=0.01, type=float, help='Observation time step.')
    parser.add_argument('--num_trajectories', type = int,  default=1000,
                        help='Number of trajectories per SDE (observations per time step).')
    parser.add_argument('--T', default=1.0, type=float, help='Total length of observation.')
    # parameters for estimation
    parser.add_argument('--entropy_reg', default=0.01, type=float,
                        help='Entropy regularization parameter for OT solver.')
    parser.add_argument('--n_iterations', default=1, type=int, help='Number of iterations for "iterative" approach.')
    # experiment parameters
    parser.add_argument('--ablation_variable_name', default='dt', help='name of ablation variable')
    parser.add_argument('--ablation_values', default=' 0.1, 0.05, 0.02',
                        help='Comma-separated values for the ablation study.')
    parser.add_argument('--methods', nargs='+', default=['OT reg'], help='List of parameter estimation methods to try')
    parser.add_argument('--save_results', default=False, help='whether or not to save parameter estimation results')
    return parser



def main(args):
    base_params, ablation_param = utils.extract_measurement_parameters(args)
    ablation_values = ablation_param[args.ablation_variable_name]
    # create measurement data along with true SDE parameters (or load them from a pre-existing file)
    if args.measurement_load_file is None:
        print('Generating the data samples')
        result = create_measurement_data(args, base_params, ablation_param)
        if isinstance(result, str):
            measurement_filename = result
            A_trues, G_trues, maximal_X_measured_list, max_num_trajectories, max_T, min_dt = utils.load_measurement_data(measurement_filename)
            print(f'Retrieved previously saved measurements from {measurement_filename}')
        else:
            A_trues, G_trues, maximal_X_measured_list, measurement_filename, max_num_trajectories, max_T, min_dt = result
            print('Finished generating the data samples')
    else:
        measurement_filename = args.measurement_load_file
        A_trues, G_trues, maximal_X_measured_list, max_num_trajectories, max_T, min_dt = utils.load_measurement_data(measurement_filename)
        print(f'Retrieved previously saved measurements from {measurement_filename}')

    experiment_name = f'tp_{args.ablation_variable_name}_from_{measurement_filename}'
    measurement_variables = ['T', 'dt', 'num_trajectories']
    parameter_estimation_variables = ['n_iterations', 'entropy_reg']

    if args.ablation_variable_name in measurement_variables:
        mse_scores_ablation = run_ablation(ablation_param, A_trues, G_trues, maximal_X_measured_list, max_num_trajectories, max_T, min_dt, args,
                                           measurement_ablation=True)
    elif args.ablation_variable_name in parameter_estimation_variables:
        mse_scores_ablation = run_ablation(ablation_param, A_trues, G_trues, maximal_X_measured_list, max_num_trajectories, max_T, min_dt, args,
                                           measurement_ablation=False)
    else:
        raise ValueError(f"Unsupported ablation variable: {args.ablation_variable_name}")

    std_errs = {}
    mean_mse_scores = {}

    for method in args.methods:
        mean_mse_scores[method] = []
        std_errs[method] = []
        for ablation_value in mse_scores_ablation:
            mean_mse = np.mean(mse_scores_ablation[ablation_value][method])
            mean_mse_scores[method].append(mean_mse)
            std_error = np.std(mse_scores_ablation[ablation_value][method]) / np.sqrt(args.n_sdes)
            std_errs[method].append(std_error)

    results = {
        'mse_scores': mean_mse_scores,
        'std_errs': std_errs
    }
    # plot and save results
    results_filename = f"results_{experiment_name}.json"
    ablation_values = ablation_param[args.ablation_variable_name]
    plots.plot_MSE(ablation_values, args.ablation_variable_name,
                   list(mean_mse_scores.values()), list(std_errs.values()), args.methods, args.d, experiment_name)
    if args.save_results:
        utils.save_detailed_experiment_data(results_filename, mse_scores_ablation)
        estimation_params = utils.extract_estimation_parameters(args)
        utils.save_experiment_results_args(results_filename, base_params, ablation_param, estimation_params, A_trues,
                                           G_trues, results)


def run_ablation(ablation_param, A_trues, G_trues, maximal_X_measured_list, max_num_trajectories, max_T, min_dt, args, measurement_ablation=True):
    ablation_values = ablation_param[args.ablation_variable_name]
    if measurement_ablation:
        X_measured_ablation_dict = utils.preprocess_measured_data(maximal_X_measured_list, ablation_values,  max_num_trajectories, max_T, min_dt, args,
                                       measurement_ablation)

    else:
        X_measured_list = utils.preprocess_measured_data(maximal_X_measured_list, ablation_values, max_num_trajectories, max_T, min_dt, args, measurement_ablation)
    # perform parameter estimation for each ablation value
    mse_scores_ablation = {}  # this will be a dictionary of dictionaries, with keys given by ablation value

    for ablation_value in ablation_values:
        if args.ablation_variable_name == 'dt':
            dt = ablation_value
        else:
            dt = float(args.dt)
        if measurement_ablation:
            print('sanity check for shape of measured data:', X_measured_ablation_dict[ablation_value][0].shape)
            mse_scores_ablation[ablation_value] = compute_mse_across_methods(X_measured_ablation_dict[ablation_value], dt, A_trues,
                                                                             ablation_value, args,
                                                                             measurement_ablation)
        else:
            print('sanity check for shape of measured data:', X_measured_list[0].shape)
            mse_scores_ablation[ablation_value] = compute_mse_across_methods(X_measured_list, dt, A_trues,
                                                                         ablation_value, args,
                                                                         measurement_ablation)
        for method in args.methods:
            print(
                f'Mean MSE ({method}) for {args.ablation_variable_name} = {ablation_value}: {np.mean(mse_scores_ablation[ablation_value][method])}, Standard Error: {np.std(mse_scores_ablation[ablation_value][method]) / np.sqrt(args.n_sdes)}')
    return mse_scores_ablation

def compute_mse_across_methods(X_measured_list, dt, A_trues, ablation_value, args, measurement_ablation=True):
    '''
    Args:
        X_measured_list: list of measured data indexed by list of SDEs
        dt: time granularity for measurements
        A_trues: list of true drift matrices A indexed by list of SDEs
        args:
    Returns:

    '''

    mse_scores = {method: [] for method in args.methods}

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
            # if method == 'OT reg':
            #     print('true A:', A_trues[i])
            #     print('estimated A from OT reg:', A_hat)
            mse_scores[method].append(np.mean((A_hat - A_trues[i]) ** 2))
    return mse_scores


if __name__ == "__main__":
    args = get_parser().parse_args()
    main(args)

    #     entropy_reg = float(args.entropy_reg)
    #     ablation_values = [int(item) for item in args.ablation_values.split(',')]
    #     methods = [item for item in args.methods.split(',')]
    #
    #     mse_scores = {method: [] for method in methods}
    #     std_errs = {method: [] for method in methods}
    #
    #     experiment_name = f'{d}D_{ablation_variable_name}_experiment'
    #     results_filename = f"results_{experiment_name}.json"
    #
    # for ablation_value in ablation_values:
    #     # Temporary storage for the current ablation value
    #     temp_mse_scores = {method: [] for method in methods}
    #     for i in tqdm(range(n_sdes)):
    #         A_estimations = estimate_A_compare_methods(X_measured_list[i], dt, entropy_reg, methods,
    #                                                    n_iterations=n_iterations)
    #         for method, A_hat in A_estimations.items():
    #             temp_mse_scores[method].append(np.mean((A_hat - A_trues[i]) ** 2))
    #     # Compute mean MSEs and standard errors for the current ablation value
    #     for method in methods:
    #         mean_mse = np.mean(temp_mse_scores[method])
    #         std_error = np.std(temp_mse_scores[method]) / np.sqrt(n_sdes)
    #         mse_scores[method].append(mean_mse)
    #         std_errs[method].append(std_error)
    #         print(
    #             f'Mean MSE ({method}) for {ablation_variable_name} = {n_iterations}: {mean_mse}, Standard Error: {std_error}')
    # #         step_ratio = int(dt / dt_EM)
    # #         X_measured = X[:, ::step_ratio, :]
    # utils.save_experiment_results(results_filename, {'mse_scores': mse_scores, 'std_errs': std_errs})

#
#
# if __name__ == "__main__":
#
#     # Variables for SDEs
#     d = 3
#     dt_EM = 0.001
#     drift_initialization = 'negative_eigenvalue'
#     diffusion_initialization = 'scaled_identity'
#     diffusion_scale = 0.1
#
#     # Variables for observations
#     num_trajectories = 50
#     T = 1
#     dt = 0.02
#     n_sdes = 1
#     fixed_X0 = np.ones(d)
#     np.random.seed(2)
#
#     # Variables for parameter estimation
#     entropy_reg = 0.01
#     n_iterations = 2
#
#     # ablation parameter
#     ablation_variable_name = 'number of iterations'
#     ablation_values = [1, 2,3,4,5]
#
#     # By default, we will compare expected value over trajectories, OT, OT with regularization
#     methods = ['OT reg']
#     mse_scores = {method: [] for method in methods}
#     std_errs = {method: [] for method in methods}
#
#     experiment_name = f'{d}D_{ablation_variable_name}_experiment'
#     results_filename = f"results_{experiment_name}.json"
#
#     A_trues = []
#     X_measured_list = []
#     print('generating the data')
#     for i in tqdm(range(n_sdes)):
#         A = utils.initialize_drift(d, initialization_type=drift_initialization)
#         A_trues.append(A)
#         print('true A:', A)
#         G = utils.initialize_diffusion(d, initialization_type=diffusion_initialization, diffusion_scale=diffusion_scale)
#         # X = multiple_ou_trajectories(num_trajectories, d, T, dt_EM, A, G, X0=fixed_X0)
#         # # subsample to simulate the measured process
#         # step_ratio = int(dt / dt_EM)
#         # X_measured = X[:, ::step_ratio, :]
#         X_measured = multiple_ou_trajectories_cell_measurement_death(num_trajectories, d, T, dt_EM, dt, A, G,
#                                                                      X0=fixed_X0)
#         print(X_measured.shape)
#         X_measured_list.append(X_measured)
#
#
#     for n_iterations in ablation_values:
#         # Temporary storage for the current ablation value
#         temp_mse_scores = {method: [] for method in methods}
#         for i in tqdm(range(n_sdes)):
#             A_estimations = estimate_A_compare_methods(X_measured_list[i], dt, entropy_reg, methods, n_iterations=n_iterations)
#             for method, A_hat in A_estimations.items():
#                 temp_mse_scores[method].append(np.mean((A_hat - A_trues[i]) ** 2))
#                 # A_hat_traj, A_hat_OT, A_hat_OT_reg = estimate_A_compare_methods(X_measured, dt, entropy_reg=entropy_reg)
#                 #
#                 # temp_mse_scores['Trajectory'].append(np.mean((A_hat_traj - A) ** 2))
#                 # temp_mse_scores['OT'].append(np.mean((A_hat_OT - A) ** 2))
#                 # temp_mse_scores['OT reg'].append(np.mean((A_hat_OT_reg - A) ** 2))
#         # Compute mean MSEs and standard errors for the current ablation value
#         for method in methods:
#             mean_mse = np.mean(temp_mse_scores[method])
#             std_error = np.std(temp_mse_scores[method]) / np.sqrt(n_sdes)
#             mse_scores[method].append(mean_mse)
#             std_errs[method].append(std_error)
#             print(f'Mean MSE ({method}) for {ablation_variable_name} = {n_iterations}: {mean_mse}, Standard Error: {std_error}')
#
#     # Save experimental variables and results
#     variables = {
#         'd': d,
#         'drift_initialization': drift_initialization,
#         'diffusion_initialization': diffusion_initialization,
#         'diffusion_scale': diffusion_scale,
#         'num_trajectories': num_trajectories,
#         'T': T,
#         'dt_EM': dt_EM,
#         'dt': dt,
#         'n_sdes': n_sdes,
#         'fixed_X0': fixed_X0.tolist(),
#         'entropy_reg': entropy_reg,
#         'n_iterations': ablation_values
#     }
#     results = {
#         'mse_scores': mse_scores,
#         'std_errs': std_errs
#     }
#     utils.save_experiment_results(results_filename, variables, results)
#
#     # Plotting the results using the generalized function
#     plots.plot_MSE(ablation_values, ablation_variable_name,
#              list(mse_scores.values()), list(std_errs.values()), methods, d, experiment_name)
#
#


# for T in ablation_values:
#     # By default, we will compare expected value over trajectories, OT, OT with regularization
#     mse_scores_traj, mse_scores_OT, mse_scores_OT_reg = [], [], []
#     std_errs_traj, std_errs_OT, std_errs_OT_reg = [], [], []
#     for i in tqdm(range(n_sdes)):
#         A = initialize_drift(d, initialization_type=drift_initialization)
#         G = initialize_diffusion(d, initialization_type=diffusion_initialization, diffusion_scale=diffusion_scale)
#         # this uses a fixed starting point across the trajectories for each SDE
#         X = multiple_ou_trajectories(num_trajectories, d, T, dt_EM, A, G, X0=fixed_X0)
#         # print('raw:', X.shape)
#         # plot_trajectories(X[0], T, dt)
#         step_ratio = int(dt / dt_EM)
#         X_measured = X[:, ::step_ratio, :]
#         # print('measured:', X_measured.shape)
#         # plot_trajectories(X_measured[0], T, dt)
#         # for i in range(10):
#         #     plot_trajectories(X[i], T, dt)
#         # break
#         A_hat_traj, A_hat_OT, A_hat_OT_reg = estimate_A_compare_methods(X_measured, dt, entropy_reg=entropy_reg)
#         mse_scores_traj.append(np.mean((A_hat_traj - A) ** 2))
#         mse_scores_OT.append(np.mean((A_hat_OT - A) ** 2))
#         mse_scores_OT_reg.append(np.mean((A_hat_OT_reg - A) ** 2))
#     # compute mean MSEs and STDErrs over all SDEs for the current variable of interest
#     mean_mse_traj = np.mean(mse_scores_traj)
#     std_error_traj = np.std(mse_scores_traj) / np.sqrt(n_sdes)
#     mean_mse_OT = np.mean(mse_scores_OT)
#     std_error_OT = np.std(mse_scores_OT) / np.sqrt(n_sdes)
#     mean_mse_OT_reg = np.mean(mse_scores_OT_reg)
#     std_error_OT_reg = np.std(mse_scores_OT_reg) / np.sqrt(n_sdes)
#     # add MSEs and STDerrs to global list for plotting
#     mse_scores_traj_g.append(mean_mse_traj)
#     mse_scores_OT_g.append(mean_mse_OT)
#     mse_scores_OT_reg_g.append(mean_mse_OT_reg)
#     std_errs_traj_g.append(std_error_traj)
#     std_errs_OT_g.append(std_error_OT)
#     std_errs_OT_reg_g.append(std_error_OT_reg)
#     print(f'Mean MSE (Trajectory) for T = {T}: {mean_mse_traj}, Standard Error: {std_error_traj}')
#     print(f'Mean MSE (OT) for T = {T}: {mean_mse_OT}, Standard Error: {std_error_OT}')
#     print(f'Mean MSE (OT reg) for T = {T}: {mean_mse_OT_reg}, Standard Error: {std_error_OT_reg}')
#
# # Save experimental variables and results
# variables = {
#     'd': d,
#     'drift_initialization': drift_initialization,
#     'diffusion_initialization': diffusion_initialization,
#     'diffusion_scale': diffusion_scale,
#     'num_trajectories': num_trajectories,
#     'T': ablation_values,
#     'dts': dt,
#     'n_sdes': n_sdes,
#     'fixed_X0': fixed_X0.tolist(),
#     'entropy_reg': entropy_reg
# }
# results = {
#     'mse_scores_traj_g': mse_scores_traj_g,
#     'mse_scores_OT_g': mse_scores_OT_g,
#     'mse_scores_OT_reg_g': mse_scores_OT_reg_g,
#     'std_errs_traj_g': std_errs_traj_g,
#     'std_errs_OT_g': std_errs_OT_g,
#     'std_errs_OT_reg_g': std_errs_OT_reg_g
# }
# save_experiment_results(results_filename, variables, results)
#
# # Plotting the results
# plt.errorbar(ablation_values, mse_scores_traj_g, yerr=std_errs_traj_g, fmt='-^', label='Trajectory')
# plt.errorbar(ablation_values, mse_scores_OT_g, yerr=std_errs_OT_g, fmt='-o', label='OT')
# plt.errorbar(ablation_values, mse_scores_OT_reg_g, yerr=std_errs_OT_reg_g, fmt='--', label='OT reg')
# plt.xlabel('length of observation (T)')
# plt.ylabel('Mean Squared Error (MSE)')
# plt.title(f'Parameter Estimation on {d}-dimensional Stationary Linear Additive Noise SDE')
# plt.legend()
# plt.grid(True)
#
# # Save the plot
# plot_filename = f"mse_plot_{experiment_name}.png"
# plt.savefig(plot_filename)
# plt.show()

#
#
# if __name__ == "__main__":
#     # Set parameters for trajectories
#     num_trajectories = 1
#     T = 1
#     dt = 0.02
#     dim = 10
#     # m = 2
#     mse_scores = []
#     mse_scores_alt = []
#     n_sdes = 100
#     for i in range(n_sdes):
#         A = generate_random_matrix(dim)#np.array([[1.76, -0.1], [0.98, 0]])#generate_negative_eigenvalue_matrix(dim) #
#         G = 0.5*generate_random_matrix(dim)#np.eye(dim) #[np.array([[-0.11, -0.14],[-0.29,-0.22]]), np.array([[-0.17, 0.59],[0.81,0.18]]) ]
#         #[0.9* np.eye(dim) for _ in range(dim)]   #np.array([[-0.11, -0.14], [-0.29, -0.22]]) #0.1*np.eye(dim)    # Variance matrix
#         # GGT = np.matmul(G, np.transpose(G))
#         X0 = np.random.randn(dim) #np.array([1.87, -0.98]) #np.random.randn(dim) #(np.zeros(dim))  #        # Initial values
#         X = multiple_ou_trajectories(num_trajectories, T, dt, A, G, X0) #multiple_multiplicative_noise_trajectories(num_trajectories, T, dt, A, G, X0) #
#         # plot_trajectories(X[0], T, dt) # plotting one trajectory
#         A_hat = estimate_A_exp(trajectories=X, dt = dt)
#         mse = 1/dim**2*np.linalg.norm((A_hat - A)**2)
#         mse_scores.append(mse)
#         A_hat_ = estimate_A_exp_alt(trajectories=X, dt=dt)
#         mse_ = 1/dim**2*np.linalg.norm((A_hat_ - A)**2)
#         mse_scores_alt.append(mse_)
#     print(f'(original) Mean MSE across {n_sdes} randomly generated SDEs of dimension {dim}: ', np.mean(mse_scores))
#     print(f'(new) Mean MSE across {n_sdes} randomly generated SDEs of dimension {dim}: ', np.mean(mse_scores_alt))

#
# # print('true GGT:', GGT)
# # print('estimated GGT:', GGT_hat)
# print('true A:', A)
# print('estimated A:', A_hat)
# # print('entry-wise differences:', abs(A_hat-A))
# print('MSE for A:', 1/dim**2*np.linalg.norm((A_hat - A)**2))
# A_hat_1 = estimate_A(X, dt)
# print('estimated A (classic):', A_hat_1)
# print('MSE for A:', 1/dim**2*np.linalg.norm((A_hat_1 - A) ** 2))


#
#
# A_hat = estimate_A_1D(X, dt)
# print('estimated A:', A_hat)
# print('MSE:', (A_hat - A)**2)
# A_hat_md = estimate_A_(X, dt, G)
# print('estimated A (with true G) mashed:', A_hat_md)
# print('MSE:', (A_hat_md - A) ** 2)
# A_hat_md = estimate_A(X, dt, np.array([[1,0,0], [0,1,0],[0,0,1]]) )
# print('estimated A (matrix code with Id):', A_hat_md)
# print('MSE:', (A_hat_md - A)**2)
# X = ou_process(T, dt, A, G, X0)
# plot_trajectories(X, T, dt)
# plot_covariance_functions(X, T, dt, A, G)


#
# # Experiment setup
# if __name__ == "__main__":
#     # Constants and configurations
#     d = 10
#     drift_initialization = 'negative_eigenvalue'
#     diffusion_initialization = 'scaled_identity'
#     diffusion_scale = 0.1
#     entropy_reg = 0.01
#     dt = 0.02
#     n_sdes = 1
#     fixed_X0 = np.ones(d)
#     experiment_name = 'garbage' #'1_run_cell_transtions_tau-0.01-pd-0.5_budget_fixed_X0-1_d-10_1SDE'
#     results_filename = f"results_{experiment_name}.json"
#
#
#     configurations = [
#         (4, 10)
#     ]
#
#     buffer = 100
#     budget = configurations[0][0]*configurations[0][1]
#     print('budget:', budget)
#     init_pop = None # int(buffer * budget)
#     tau = None
#     pd = None
#
#     # Lists to store results
#     mse_scores_traj_g, mse_scores_OT_g, mse_scores_OT_reg_g = [], [], []
#     std_errs_traj_g, std_errs_OT_g, std_errs_OT_reg_g = [], [], []
#
#     # Run experiments
#     for T, num_trajectories in configurations:
#         mse_scores_traj, mse_scores_OT, mse_scores_OT_reg = [], [], []
#         for _ in tqdm(range(n_sdes)):
#             A = initialize_drift(d, initialization_type=drift_initialization)
#             G = initialize_diffusion(d, initialization_type=diffusion_initialization, diffusion_scale=diffusion_scale)
#             if init_pop is not None:
#                 if tau is None:
#                     X = multiple_ou_trajectories_cell(num_trajectories, d, T, dt, A, G, init_pop, X0=fixed_X0)
#                 else:
#                     print('with branching')
#                     X = multiple_ou_trajectories_cell_branching(num_trajectories, d, T, dt, A, G, init_pop, X0=fixed_X0, tau=tau, pd = pd)
#             else:
#                 X = multiple_ou_trajectories(num_trajectories, d, T, dt, A, G, X0=fixed_X0)
#             # for i in range(10):
#             #     plot_trajectories(X[i], T, dt)
#             #     break
#             A_hat_traj, A_hat_OT, A_hat_OT_reg = estimate_A_compare_methods(X, dt, entropy_reg=entropy_reg)
#             mse_scores_traj.append(np.mean((A_hat_traj - A) ** 2))
#             mse_scores_OT.append(np.mean((A_hat_OT - A) ** 2))
#             mse_scores_OT_reg.append(np.mean((A_hat_OT_reg - A) ** 2))
#
#         # compute mean MSEs and STDErrs over all SDEs for the current variable of interest
#         mean_mse_traj = np.mean(mse_scores_traj)
#         std_error_traj = np.std(mse_scores_traj) / np.sqrt(n_sdes)
#         mean_mse_OT = np.mean(mse_scores_OT)
#         std_error_OT = np.std(mse_scores_OT) / np.sqrt(n_sdes)
#         mean_mse_OT_reg = np.mean(mse_scores_OT_reg)
#         std_error_OT_reg = np.std(mse_scores_OT_reg) / np.sqrt(n_sdes)
#         # add MSEs and STDerrs to global list for plotting
#         mse_scores_traj_g.append(mean_mse_traj)
#         mse_scores_OT_g.append(mean_mse_OT)
#         mse_scores_OT_reg_g.append(mean_mse_OT_reg)
#         std_errs_traj_g.append(std_error_traj)
#         std_errs_OT_g.append(std_error_OT)
#         std_errs_OT_reg_g.append(std_error_OT_reg)
#         print(f'Mean MSE (Trajectory) for T = {T}, num_traj = {num_trajectories}: {mean_mse_traj}, Standard Error: {std_error_traj}')
#         print(f'Mean MSE (OT) for T = {T}, num_traj = {num_trajectories}: {mean_mse_OT}, Standard Error: {std_error_OT}')
#         print(f'Mean MSE (OT reg) for T = {T}, num_traj = {num_trajectories}: {mean_mse_OT_reg}, Standard Error: {std_error_OT_reg}')
#
#
#     # Save results and plot
#     results = {
#         'mse_scores_traj_g': mse_scores_traj_g, 'mse_scores_OT_g': mse_scores_OT_g, 'mse_scores_OT_reg_g': mse_scores_OT_reg_g,
#         'std_errs_traj_g': std_errs_traj_g, 'std_errs_OT_g': std_errs_OT_g, 'std_errs_OT_reg_g': std_errs_OT_reg_g
#     }
#     save_experiment_results(results_filename, {
#         'configurations': configurations, 'other_variables': { 'init_pop': init_pop, 'tau': tau, 'pd': pd,
#             'd': d, 'dt': dt, 'n_sdes': n_sdes, 'fixed_X0': list(fixed_X0), 'entropy_reg': entropy_reg
#         }
#     }, results)
#
#     # Plotting
#     ratios = [f"{T}:{n}" for T, n in configurations]
#     plt.figure(figsize=(12, 8))
#     plt.errorbar(ratios, mse_scores_traj_g, yerr=std_errs_traj_g, fmt='-^', label='Trajectory', markersize=8)
#     plt.errorbar(ratios, mse_scores_OT_g, yerr=std_errs_OT_g, fmt='-o', label='OT', markersize=8)
#     plt.errorbar(ratios, mse_scores_OT_reg_g, yerr=std_errs_OT_reg_g, fmt='--', label='OT reg', markersize=8)
#     plt.xlabel('Ratio T: num_trajectories')
#     plt.xticks(rotation=45)
#     plt.ylabel('Mean Squared Error (MSE)')
#     plt.title(f'Parameter Estimation on {d}-dimensional Stationary Linear Additive Noise SDE')
#     plt.legend()
#     plt.grid(True)
#     plt.tight_layout()
#     # Save the plot
#     plot_filename = f"mse_plot_{experiment_name}.png"
#     plt.savefig(plot_filename)
#     plt.show()
