import matplotlib as plt
from parameter_estimation import *
import utils
from plots import plot_trajectories
import pickle
from simulate_trajectories import *

if __name__ == "__main__":
    # Example usage:
    N = 1000
    n_sdes = 50
    d=5
    T=1
    dt=0.02
    num_steps_truncate = 50
    N_truncate = 20
    N_plot = 5
    A_biases, G_biases = [], []


    filename = f"unkilled_seed-2_X0-none_d-{d}_n_sdes-{n_sdes}_dt-0.02_N-1000_T-1.0"#f"unkilled_seed-0_d-{d}_n_sdes-10_dt-0.02_N-{N}_T-1.0"
    A_trues, G_trues, maximal_X_measured_list, max_num_trajectories, max_T, min_dt = utils.load_measurement_data(filename)
    for idx in range(n_sdes):
    # idx = 2
        X = maximal_X_measured_list[idx][:N_truncate, :num_steps_truncate, :]
        # for j in range(10):
        #     plot_trajectories(X[j], T, dt)
        A = A_trues[idx]
        G = G_trues[idx]
        X0 =X[0, 0, :]
        print('true A:', A_trues[idx])
        print('X0:', X0)
        for i in range(N_plot):
            plot_trajectories(X[i], T, dt)
            # X = ou_process(T, dt, A, G, X0)
            # plot_trajectories(X, T, dt)




        print('true GGT:', G_trues[idx]@np.transpose(G_trues[idx]))
        # plot_trajectories(X[0], T, dt)
        dt = 0.02
        shuffle = True
        reg = 0.01
        shuffled_X = np.zeros((N_truncate, num_steps_truncate, d))
        # Fill shuffled_X with shuffled marginal samples
        shuffled_samples = extract_marginal_samples(X, shuffle=True)
        for i in range(num_steps_truncate):
            shuffled_X[:, i, :] = shuffled_samples[i]
        # for j in range(N_plot):
        #     plot_fuck(X, shuffled_X, trajectory_index=j)
        #
        # print(X)
        # print(shuffled_X)

        # X_OT = estimate_next_step_OT(X, dt, entropy_reg=0, shuffle = shuffle)
        # X_OT_reg = estimate_next_step_OT(X, dt, entropy_reg=reg, shuffle= shuffle)
        raw_avg = True
        A_OT_reg, X_OT_reg = estimate_A_exp_ot(shuffled_samples, dt, entropy_reg=reg, return_OT_traj = True, use_raw_avg=raw_avg)
        GG_T_OT_reg = estimate_GGT(X_OT_reg, T)
        print(f'OT reg (reg={reg}) estimated A:', A_OT_reg)
        print(f'A diff:', (A_OT_reg - A_trues[idx]))
        print(f'MSE OT reg:', np.mean((A_OT_reg - A_trues[idx]) ** 2))
        print(f'OT reg (reg={reg}) estimated GGT:', GG_T_OT_reg)
        print(f'G diff:', (GG_T_OT_reg - G_trues[idx] @G_trues[idx].T  ))
        A_OT, X_OT = estimate_A_exp_ot(shuffled_samples, dt, entropy_reg=0, return_OT_traj = True, use_raw_avg=raw_avg)#estimate_A_exp(X_OT, dt)
        GG_T_OT = estimate_GGT(X_OT, T)
        print(f'OT estimated A:', A_OT)
        A_bias = (A_OT - A_trues[idx])[0][0]
        A_biases.append(A_bias)
        print(f'A diff:', A_bias)
        print(f'MSE OT:', np.mean((A_OT - A_trues[idx]) ** 2))
        print(f'OT estimated GGT:', GG_T_OT)
        G_bias = (GG_T_OT - G_trues[idx] @G_trues[idx].T)[0][0]
        G_biases.append(G_bias)
        print(f'G diff:', G_bias)
        for j in range(N_plot):
            plot_comparison(X, X_OT, X_OT_reg, trajectory_index=j)
    print('A biases:', A_biases)
    print('G biases:', G_biases)





    # filename = "seed-0_d-2_n_sdes-10_dt-0.02_N-1000_T-1.0"
    # loaded_data = utils.load_measurement_data(filename)




def master_sde_simulation_and_estimation(noise_type, T, dt, num_trajectories, num_sdes, d, m,
                                         A_initialization, G_initialization,
                                         fixed_starting_point=True, starting_point=None,
                                         A_params=None, G_params=None,
                                         sparsity=None, scale_identity=1,
                                         parameter_estimation_solver="classic",
                                         OT_regularization=None, ablation_variable=None,
                                         ablation_range=None):
    if A_params is None:
        A_params = {'low': -1, 'high': 1}
    if G_params is None:
        G_params = {'low': -1, 'high': 1}


    if starting_point is None:
        starting_point = np.random.randn(num_trajectories, d) if fixed_starting_point else [np.random.randn(d) for _ in range(num_trajectories)]

    true_As, estimated_As = [], []
    for _ in range(num_sdes):
        true_A = generate_or_configure_matrix(d, cols=d, initialization_type=A_initialization, params = A_params, sparsity=sparsity)
        true_As.append(true_A)
        A = np.copy(true_A)
        G = generate_or_configure_matrix(d, m, initialization_type=G_initialization, params=G_params, sparsity = sparsity, scale_identity=scale_identity)

        trajectories = simulate_sde_trajectories(noise_type, T, dt, num_trajectories, d, m, A, G, starting_point)
        estimated_A = perform_parameter_estimation(trajectories, dt, parameter_estimation_solver, OT_regularization)
        estimated_As.append(estimated_A)

    return true_As, estimated_As



def generate_or_configure_matrix(rows, cols=None, initialization_type="random", params=None, sparsity=None, scale_identity=1):
    """
    Generate or configure a matrix based on specified parameters.

    Parameters:
        rows (int): Number of rows in the matrix.
        cols (int): Number of columns in the matrix, defaults to rows if None.
        initialization_type (str): Type of matrix to generate ('random', 'negative_eigval', 'identity').
        params (dict): Parameters for the matrix generation such as 'low' and 'high' for uniform distribution.
        sparsity (float): Fraction of elements to set as zero (sparsity), optional.
        scale_identity (float): Scaling factor for the identity matrix if chosen.

    Returns:
        numpy.ndarray: Generated or configured matrix.
    """
    if cols is None:
        cols = rows  # Default to a square matrix if cols not specified

    if initialization_type == "random":
        if params is None:
            params = {'low': -1, 'high': 1}
        matrix = np.random.uniform(params['low'], params['high'], (rows, cols))
    elif initialization_type == "negative_eigval":
        if rows != cols:
            raise ValueError("Negative eigenvalue matrices must be square.")
        matrix = generate_negative_eigenvalue_matrix(rows)
    elif initialization_type == "identity":
        if rows != cols:
            raise ValueError("Identity matrices must be square.")
        matrix = np.eye(rows) * scale_identity

    if sparsity is not None:
        mask = np.random.rand(rows, cols) < sparsity
        matrix = matrix * mask

    return matrix


def simulate_sde_trajectories(noise_type, T, dt, num_trajectories, d, m, A, G, starting_points):
    """
    Simulate trajectories from a d-dimensional SDE with specified parameters.

    Parameters:
        noise_type (str): 'additive' or 'multiplicative' noise type.
        T (float): Total simulation time.
        dt (float): Time step size.
        num_trajectories (int): Number of trajectories to simulate.
        d (int): Dimension of the SDE process.
        m (int): Dimension of the Brownian motion.
        A (numpy.ndarray): Drift matrix (d x d).
        G (numpy.ndarray): Diffusion matrix (d x m).
        starting_points (numpy.ndarray): Initial points for each trajectory (num_trajectories x d).

    Returns:
        numpy.ndarray: Simulated trajectories (num_trajectories x num_time_steps x d).
    """
    num_time_steps = int(T / dt)
    trajectories = np.zeros((num_trajectories, num_time_steps, d))

    for i in range(num_trajectories):
        x = starting_points[i]
        trajectories[i, 0] = x

        for t in range(1, num_time_steps):
            dt_sqrt = np.sqrt(dt)
            dW = np.random.normal(0, dt_sqrt, (m,))  # Brownian increments

            if noise_type == "additive":
                dx = A @ x * dt + G @ dW
            elif noise_type == "multiplicative":
                # Ensure that the multiplication of G and x is compatible with dW
                Gx = G @ x if G.shape[1] == d else np.dot(G, x.reshape(-1, 1)).flatten()
                dx = A @ x * dt + Gx * dW
            else:
                raise ValueError("Invalid noise type specified. Choose 'additive' or 'multiplicative'.")

            x += dx
            trajectories[i, t] = x

    return trajectories


def perform_parameter_estimation(trajectories, dt, solver_type, OT_regularization=None):
    if solver_type == "classic":
        estimated_A = estimate_A(trajectories, dt)  # Function from parameter_estimation.py
    elif solver_type == "trajectory_expectation":
        estimated_A = estimate_A_exp(trajectories, dt)
    elif solver_type == "OT_expectation":
        # Assume marginals are passed or can be derived from trajectories
        marginals = extract_marginals(trajectories)
        estimated_A = estimate_A_exp_ot(marginals, dt, entropy_reg=OT_regularization)
    else:
        raise ValueError("Unsupported parameter estimation solver type specified.")

    return estimated_A


def perform_ablation_study(variable, range_values, noise_type, T, dt, num_trajectories, num_sdes,
                           d, m, A_initialization, G_initialization, fixed_starting_point, starting_point,
                           A_params, G_params, sparsity, scale_identity, ablation_methods=None):
    if ablation_methods is None:
        ablation_methods = ["trajectory_expectation", "OT_expectation", "OT_expectation_reg"]
    results = {method: [] for method in ablation_methods}

    for value in range_values:
        # Adjust variable based on the ablation variable
        if variable == 'd':
            current_d = value
            current_m = d
        elif variable == 'm':
            current_m = value
            current_d = d
        else:
            raise ValueError("Unsupported ablation variable")

        for method in results.keys():
            OT_reg = 0.01 if "reg" in method else None
            simulated_true_As, estimated_As = master_sde_simulation_and_estimation(
                noise_type, T, dt, num_trajectories, num_sdes, current_d, current_m,
                A_initialization, G_initialization,
                fixed_starting_point, starting_point,
                A_params, G_params, sparsity, scale_identity,
                method.replace("_reg", ""), OT_reg
            )

            mse = np.mean([np.mean((est_A - tru_A) ** 2) for est_A, tru_A in zip(estimated_As, simulated_true_As)])
            results[method].append(mse)

    return range_values, results




def plot_mse(range_values, results, ablation_variable):
    """
    Plot the mean squared error across different values of the ablation variable.

    Parameters:
        range_values (list): The range of values for the ablation variable.
        results (dict): Dictionary containing MSE values for each method keyed by the method name.
        ablation_variable (str): The name of the ablation variable to label the x-axis appropriately.
    """
    plt.figure(figsize=(10, 5))
    for method, mses in results.items():
        plt.plot(range_values, mses, label=method.replace("_", " ").title(), marker='o')

    plt.title('MSE Across Different Values of ' + ablation_variable.title())
    plt.xlabel(ablation_variable.title() + ' Value')
    plt.ylabel('Mean Squared Error')
    plt.legend()
    plt.grid(True)
    plt.show()
def generate_negative_eigenvalue_matrix(dimension):
    """ Generate a symmetric matrix with negative eigenvalues. """
    A = np.random.randn(dimension, dimension)
    A = (A + A.T) / 2  # Make the matrix symmetric
    for i in range(dimension):
        A[i, i] -= np.sum(np.abs(A[i])) + 1  # Ensure diagonal dominance with negative values
    return A

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

# Parameters
A = 1  # Drift coefficient
G = 0.1 # Diffusion coefficient
T = 1.0  # Time horizon
N = 10000  # Number of time steps
dt = T / N  # Time step size
num_simulations = 1000  # Number of simulations


# Function to simulate X_t
def simulate_X_t(A, G, T, N, num_simulations):
    dt = T / N
    X = np.zeros((num_simulations, N + 1))
    W = np.random.normal(scale=np.sqrt(dt), size=(num_simulations, N))

    for i in range(1, N + 1):
        X[:, i] = X[:, i - 1] + A * X[:, i - 1] * dt + G * W[:, i - 1]

    return X


# Function to compute integrals
def compute_integrals(X, dt):
    integral_XdW = np.sum(X[:, :-1] * np.random.normal(scale=np.sqrt(dt), size=X[:, :-1].shape), axis=1)
    integral_X2dt = np.sum(X[:, :-1] ** 2 * dt, axis=1)
    return integral_XdW, integral_X2dt

# if __name__ == "__main__":
#     # Simulate X_t
#     X = simulate_X_t(A, G, T, N, num_simulations)
#
#     # Compute integrals
#     integral_XdW, integral_X2dt = compute_integrals(X, dt)
#
#     # Compute Pearson correlation coefficient as independence score
#     correlation_coefficient, _ = pearsonr(integral_XdW, integral_X2dt)
#
#     print(f"Pearson correlation coefficient: {correlation_coefficient}")
#
#     # Plot the results
#     plt.scatter(integral_XdW, integral_X2dt)
#     plt.xlabel(r'$\int_0^T X_t \, dW_t$')
#     plt.ylabel(r'$\int_0^T X_t^2 \, dt$')
#     plt.title('Scatter plot of the two integrals')
#     plt.show()

# if __name__ == "__main__":
#     # Define variables
#     noise_type = "additive"
#     T, dt = 1.0, 0.01
#     num_trajectories, num_sdes, d, m = 50, 10, 2, 2
#     ablation_variable = 'd'
#     ablation_range = [1,2,3]
#
#     # Run ablation study
#     range_values, ablation_results = perform_ablation_study(
#         ablation_variable, ablation_range,
#         noise_type, T, dt, num_trajectories, num_sdes, d, m,
#         "random", "random", True, None,
#         None, None, None, 1
#     )
#
#     # Plot the results
#     plot_mse(range_values, ablation_results, ablation_variable)