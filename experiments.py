from experiments_helpers import *

def default_measurement_settings():
    dt = 0.02
    dt_EM = 0.001
    T = 1
    N = 100
    return dt, dt_EM, T, N

def run_experiment_1(points, version = 1, linearization = True):
    d = 1
    dt, dt_EM, T, N = default_measurement_settings()
    if version == 1:
        A = np.array([[-1]])
    else:
        A = np.array([[-10]])
    G = np.eye(d)
    X0_dist = [(point, 1 / len(points)) for point in points]
    print(rf'Generating data for experiment 1: $dX_t = {A} X_t \, dt + {G} \, dW_t$')
    print(rf'X0 is initialised uniformly from the points: {points}')
    X_measured = linear_additive_noise_data(N, d=d, T=1, dt_EM=dt, dt=dt, A= A, G =G, X0_dist=X0_dist, stationary=False,
                                              matrix_exponential=True)
    # plot_trajectories(X_measured_2, T=1, dt=0.02, save_file=False, N_truncate=5)
    print('estimating parameters')
    max_its = 50
    its = 1
    mean = np.trace(np.matmul(G, G.T))/d
    matrix = np.zeros((d, d))
    # Fill the diagonal
    np.fill_diagonal(matrix, np.random.uniform(low=mean * 0.1, high=mean * 10, size=d))
    initial_D = matrix
    print('initial guess for D:', initial_D)
    est_A_list, est_GGT_list = [], []
    est_A, est_GGT, X_OT = estimate_A_exp_ot_with_traj(X_measured, dt, T, cur_est_D=initial_D)
    est_A_list.append(est_A)
    est_GGT_list.append(est_GGT)
    # plot_trajectories(X_OT, T, dt, save_file=False, N_truncate=5)
    print(f'Estimated A at iteration {its}:', est_A)
    print(f'Estimated D at iteration {its}:', est_GGT)
    while its < max_its:
        t1 = time.time()
        est_A, est_GGT, X_OT = estimate_A_exp_ot_with_traj(X_measured, dt, T, cur_est_A=est_A, cur_est_D=est_GGT, linearization=linearization)
        t2 = time.time()
        est_A_list.append(est_A)
        est_GGT_list.append(est_GGT)
        # plot_trajectories(X_OT, T, dt, save_file=False, N_truncate=5)
        its += 1
        print(f'Estimated A at iteration {its}:', est_A)
        print(f'Estimated D at iteration {its}:', est_GGT)
        print(f'Iteration time:', t2-t1)
    results_data = {}
    results_data['X0 points'] = points
    results_data['initial D'] = initial_D
    results_data['est D values'] = est_GGT_list
    results_data['est A values'] = est_A_list
    return results_data

def run_experiment_1_iterates(num_iterates=10, version=2, linearization=True):
    all_results = {}

    for i in range(1, num_iterates + 1):
        print(f'\nRunning iterate {i} of experiment 1 version {version}')
        d = 1
        points = generate_independent_points(d, d)  # Generate new points for each iteration
        for point in points:
            print(point, "Magnitude:", np.linalg.norm(point))

        # Run the experiment with the generated points
        results_data = run_experiment_2(points, version=version, linearization=linearization)

        # Save the result in the dictionary with the experiment number as the key
        all_results[i] = results_data

    # Save the entire dictionary as a pickle file
    filename = f'experiment_results_1_version-{version}.pkl'
    with open(filename, 'wb') as f:
        pickle.dump(all_results, f)

    print(f'All iterates of experiment 1 completed and saved to {filename}')
def run_experiment_2(points, version = 1, linearization = True):
    d = 2
    dt, dt_EM, T, N = default_measurement_settings()
    if version == 1:
        A = np.zeros((d,d))
    else:
        A = np.array([[0, 1], [-1, 0]])
    G = np.eye(d)
    X0_dist = [(point, 1 / len(points)) for point in points]
    print(rf'Generating data for experiment 2: $dX_t = {A} X_t \, dt + {G} \, dW_t$')
    print(rf'X0 is initialised uniformly from the points: {points}')
    X_measured = linear_additive_noise_data(N, d=d, T=1, dt_EM=dt, dt=dt, A= A, G =G, X0_dist=X0_dist, stationary=False,
                                              matrix_exponential=True)
    # plot_trajectories(X_measured_2, T=1, dt=0.02, save_file=False, N_truncate=5)
    print('estimating parameters')
    max_its = 50
    its = 1
    mean = np.trace(np.matmul(G, G.T))/d
    matrix = np.zeros((d, d))
    # Fill the diagonal
    np.fill_diagonal(matrix, np.random.uniform(low=mean * 0.1, high=mean * 10, size=d))
    initial_D = matrix
    print('initial guess for D:', initial_D)
    est_A_list, est_GGT_list = [], []
    est_A, est_GGT, X_OT = estimate_A_exp_ot_with_traj(X_measured, dt, T, cur_est_D=initial_D)
    est_A_list.append(est_A)
    est_GGT_list.append(est_GGT)
    # plot_trajectories(X_OT, T, dt, save_file=False, N_truncate=5)
    print(f'Estimated A at iteration {its}:', est_A)
    print(f'Estimated D at iteration {its}:', est_GGT)
    while its < max_its:
        t1 = time.time()
        est_A, est_GGT, X_OT = estimate_A_exp_ot_with_traj(X_measured, dt, T, cur_est_A=est_A, cur_est_D=est_GGT, linearization=linearization)
        t2 = time.time()
        est_A_list.append(est_A)
        est_GGT_list.append(est_GGT)
        # plot_trajectories(X_OT, T, dt, save_file=False, N_truncate=5)
        its += 1
        print(f'Estimated A at iteration {its}:', est_A)
        print(f'Estimated D at iteration {its}:', est_GGT)
        print(f'Iteration time:', t2-t1)
    results_data = {}
    results_data['X0 points'] = points
    results_data['initial D'] = initial_D
    results_data['est D values'] = est_GGT_list
    results_data['est A values'] = est_A_list
    return results_data

def run_experiment_2_iterates(num_iterates=10, version=2, linearization=True):
    all_results = {}

    for i in range(1, num_iterates + 1):
        print(f'\nRunning iterate {i} of experiment 2 version {version}')
        d = 2
        points = generate_independent_points(d, d)  # Generate new points for each iteration
        for point in points:
            print(point, "Magnitude:", np.linalg.norm(point))

        # Run the experiment with the generated points
        results_data = run_experiment_2(points, version=version, linearization=linearization)

        # Save the result in the dictionary with the experiment number as the key
        all_results[i] = results_data

    # Save the entire dictionary as a pickle file
    filename = f'experiment_results_2_version-{version}.pkl'
    with open(filename, 'wb') as f:
        pickle.dump(all_results, f)

    print(f'All iterates of experiment 2 completed and saved to {filename}')


def run_experiment_3(points, version = 1, linearization = True):
    d = 2
    dt, dt_EM, T, N = default_measurement_settings()
    if version == 1:
        A = np.array([[1,2], [1, 0]])
    else:
        A = np.array([[1/3, 4/3], [2/3, -1/3]])
    G = np.array([[1, 2], [-1, -2]])
    X0_dist = [(point, 1 / len(points)) for point in points]
    print(rf'Generating data for experiment 3: $dX_t = {A} X_t \, dt + {G} \, dW_t$')
    print(rf'X0 is initialised uniformly from the points: {points}')
    X_measured = linear_additive_noise_data(N, d=d, T=1, dt_EM=dt, dt=dt, A= A, G =G, X0_dist=X0_dist, stationary=False,
                                              matrix_exponential=True)
    # plot_trajectories(X_measured_2, T=1, dt=0.02, save_file=False, N_truncate=5)
    print('estimating parameters')
    max_its = 50
    its = 1
    mean = np.trace(np.matmul(G, G.T))/d
    matrix = np.zeros((d, d))
    # Fill the diagonal
    np.fill_diagonal(matrix, np.random.uniform(low=mean * 0.1, high=mean * 10, size=d))
    initial_D = matrix
    print('initial guess for D:', initial_D)
    est_A_list, est_GGT_list = [], []
    est_A, est_GGT, X_OT = estimate_A_exp_ot_with_traj(X_measured, dt, T, cur_est_D=initial_D)
    est_A_list.append(est_A)
    est_GGT_list.append(est_GGT)
    # plot_trajectories(X_OT, T, dt, save_file=False, N_truncate=5)
    print(f'Estimated A at iteration {its}:', est_A)
    print(f'Estimated D at iteration {its}:', est_GGT)
    while its < max_its:
        t1 = time.time()
        est_A, est_GGT, X_OT = estimate_A_exp_ot_with_traj(X_measured, dt, T, cur_est_A=est_A, cur_est_D=est_GGT, linearization=linearization)
        t2 = time.time()
        est_A_list.append(est_A)
        est_GGT_list.append(est_GGT)
        # plot_trajectories(X_OT, T, dt, save_file=False, N_truncate=5)
        its += 1
        print(f'Estimated A at iteration {its}:', est_A)
        print(f'Estimated D at iteration {its}:', est_GGT)
        print(f'Iteration time:', t2-t1)
    results_data = {}
    results_data['X0 points'] = points
    results_data['initial D'] = initial_D
    results_data['est D values'] = est_GGT_list
    results_data['est A values'] = est_A_list
    return results_data


def run_experiment_3_iterates(num_iterates=10, version=2, linearization=True):
    all_results = {}

    for i in range(1, num_iterates + 1):
        print(f'\nRunning iterate {i} of experiment 3 version {version}')
        d = 2
        points = generate_independent_points(d, d)  # Generate new points for each iteration
        for point in points:
            print(point, "Magnitude:", np.linalg.norm(point))

        # Run the experiment with the generated points
        results_data = run_experiment_3(points, version=version, linearization=linearization)

        # Save the result in the dictionary with the experiment number as the key
        all_results[i] = results_data

    # Save the entire dictionary as a pickle file
    filename = f'experiment_results_3_version-{version}.pkl'
    with open(filename, 'wb') as f:
        pickle.dump(all_results, f)

    print(f'All iterates of experimetn 3 completed and saved to {filename}')

# Run the experiments
run_experiment_3_iterates(num_iterates=10, version=1, linearization=True)
run_experiment_3_iterates(num_iterates=10, version=2, linearization=True)

# Run the experiments
run_experiment_1_iterates(num_iterates=10, version=1, linearization=True)
run_experiment_1_iterates(num_iterates=10, version=2, linearization=True)

# Run the experiments
run_experiment_2_iterates(num_iterates=10, version=1, linearization=True)
run_experiment_2_iterates(num_iterates=10, version=2, linearization=True)

# filename = 'experiment_results_2.pkl'
# with open(filename, 'rb') as f:
#     data = pickle.load(f)
#
# print(data[1]['est D values'])








