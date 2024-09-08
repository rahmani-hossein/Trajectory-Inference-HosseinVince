from experiments_helpers import *


def default_measurement_settings():
    dt = 0.05
    dt_EM = 0.01
    T = 1
    N = 500
    max_its = 30
    linearization = True
    killed = False
    report_time_splits = True
    return dt, dt_EM, T, N, max_its, linearization, killed, report_time_splits


def run_generic_experiment(points, A, G, d):
    X0_dist = [(point, 1 / len(points)) for point in points]
    print(rf'Generating data for experiment 1: $dX_t = {A} X_t \, dt + {G} \, dW_t$')
    print(rf'X0 is initialised uniformly from the points: {points}')
    dt, dt_EM, T, N, max_its, linearization, killed, report_time_splits = default_measurement_settings()
    print(
        f"dt: {dt}, dt_EM: {dt_EM}, T: {T}, N: {N}, max_its: {max_its}, linearization: {linearization}, killed: {killed}")
    if not killed:
        X_measured = linear_additive_noise_data(N, d=d, T=1, dt_EM=dt_EM, dt=dt, A=A, G=G, X0_dist=X0_dist)
    else:
        X_measured = killed_linear_additive_noise_data(N, d=d, T=1, dt_EM=dt_EM, dt=dt, A=A, G=G, X0_dist=X0_dist)
    # plot_trajectories(X_measured_2, T=1, dt=0.02, save_file=False, N_truncate=5)
    print('estimating parameters')
    its = 1
    mean = np.trace(np.matmul(G, G.T)) / d  # get average magnitude of main diagonal entry of true diffusion
    order_magnitude = np.random.uniform(low=-1, high=1)
    random_scale = 10**(order_magnitude)
    initial_D = random_scale * mean * np.eye(d)
    print('initial guess for D:', initial_D)
    est_A_list, est_GGT_list = [], []
    est_A, est_GGT, X_OT = estimate_A_exp_ot_with_traj(X_measured, dt, T, cur_est_D=initial_D,
                                                       report_time_splits=report_time_splits)
    est_A_list.append(est_A)
    est_GGT_list.append(est_GGT)
    # plot_trajectories(X_OT, T, dt, save_file=False, N_truncate=5)
    print(f'Estimated A at iteration {its}:', est_A)
    print(f'Estimated D at iteration {its}:', est_GGT)
    while its < max_its:
        t1 = time.time()
        est_A, est_GGT, X_OT = estimate_A_exp_ot_with_traj(X_measured, dt, T, cur_est_A=est_A, cur_est_D=est_GGT,
                                                           linearization=linearization,
                                                           report_time_splits=report_time_splits)
        t2 = time.time()
        est_A_list.append(est_A)
        est_GGT_list.append(est_GGT)
        # plot_trajectories(X_OT, T, dt, save_file=False, N_truncate=5)
        its += 1
        print(f'Estimated A at iteration {its}:', est_A)
        print(f'Estimated D at iteration {its}:', est_GGT)
        if report_time_splits:
            print(f'Iteration time:', t2 - t1)
    results_data = {}
    results_data['X0 points'] = points
    results_data['initial D'] = initial_D
    results_data['est D values'] = est_GGT_list
    results_data['est A values'] = est_A_list
    return results_data

def run_generic_experiment_replicates(exp_number, num_replicates, version):
    if exp_number == 1:
        d = 1
    elif exp_number == 2 or exp_number == 3:
        d = 2
    for i in range(1, num_replicates + 1):
        print(f'\nRunning iterate {i} of experiment {exp_number} version {version}')
        points = generate_independent_points(d, d)  # Generate new points for each iteration
        for point in points:
            print(point, "Magnitude:", np.linalg.norm(point))
        if exp_number == 1:
            results_data = run_experiment_1(points, version=version)
        elif exp_number == 2:
            results_data = run_experiment_2(points, version=version)
        elif exp_number == 3:
            results_data = run_experiment_3(points, version=version)
        # Ensure the directory exists
        results_dir = f'Results_experiment_{exp_number}'
        os.makedirs(results_dir, exist_ok=True)

        # Create the filename and filepath
        filename = f'version-{version}_replicate-{i}.pkl'
        filepath = os.path.join(results_dir, filename)

        # Get a unique filepath to avoid overwriting
        unique_filepath = save_with_unique_filename(filepath)

        # Save the results using pickle
        with open(unique_filepath, 'wb') as f:
            pickle.dump(results_data, f)

def run_experiment_1(points, version=1):
    d = 1
    if version == 1:
        A = np.array([[-1]])
        G = np.eye(d)
    else:
        A = np.array([[-10]])
        G = math.sqrt(10) * np.eye(d)
    return run_generic_experiment(points, A, G, d)

def run_experiment_2(points, version=1):
    d = 2
    if version == 1:
        A = np.zeros((d, d))
    else:
        A = np.array([[0, 1], [-1, 0]])
    G = np.eye(d)
    return run_generic_experiment(points, A, G, d)

def run_experiment_3(points, version=1):
    d = 2
    if version == 1:
        A = np.array([[1, 2], [1, 0]])
    else:
        A = np.array([[1 / 3, 4 / 3], [2 / 3, -1 / 3]])
    G = np.array([[1, 2], [-1, -2]])
    return run_generic_experiment(points, A, G, d)


# run_generic_experiment_replicates(exp_number=1, num_replicates=10, version=2)
# run_generic_experiment_replicates(exp_number=1, num_replicates=10, version=1)
# run_generic_experiment_replicates(exp_number=3, num_replicates=10, version=1)
# run_generic_experiment_replicates(exp_number=3, num_replicates=10, version=2)
# for Hossein
run_generic_experiment_replicates(exp_number=2, num_replicates=10, version=1)
# for Hossein
run_generic_experiment_replicates(exp_number=2, num_replicates=10, version=2)

