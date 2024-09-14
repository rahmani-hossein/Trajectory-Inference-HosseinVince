from experiments_helpers import *



def default_measurement_settings():
    dt = 0.05
    dt_EM = 0.01
    T = 1
    N = 500
    max_its = 2
    linearization = True
    killed = False
    report_time_splits = True
    return dt, dt_EM, T, N, max_its, linearization, killed, report_time_splits


def run_generic_experiment(points, A, G, d, N = None, verbose=False):
    D = np.matmul(G, G.T)
    X0_dist = [(point, 1 / len(points)) for point in points]
    print(rf'Generating data for experiment: $dX_t = {A} X_t \, dt + {G} \, dW_t$')
    print(rf'X0 is initialised uniformly from the points: {points}')
    dt, dt_EM, T, N_, max_its, linearization, killed, report_time_splits, = default_measurement_settings()
    if N is None:
        N = N_
    print(
        f"dt: {dt}, dt_EM: {dt_EM}, T: {T}, N: {N}, max_its: {max_its}, linearization: {linearization}, killed: {killed}")

    if not killed:
        X_measured = linear_additive_noise_data(N, d=d, T=T, dt_EM=dt_EM, dt=dt, A=A, G=G, X0_dist=X0_dist)
    else:
        X_measured = killed_linear_additive_noise_data(N, d=d, T=T, dt_EM=dt_EM, dt=dt, A=A, G=G, X0_dist=X0_dist)

    print('Estimating parameters')
    its = 1
    np.random.seed(43) #for reproducibility
    mean = np.trace(np.matmul(G, G.T)) / d  # get average magnitude of main diagonal entry of true diffusion
    order_magnitude = np.random.uniform(low=-1, high=1)
    random_scale = 10 ** order_magnitude
    initial_D = random_scale * mean * np.eye(d)
    print('Initial guess for D:', initial_D)
    est_A_list, est_GGT_list = [], []
    est_A, est_GGT, X_OT = estimate_A_exp_ot_with_traj(X_measured, dt, T, cur_est_D=initial_D,
                                                       report_time_splits=report_time_splits)
    est_A_list.append(est_A)
    est_GGT_list.append(est_GGT)

    if verbose:
        print(f'Estimated A at iteration {its}:', est_A)
        print(f'Estimated D at iteration {its}:', est_GGT)

    print(f'MAE to true A at iteration {its}: {compute_mae(est_A, A)}')
    print(f'MAE to true D at iteration {its}: {compute_mae(est_GGT, D)}')

    while its < max_its:
        est_A, est_GGT, X_OT = estimate_A_exp_ot_with_traj(X_measured, dt, T, cur_est_A=est_A, cur_est_D=est_GGT,
                                                           linearization=linearization,
                                                           report_time_splits=report_time_splits)
        est_A_list.append(est_A)
        est_GGT_list.append(est_GGT)
        its += 1
        if verbose:
            print(f'Estimated A at iteration {its}:', est_A)
            print(f'Estimated D at iteration {its}:', est_GGT)
        print(f'MAE to true A at iteration {its}: {compute_mae(est_A, A)}')
        print(f'MAE to true D at iteration {its}: {compute_mae(est_GGT, D)}')

    results_data = {
        'true_A': A,
        'true_D': np.matmul(G, G.T),
        'X0 points': points,
        'initial D': initial_D,
        'est D values': est_GGT_list,
        'est A values': est_A_list,
        'N': N
    }
    return results_data


def run_generic_experiment_replicates(exp_number, num_replicates, N_list, version='random', d=None):
    if exp_number == 1:
        d = 1
    elif exp_number == 2 or exp_number == 3:
        d = 2

    # Generate points once to ensure reproducibility
    points = generate_independent_points(d, d)
    for N in N_list:
        print(f'\nRunning experiments for N={N}')
        for i in range(1, num_replicates + 1):
            print(f'\nRunning replicate {i} of experiment {exp_number} version {version} with N={N}')
            if exp_number == 1:
                results_data = run_experiment_1(points, version=version, N=N)
            elif exp_number == 2:
                results_data = run_experiment_2(points, version=version, N=N)
            elif exp_number == 3:
                results_data = run_experiment_3(points, version=version, N=N)
            elif exp_number == 'random':
                results_data = run_experiment_random(points, d, N=N)

            # Create the filename and filepath
            if exp_number != 'random':
                results_dir = f'Results_experiment_{exp_number}'
                filename = f'version-{version}_N-{N}_replicate-{i}.pkl'
            else:
                results_dir = f'Results_experiment_{exp_number}_{d}'
                filename = f'replicate-{i}_N-{N}.pkl'
            os.makedirs(results_dir, exist_ok=True)
            filepath = os.path.join(results_dir, filename)

            # Save the results using pickle
            with open(filepath, 'wb') as f:
                pickle.dump(results_data, f)


def run_experiment_1(points, version=1, N=None):
    d = 1
    if version == 1:
        A = np.array([[-1]])
        G = np.eye(d)
    else:
        A = np.array([[-10]])
        G = math.sqrt(10) * np.eye(d)
    return run_generic_experiment(points, A, G, d, N, verbose = True)

def run_experiment_2(points, version=1, N=500):
    d = 2
    if version == 1:
        A = np.zeros((d, d))
    else:
        A = np.array([[0, 1], [-1, 0]])
    G = np.eye(d)
    return run_generic_experiment(points, A, G, d, N, verbose=True)


def run_experiment_3(points, version=1, N = 500):
    d = 2
    if version == 1:
        A = np.array([[1, 2], [1, 0]])
    else:
        A = np.array([[1 / 3, 4 / 3], [2 / 3, -1 / 3]])
    G = np.array([[1, 2], [-1, -2]])
    return run_generic_experiment(points, A, G, d, N, verbose = True)

def run_experiment_random(points, d, N= None):
    np.random.seed(43)
    A = generate_random_matrix_with_eigenvalue_constraint(d, eigenvalue_threshold=1)
    G = np.random.uniform(low=-1, high=1, size=(d, d))
    return run_generic_experiment(points, A, G, d, N)

def run_safely(exp_number, num_replicates, d):
    try:
        run_generic_experiment_replicates(exp_number=exp_number, num_replicates=num_replicates, d=d)
    except Exception as e:
        print(f"Error occurred during execution with d={d}: {e}")

# Define the list of N values you want to test
N_list = [2000]

# Run experiment 2 version 1 with multiple values of N
run_generic_experiment_replicates(exp_number=3, d = 1, num_replicates=1, N_list=N_list, version=1)

# Safely running the experiments
# run_safely(exp_number='random', num_replicates=10, d=20)
# run_safely(exp_number='random', num_replicates=10, d=50)
# run_safely(exp_number='random', num_replicates=10, d=100)
# run_safely(exp_number='random', num_replicates=10, d=5)
# run_safely(exp_number='random', num_replicates=10, d=6)
# run_safely(exp_number='random', num_replicates=10, d=7)
# run_safely(exp_number='random', num_replicates=10, d=8)
# run_safely(exp_number='random', num_replicates=10, d=9)
# run_safely(exp_number='random', num_replicates=10, d=10)

# run_generic_experiment_replicates(exp_number=1, num_replicates=10, version=2)
# run_generic_experiment_replicates(exp_number=1, num_replicates=10, version=1)
# run_generic_experiment_replicates(exp_number=3, num_replicates=10, version=1)
# run_generic_experiment_replicates(exp_number=3, num_replicates=10, version=2)
# # for Hossein
# run_generic_experiment_replicates(exp_number=2, num_replicates=10, version=1)
# # for Hossein
# run_generic_experiment_replicates(exp_number=2, num_replicates=10, version=2)

