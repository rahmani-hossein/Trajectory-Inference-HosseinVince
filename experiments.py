from experiments_helpers import *



def default_measurement_settings():
    dt = 0.05
    dt_EM = 0.01
    T = 1
    N = 500
    max_its = 30
    linearization = True
    killed = False
    report_time_splits = False
    return dt, dt_EM, T, N, max_its, linearization, killed, report_time_splits

def run_generic_experiment(points, A, G, d, N = None, verbose=True, gaussian_start = False, actual_confounder = False):
    D = np.matmul(G, G.T)
    print(rf'Generating data for experiment: $dX_t = {A} X_t \, dt + {G} \, dW_t$')
    if gaussian_start:
        mean = np.ones(d)
        # N = 1000  # Number of points to generate
        # Generate N points from the 2D Gaussian distribution
        gaussian_points = np.random.multivariate_normal(mean, D, size=N)
        # Assign equal probabilities to each point
        X0_dist = [(point, 1 / N) for point in gaussian_points]
        print(rf'X0 is initialised uniformly from N(0, {D})')
    else:
        X0_dist = [(point, 1 / len(points)) for point in points]
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
    print(X_measured.shape)
    print('Estimating parameters')
    its = 1
    if actual_confounder:
        d = d-1
        X_measured = X_measured[:, :, :-1]
        G_trunc = G[:-1, :-1]
    else:
        G_trunc = G
    # np.random.seed(43) #for reproducibility
    mean = np.trace(np.matmul(G_trunc, G_trunc.T)) / d  # get average magnitude of main diagonal entry of true diffusion
    order_magnitude = np.random.uniform(low=-1, high=1)
    random_scale = 10 ** order_magnitude
    initial_D = random_scale * mean * np.eye(d)
    print('Initial guess for D:', initial_D)
    est_A_list, est_GGT_list = [], []
    est_A, est_GGT = estimate_A_exp_ot_with_traj(X_measured, dt, T, cur_est_D=initial_D,
                                                       report_time_splits=report_time_splits)
    est_A_list.append(est_A)
    est_GGT_list.append(est_GGT)

    if verbose:
        print(f'Estimated A at iteration {its}:', est_A)
        print(f'Estimated D at iteration {its}:', est_GGT)

    if actual_confounder:
        print(f'MAE to true A at iteration {its}: {compute_mae(est_A, A[:-1, :-1])}')
        print(f'MAE to true D at iteration {its}: {compute_mae(est_GGT, D[:-1, :-1])}')
    else:
        print(f'MAE to true A at iteration {its}: {compute_mae(est_A, A)}')
        print(f'MAE to true D at iteration {its}: {compute_mae(est_GGT, D)}')

    while its < max_its:
        est_A, est_GGT= estimate_A_exp_ot_with_traj(X_measured, dt, T, cur_est_A=est_A, cur_est_D=est_GGT,
                                                           linearization=linearization,
                                                           report_time_splits=report_time_splits)
        est_A_list.append(est_A)
        est_GGT_list.append(est_GGT)
        its += 1
        if verbose:
            print(f'Estimated A at iteration {its}:', est_A)
            print(f'Estimated D at iteration {its}:', est_GGT)
        if actual_confounder:
            print(f'MAE to true A at iteration {its}: {compute_mae(est_A, A[:-1, :-1])}')
            print(f'MAE to true D at iteration {its}: {compute_mae(est_GGT, D[:-1, :-1])}')
        else:
            print(f'MAE to true A at iteration {its}: {compute_mae(est_A, A)}')
            print(f'MAE to true D at iteration {its}: {compute_mae(est_GGT, D)}')

    results_data = {
        'true_A': A,
        'true_G': G,
        'true_D': np.matmul(G, G.T),
        'X0 points': points,
        'initial D': initial_D,
        'est D values': est_GGT_list,
        'est A values': est_A_list,
        'N': N
    }
    return results_data


def run_generic_experiment_replicates(exp_number, num_replicates, N_list = None, version='random', d=None, p = 0.5, causal_sufficiency = False, actual_confounder = True):
    if exp_number == 1:
        d = 1
    elif exp_number == 2 or exp_number == 3:
        d = 2
    if N_list is None:
        N_list = [500]
    # Generate points once to ensure reproducibility
    if actual_confounder:
        points = generate_independent_points(d, d+1)
        points = [np.append(point, 0) for point in points]
    else:
        points = generate_independent_points(d, d+1)
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
                results_data = run_experiment_random(points, d, N=N, p=p, causal_sufficiency=causal_sufficiency, actual_confounder=actual_confounder)

            # Create the filename and filepath
            if exp_number != 'random':
                results_dir = f'Results_experiment_{exp_number}'
                filename = f'version-{version}_N-{N}_replicate-{i}.pkl'
            else:
                results_dir = f'Results_experiment_actual_confounder_random_{d}_sparsity_{p}_mag_1_mean_0'
                filename = f'replicate-{i}_N-{N}.pkl'
            os.makedirs(results_dir, exist_ok=True)
            filepath = os.path.join(results_dir, filename)

            # Save the results using pickle
            with open(filepath, 'wb') as f:
                pickle.dump(results_data, f)


def run_experiment_1(points, version=1, N=None):
    d = 1
    if version == 1:
        A = np.array([[0]])
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

def run_experiment_random(points, d, N= None, causal_sufficiency = True, actual_confounder = True, causal_experiment = True, p = 1/2):
    # np.random.seed(43)
    if actual_confounder:
        A = generate_random_matrix_with_eigenvalue_constraint(d, eigenvalue_threshold=1, sparsity_threshold=p)
        # Expand A to be (d+1) x (d+1) to include the latent confounder
        A = np.pad(A, ((0, 1), (0, 1)), mode='constant', constant_values=0)

        # Select two random variables to be affected by the latent confounder
        affected_indices = np.random.choice(d, 2, replace=False)

        # Set the last column for the affected variables to 1 (excluding the last row)
        for idx in affected_indices:
            A[idx, -1] = 1
    else:
        A = generate_random_matrix_with_eigenvalue_constraint(d, eigenvalue_threshold=1, sparsity_threshold=p)
    if causal_experiment:
        if causal_sufficiency:
            G = np.eye(d)
            np.fill_diagonal(G, np.random.uniform(low=0.1, high=1, size=d))
        else:
            if actual_confounder:
                G = np.eye(d+1)
            else:
                G = np.eye(d)
                if d == 3:
                    num_columns = np.random.choice([1, 2])
                elif d == 5:
                    num_columns = np.random.choice([1, 2, 3])
                elif d == 10:
                    num_columns = np.random.choice([1, 2, 3, 4, 5, 6])
                columns_with_shared_noise = np.random.choice(d, num_columns, replace=False)
                for col in columns_with_shared_noise:
                    num_nonzero_entries = 2 # latent confounder will be over 2 variables
                    nonzero_indices = np.random.choice(d, num_nonzero_entries, replace=False)
                    G[nonzero_indices, col] = 1
    else:
        print("not the causal code")
        G = np.random.uniform(low=-1, high=1, size=(d, d))

    if actual_confounder:
        return run_generic_experiment(points, A, G, d+1, N, actual_confounder=True)
    else:
        return run_generic_experiment(points, A, G, d, N)

def run_safely(exp_number, num_replicates, d):
    try:
        run_generic_experiment_replicates(exp_number=exp_number, num_replicates=num_replicates, d=d)
    except Exception as e:
        print(f"Error occurred during execution with d={d}: {e}")

# Define the list of N values you want to test
N_list = [500]
d_list = [2]
p_list = [0.25]

for p in p_list:
    for d in d_list:
        run_generic_experiment_replicates(exp_number='random', causal_sufficiency=True, actual_confounder = False, d=d, num_replicates=10, N_list=N_list, p=p)



# # Run experiment 2 version 1 with multiple values of N
# run_generic_experiment_replicates(exp_number='random', d = 3, num_replicates=10, N_list=N_list, p=0.5)
# # run_generic_experiment_replicates(exp_number='random', d = 4, num_replicates=10, N_list=N_list)
# run_generic_experiment_replicates(exp_number='random', d = 5, num_replicates=10, N_list=N_list, p =0.5)
# run_generic_experiment_replicates(exp_number='random', d = 10, num_replicates=10, N_list=N_list, )
# run_generic_experiment_replicates(exp_number='random', d = 25, num_replicates=10, N_list=N_list)
# run_generic_experiment_replicates(exp_number='random', d = 50, num_replicates=10, N_list=N_list)

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

