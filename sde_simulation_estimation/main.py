from plots import *
from parameter_estimation import *
from simulate_trajectories import *



# def simulate_trajectories()

if __name__ == "__main__":
    # Set parameters for trajectories
    num_trajectories = 1000
    T = 1
    dt = 0.02
    dim = 3
    n_sdes = 100
    dimensions = [5]
    mse_scores = []
    mse_scores_ = []
    standard_errors = []
    standard_errors_ = []
    num_num_trajectories = [100, 500, 1000]

    for num_trajectories in num_num_trajectories:
        mse_dim_scores = []
        mse_dim_scores_ = []
        for i in range(n_sdes):
            A = generate_negative_eigenvalue_matrix(dim)
            G = 0.1*np.eye(dim)
            X0 = np.random.randn(dim)
            X = multiple_ou_trajectories(num_trajectories, T, dt, A, G, X0)
            A_hat, GGT_hat = estimate_drift_diffusion(X, T, dt, use_estimated_GGT=False, expectation=True, OT=True)
            A_hat_, GGT_hat_ = estimate_drift_diffusion(X, T, dt, use_estimated_GGT=False, expectation=True, OT=False)
            mse = np.mean((A_hat - A) ** 2)
            mse_dim_scores.append(mse)
            mse_ = np.mean((A_hat_ - A) ** 2)
            mse_dim_scores_.append(mse_)

        mean_mse = np.mean(mse_dim_scores)
        mean_mse_ = np.mean(mse_dim_scores_)
        std_error = np.std(mse_dim_scores) / np.sqrt(n_sdes)  # Standard error of the mean
        std_error_ = np.std(mse_dim_scores_) / np.sqrt(n_sdes)
        mse_scores.append(mean_mse)
        mse_scores_.append(mean_mse_)
        standard_errors.append(std_error)
        standard_errors_.append(std_error_)
        print(f'Mean MSE (OT) for {num_trajectories} trajectories: {mean_mse}, Standard Error: {std_error}')
        print(f'Mean MSE (Trajectory) for {num_trajectories} trajectories: {mean_mse_}, Standard Error: {std_error_}')

    # Plotting the results
    plt.errorbar(num_num_trajectories, mse_scores, yerr=standard_errors, fmt='-o', label='OT')
    plt.errorbar(num_num_trajectories, mse_scores_, yerr=standard_errors_, fmt='-^', label='Trajectory')
    # plt.yscale('log')
    plt.xlabel('Number of Trajectories')
    plt.ylabel('Mean Squared Error (MSE)')
    plt.title('Parameter Estimation on d-dimensional Stationary Linear Additive Noise SDE')
    plt.legend()
    plt.grid(True)
    plt.show()

#
# if __name__ == "__main__":
#     # Set parameters for trajectories
#     num_trajectories = 100
#     T = 1
#     dt = 0.02
#     dim = 10
#     # m = 2
#     mse_scores = []
#     n_sdes = 100
#     for i in range(n_sdes):
#         A = generate_random_matrix(dim)#np.array([[1.76, -0.1], [0.98, 0]])#generate_negative_eigenvalue_matrix(dim) #
#         G = 0.5*generate_random_matrix(dim)#np.eye(dim) #[np.array([[-0.11, -0.14],[-0.29,-0.22]]), np.array([[-0.17, 0.59],[0.81,0.18]]) ]
#         #[0.9* np.eye(dim) for _ in range(dim)]   #np.array([[-0.11, -0.14], [-0.29, -0.22]]) #0.1*np.eye(dim)    # Variance matrix
#         # GGT = np.matmul(G, np.transpose(G))
#         X0 = np.random.randn(dim) #np.array([1.87, -0.98]) #np.random.randn(dim) #(np.zeros(dim))  #        # Initial values
#         X = multiple_ou_trajectories(num_trajectories, T, dt, A, G, X0) #multiple_multiplicative_noise_trajectories(num_trajectories, T, dt, A, G, X0) #
#         # plot_trajectories(X[0], T, dt) # plotting one trajectory
#         A_hat, GGT_hat = estimate_drift_diffusion(X, T, dt, use_estimated_GGT = True, expectation = True)
#         mse = 1/dim**2*np.linalg.norm((A_hat - A)**2)
#         mse_scores.append(mse)
#     print(f'Mean MSE across {n_sdes} randomly generated SDEs of dimension {dim}: ', np.mean(mse_scores))

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
