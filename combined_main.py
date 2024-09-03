import matplotlib.pyplot as plt
import torch
import gwot
from gwot import models, sim
import gwot.bridgesampling as bs
import numpy as np
from sde_parameter_estimation import parameter_estimation

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.set_default_dtype(torch.float64)
cuda_available = torch.cuda.is_available()
if cuda_available:
    print('Using cuda optimization')
else:
    print('Not using cuda optimization')

def param_est_OT(A, D, X, dt, t_final, entropy_reg = 0, N_paths_gen = 1000):
    A = -A
    # print('helper marginals:', marginals)
    est_A_OT, est_G_OT = parameter_estimation.estimate_A_exp_ot_with_traj(X, dt, T = t_final, entropy_reg = entropy_reg, estimate_G=True, cur_est_A=A, frac_other_time_samples=0.5)
    if entropy_reg == 0:
        print(f'dt = {dt}, estimated A from classic OT:', est_A_OT)
        print(f'dt = {dt}, estimated D from classic OT:', est_G_OT)
    else:
        print(f'dt = {dt}, estimated A from regularized OT with eps={entropy_reg}:', est_A_OT)
        print(f'dt = {dt}, estimated D from regularized OT with eps={entropy_reg}:', est_G_OT)
    print(f'A bias:', est_A_OT[0][0] - A[0][0])
    print(f'D bias:', est_G_OT[0][0] - D)
    # print('MSE:', np.mean((est_A_OT - A) ** 2))


def seeded_random_vector(N, dim, n_seed):
    np.random.seed(n_seed)
    return np.random.randn(N, dim) * 0.1

def X0_func(init_type):
    if init_type == 'none':
        return lambda N, d: np.random.randn(N, dim) * 0.1
    elif init_type == '0':
        return lambda N, d: np.zeros((N, dim))
    elif init_type == '1':
        return lambda N, d: np.ones((N, dim))
    elif init_type == 'intermediate':
        return lambda N, d: seeded_random_vector(N, dim, n_seed)

if __name__ == "__main__":
    run_gwot = False
    A = np.array([[1]])
    A_scalar = A[0][0]
    D = 1  # diffusivity
    # setup simulation parameters
    # n_seed = 0
    # np.random.seed(n_seed)
    init_type = '1'
    eps_multiplier = 1
    dim = 1  # dimension of simulation
    sim_steps = 1000  # number of steps to use for Euler-Maruyama method
    # T = 10  # number of timepoints
    N = 5 # number of particles per timepoint
    t_final = 1  # simulation run on [0, t_final]
    for T in [10, 20, 50, 100]:
        dt = t_final / T


        def drift(x, t, dim=dim):
            return (A @ x.T).T


        # branching rates
        beta = lambda x, t: 5 * ((np.tanh(2 * x[0]) + 1) / 2)
        delta = lambda x, t: 0

        # function for particle initialisation
        ic_func = X0_func(init_type)

        # setup simulation object
        sim = gwot.sim.Simulation(V=None, dV=drift, birth_death=False,
                                  birth=0,
                                  death=0,
                                  N=np.repeat(N, T),
                                  T=T,
                                  d=dim,
                                  D=D,
                                  t_final=t_final,
                                  ic_func=ic_func,
                                  pool=None)

        # sample from simulation
        sim.sample(steps_scale=int(sim_steps / sim.T), trunc=N)

        # # plot samples
        plt.scatter(np.kron(np.linspace(0, t_final, T), np.ones(N)), sim.x[:, 0], alpha=0.1, color="red")
        plt.xlabel("t");
        plt.ylabel("dim 1")
        plt.show()

        # convert sim.x to standard form
        X = np.zeros((N, T, dim))
        print(sim.x)
        for i in range(T):
            X[:, i, :] = sim.x[sim.t_idx == i]
        #
        print('adjusted X from sim: ', X[:5, :5, :])
        # with open('gwot_simulated_A-1_D-1_N-20_dt-0.01.pickle', 'wb') as f:
        #     pickle.dump(X, f)
        # np.save('gwot_simulated_A-1_D-1_N-20_dt-0.01', X)
        # for i in range(1):
        #     plots.plot_trajectories(X[i], t_final, dt)

        if run_gwot:
            # set up gWOT model
            # no a priori estimate on the branching (g = 1)
            lamda_reg = 0.00215
            eps_df = D * dt * eps_multiplier
            print('setting up the model')
            if cuda_available:
                model_nullgrowth = gwot.models.OTModel(sim, lamda_reg=lamda_reg,
                                                       eps_df=eps_df * torch.ones(sim.T).cuda(),
                                                       growth_constraint="exact",
                                                       pi_0="uniform",
                                                       use_keops=True,
                                                       device=device)
            else:
                model_nullgrowth = gwot.models.OTModel(sim, lamda_reg=lamda_reg,
                                                       eps_df=eps_df * torch.ones(sim.T),
                                                       growth_constraint="exact",
                                                       pi_0="uniform",
                                                       use_keops=True,
                                                       device=device)

            # solve both gWOT models using L-BFGS
            print('optimizing via L-BFGS')
            for m in [model_nullgrowth]:
                m.solve_lbfgs(steps=25,
                              max_iter=50,
                              lr=1,
                              history_size=50,
                              line_search_fn='strong_wolfe',
                              factor=2,
                              tol=1e-4,
                              retry_max=0)

            # path sampling from gwot
            # sample paths
            N_paths = 5000
            print('generating gWOT trajectories')
            with torch.no_grad():
                paths_nullgrowth = bs.sample_paths(None, N=N_paths, coord=True, x_all=[sim.x, ] * sim.T,
                                                   get_gamma_fn=lambda i: model_nullgrowth.get_coupling_reg(i,
                                                                                                            K=model_nullgrowth.get_K(
                                                                                                                i)).cpu(),
                                                   num_couplings=sim.T - 1)
                # paths_growth = bs.sample_paths(None, N = N_paths, coord = True, x_all = [sim.x, ]*sim.T,
                #                     get_gamma_fn = lambda i : model_growth.get_coupling_reg(i, K = model_growth.get_K(i)).cpu(), num_couplings = sim.T-1)
            # np.save(f'X0-{init_type}_paths_nullgrowth_d-{dim}_from_N-{N}_A-{A_scalar}_diff-{D}_dt-{dt}.npy', paths_nullgrowth)
            # parameter estimation
            print(paths_nullgrowth.shape)
            print('dt:', dt)
            est_A = parameter_estimation.estimate_A_exp(paths_nullgrowth, dt)
            print('true A:', A)
            print('estimated A from GWOT:', est_A)
            print(f'A bias:', est_A[0][0] - A[0][0])
            # print('MSE:', np.mean((est_A - A) ** 2))
            est_G = parameter_estimation.estimate_GGT(paths_nullgrowth, T)
            print('estimated D from GWOT:', est_G)
            print('True D', D)
            print(f'D bias:', est_G[0][0] - D)

        # param_est_OT(A, D, X, dt, t_final, entropy_reg=D * dt)
        # param_est_OT(A, D, X, dt, t_final, entropy_reg=0)

