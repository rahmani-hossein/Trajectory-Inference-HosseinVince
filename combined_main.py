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
import numpy as np
from sde_parameter_estimation import parameter_estimation, utils

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.set_default_dtype(torch.float64)
cuda_available = torch.cuda.is_available()
if cuda_available:
    print('Using cuda optimization')
else:
    print('Not using cuda optimization')


if __name__ == "__main__":
    # setup simulation parameters
    dim = 1  # dimension of simulation
    sim_steps = 1000  # number of steps to use for Euler-Maruyama method
    T = 50  # number of timepoints
    N = 100  # number of particles per timepoint
    D = 1  # diffusivity
    t_final = 1  # simulation run on [0, t_final]
    dt = t_final / T

    # A = np.identity(dim)
    # for i in range(0,dim-1):
    #     A[i,i+1] = 5
    A = np.random.randn(dim, dim)

    magnitude = 1
    # Adjust the diagonal entries to ensure negative eigenvalues
    for i in range(dim):
        # Making diagonal entries negatively dominant
        # Subtract the sum of absolute values of row elements (excluding diagonal) and add a magnitude
        A[i, i] = np.abs(A[i, i]) + np.sum(np.abs(A[i, :])) - np.abs(A[i, i]) + magnitude

    A = [[3.5]]

    def drift(x, t, dim=dim):
        return (A @ x.T).T


    # branching rates
    beta = lambda x, t: 5 * ((np.tanh(2 * x[0]) + 1) / 2)
    delta = lambda x, t: 0

    # function for particle initialisation
    ic_func = lambda N, d: np.random.randn(N, dim) * 0.1

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

    # plot samples
    plt.scatter(np.kron(np.linspace(0, t_final, T), np.ones(N)), sim.x[:, 0], alpha=0.1, color="red")
    plt.xlabel("t");
    plt.ylabel("dim 1")
    plt.show()

    # convert sim.x to standard form
    X = np.zeros((N, T, dim))
    for i in range(T):
        X[:, i, :] = sim.x[sim.t_idx == i]

    # set up gWOT model
    # no a priori estimate on the branching (g = 1)
    lamda_reg = 0.00215
    eps_df = D * dt
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
    with torch.no_grad():
        paths_nullgrowth = bs.sample_paths(None, N=N_paths, coord=True, x_all=[sim.x, ] * sim.T,
                                           get_gamma_fn=lambda i: model_nullgrowth.get_coupling_reg(i,
                                                                                                    K=model_nullgrowth.get_K(
                                                                                                        i)).cpu(),
                                           num_couplings=sim.T - 1)
        # paths_growth = bs.sample_paths(None, N = N_paths, coord = True, x_all = [sim.x, ]*sim.T,
        #                     get_gamma_fn = lambda i : model_growth.get_coupling_reg(i, K = model_growth.get_K(i)).cpu(), num_couplings = sim.T-1)

    # parameter estimation

    print(paths_nullgrowth.shape)
    print(dt)
    est_A = -parameter_estimation.estimate_A_exp(paths_nullgrowth, dt)
    print(est_A)
    print(A)
    print(np.mean((est_A - A) ** 2))