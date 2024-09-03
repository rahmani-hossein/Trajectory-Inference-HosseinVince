from parameter_estimation import *
import numpy as np
from simulate_trajectories import *
from plots import *
import math
from scipy.stats import norm

N = 100
d = 1
T = 1
dt = 0.01
dt_EM = 0.001
alpha = 0 #0.6
scale = 1
stationary_start = False
hist_time_jump = 1
plot_trajectory_graph = False
estimate_parameters = True
plot_histograms = False
save_histograms = False
unkilled = False


ratio = 1
print(f"The diffusion-drift ratio is set to {ratio}")

drift_scales = [1, 10]
X_measured_list = []
A_list = []
D_list = []

for drift_scale in drift_scales:
    A = -np.array([drift_scale])
    A_list.append(-drift_scale)
    D = drift_scale * ratio
    stationary_variance = D / (2 * d * drift_scale)
    print('stationary variance:', stationary_variance)

    sigma = math.sqrt(D)
    G = np.array([sigma])
    D_list.append(D)
    print("drift scale: ", drift_scale)
    print("diffusivity D= ", D)

    n_measured_times = int(T / dt)
    if unkilled:
        X_measured = np.zeros((N, n_measured_times, d))
        rate = int(dt/ dt_EM)
        for n in range(N):
            if stationary_start:
                np.random.seed(n) # for reproducibility
                X0 = np.array([np.random.randn() * math.sqrt(stationary_variance)])
            else:
                X0 = scale * np.ones(d)
            X_n = ou_process(T, dt_EM, A, G, X0, seed = n)
            for i in range(n_measured_times):
                X_measured[n, i, :] = X_n[i * rate, :]
        X_measured_list.append(X_measured)
        if estimate_parameters:
            est_A_bib = estimate_A_bibbona(X_measured, dt)
            print(f'estimated A (bibbona) = {est_A_bib[0][0]}')
            est_A_exp = estimate_A_exp(X_measured, dt)
            print(f'estimated A (exp) = {est_A_exp[0][0]}')
            est_D = estimate_GGT_(X_measured, T)
            print(f'estimated D split = {est_D[0][0]}')
            est_D = estimate_GGT(X_measured, T)
            print(f'estimated D = {est_D[0][0]}')
        if plot_trajectory_graph:
            plot_trajectories(X_measured, T, dt, save_file=False, N_truncate=5)
    else:
        if stationary_start:
            X_measured = generate_maximal_dataset_cell_measurement_death(N, T, dt, d, dt_EM, A, G, stationary=True)
        else:
            X_measured = generate_maximal_dataset_cell_measurement_death(N, T, dt, d, dt_EM, A, G, stationary=False, X0=scale*np.ones(d))
        X_measured_list.append(X_measured)
        if estimate_parameters:
            X_OT = create_OT_traj(X_measured, entropy_reg=D*dt, dt=dt, metric='sqeuclidean', frac_other_time_samples=alpha)
            est_A_bib = estimate_A_bibbona(X_OT, dt)
            print(f'estimated A (bibbona) = {est_A_bib[0][0]}')
            est_A_exp = estimate_A_exp(X_OT, dt)
            print(f'estimated A (exp) = {est_A_exp[0][0]}')
            est_D = estimate_GGT_split(X_OT, T, est_A=est_A_exp)
            print(f'estimated D split = {est_D[0][0]}')
            est_D = estimate_GGT(X_OT, T, est_A = est_A_bib)
            print(f'estimated D = {est_D[0][0]}')
            if plot_trajectory_graph:
                plot_trajectories(X_OT, T, dt, save_file=False, N_truncate=5)

#
# num_steps = int(T / dt)
# if plot_histograms:
#     for i in range(0, num_steps, hist_time_jump):
#         for m in range(len(X_measured_list)):
#             # Extract time marginal data
#             time_marginal = X_measured_list[m][:, i, :]
#
#             # Define the mean and standard deviation of the Gaussian
#             mean = 0
#             std_dev = math.sqrt(stationary_variance)
#
#             # Define the range for the Gaussian plot based on the standard deviation
#             x_min = mean - 4 * std_dev  # 4 standard deviations below the mean
#             x_max = mean + 4 * std_dev  # 4 standard deviations above the mean
#             x = np.linspace(x_min, x_max, 100)
#
#             # Create the plot
#             fig, ax = plt.subplots(1, 1)
#
#             # Plot the Gaussian distribution
#             ax.plot(x, norm.pdf(x, loc=mean, scale=std_dev), 'r-', lw=5, alpha=0.6, label='equilibrium distribution')
#
#             # Plot the histogram
#             ax.hist(time_marginal, density=True, bins=50, histtype='stepfilled', alpha=0.2)
#
#             # Set the x-axis limits to show the full range of the normal distribution
#             ax.set_xlim([x_min, x_max])
#             ax.set_ylim([0, 5])
#
#             # Add title and show the plot
#             plt.title(f'Marginal at time {round(i * dt,2)} for $dX_t = {A_list[m]}X_tdt + \\sqrt{{{D_list[m]}}}$dW_t')
#
#             plt.legend(loc='upper right')
#             os.makedirs('FP_graphs', exist_ok=True)
#             os.makedirs(os.path.join('FP_graphs', f'{A_list[m]}_ratio-{ratio}'), exist_ok=True)
#             if stationary_start:
#                 plot_filename = os.path.join(f'FP_graphs/{A_list[m]}_ratio-{ratio}', f"unkilled-{unkilled}_stationary_{A_list[m]}_step-{i}.png")
#             else:
#                 plot_filename = os.path.join(f'FP_graphs/{A_list[m]}_ratio-{ratio}',
#                                              f"unkilled-{unkilled}_X0_scale-{scale}_{A_list[m]}_step-{i}.png")
#             if save_histograms:
#                 if os.path.isfile(plot_filename):
#                     os.remove(plot_filename)
#                 plt.savefig(plot_filename)
#             plt.show()
num_steps = int(T / dt)
if plot_histograms:
    for i in range(0, num_steps, hist_time_jump):
        fig, axes = plt.subplots(1, len(X_measured_list), figsize=(12, 6), sharex=True, sharey=True)

        for m in range(len(X_measured_list)):
            # Extract time marginal data
            time_marginal = X_measured_list[m][:, i, :]

            # Define the mean and standard deviation of the Gaussian
            mean = 0
            std_dev = math.sqrt(stationary_variance)

            # Define the range for the Gaussian plot based on the standard deviation
            x_min = mean - 4 * std_dev  # 4 standard deviations below the mean
            x_max = mean + 4 * std_dev  # 4 standard deviations above the mean
            x = np.linspace(x_min, x_max, 100)

            # Create the plot
            ax = axes[m]  # Use subplot axes

            # Plot the Gaussian distribution
            ax.plot(x, norm.pdf(x, loc=mean, scale=std_dev), 'r-', lw=5, alpha=0.6, label='equilibrium distribution')

            # Plot the histogram
            ax.hist(time_marginal, density=True, bins=50, histtype='stepfilled', alpha=0.2)

            # Set the x-axis limits to show the full range of the normal distribution
            ax.set_xlim([x_min, x_max])
            ax.set_ylim([0, 5])

            # Set title for the individual subplot with equation
            ax.set_title(
                f'$dX_t = {A_list[m]}X_tdt + \\sqrt{{{D_list[m]}}}dW_t$')

        # Add a main title to the entire figure
        plt.suptitle(f'Marginal at time {round(i * dt, 2)} given $X_0 = {X0}$', fontsize=16)

        # Adjust layout to prevent overlap
        plt.tight_layout(rect=[0, 0, 1, 0.95])  # Leave space for the suptitle

        # Add a legend for the entire figure
        handles, labels = ax.get_legend_handles_labels()
        fig.legend(handles, labels, loc='upper right')

        # Create directories if they don't exist
        os.makedirs('FP_graphs', exist_ok=True)
        os.makedirs(os.path.join('FP_graphs', f'{A_list[0]}_ratio-{ratio}'), exist_ok=True)

        # Save or show the plot
        if stationary_start:
            plot_filename = os.path.join(f'FP_graphs/{A_list[0]}_ratio-{ratio}',
                                         f"unkilled-{unkilled}_stationary_step-{i}.png")
        else:
            plot_filename = os.path.join(f'FP_graphs/{A_list[0]}_ratio-{ratio}',
                                         f"unkilled-{unkilled}_X0_scale-{scale}_step-{i}.png")
        if save_histograms:
            if os.path.isfile(plot_filename):
                os.remove(plot_filename)
            plt.savefig(plot_filename)

        plt.show()







