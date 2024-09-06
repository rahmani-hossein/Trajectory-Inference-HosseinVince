
from utils.load_save import *
import pickle
from parameter_estimation import *
import numpy as np
from simulate_trajectories import *
from plots import *
import math
from scipy.stats import norm

# Instantiate A_trues and G_trues
A_trues = np.array([[0,1], [-1, 0]])
G_trues = np.eye(2)



# Save the data with the specified filename
# save_drifts_diffusions('2D_ex_rotation', A_trues, G_trues)
#
#
# hist_time_jump =5
# A_trues, G_trues, maximal_X_measured_list, max_num_trajectories, max_T, min_dt = load_measurement_data('seed-60_X0-ones_d-1_n_sdes-1_dt-0.01_N-100_T-1.0_D-1.0')
# num_steps = int(max_T / min_dt)
# if True:
#     for i in range(0, num_steps, hist_time_jump):
#         fig, axes = plt.subplots(1, len(maximal_X_measured_list), figsize=(12, 6), sharex=True, sharey=True)
#
#         for m in range(len(maximal_X_measured_list)):
#             # Extract time marginal data
#             time_marginal = maximal_X_measured_list[m][:, i, :]
#
#
#             # Define the mean and standard deviation of the Gaussian
#             mean = 0
#
#             std_dev = math.sqrt(1/2)
#
#             # Define the range for the Gaussian plot based on the standard deviation
#             x_min = mean - 4 * std_dev  # 4 standard deviations below the mean
#             x_max = mean + 4 * std_dev  # 4 standard deviations above the mean
#             x = np.linspace(x_min, x_max, 100)
#
#             # Plot the Gaussian distribution
#             axes.plot(x, norm.pdf(x, loc=mean, scale=std_dev), 'r-', lw=5, alpha=0.6, label='equilibrium distribution')
#
#             axes.set_ylim([0, 5])
#             axes.hist(time_marginal, density=True, bins=50, histtype='stepfilled', alpha=0.2)
#
#
#             # Set title for the individual subplot with equation
#             axes.set_title(
#                 f'$dX_t = {A_trues[m]}X_tdt + \\sqrt{{{G_trues[m]}}}dW_t$')
#
#         # Add a main title to the entire figure
#         plt.suptitle(f'Marginal at time {round(i * min_dt, 2)} given $X_0 = 1$', fontsize=16)
#
#         # Adjust layout to prevent overlap
#         plt.tight_layout(rect=[0, 0, 1, 0.95])  # Leave space for the suptitle
#
#         # Add a legend for the entire figure
#         handles, labels = axes.get_legend_handles_labels()
#         fig.legend(handles, labels, loc='upper right')
#
#
#
#         plt.show()
#
