from parameter_estimation import *
import numpy as np
from simulate_trajectories import *
from plots import *
import matplotlib.gridspec as gridspec
import math
from scipy.stats import multivariate_normal
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.distance import jensenshannon
from scipy.stats import wasserstein_distance
from matplotlib.colors import Normalize
from scipy.stats import norm
from fractions import Fraction
from matplotlib.patches import Patch


A_list = []
G_list = []
d = 2
epsilon = 0
matrix = np.array([[0, 1], [-1, 0]]) #np.zeros((d,d)) #np.array([[-1,0], [0,-1]])
A = matrix #matrix #+ np.array([[0, 1], [-1, 0]]) #+ np.array([[0, 7], [-1, 0]])  #np.ones((d,d))*epsilon  #-2* np.eye(d)
A_list.append(A)
G = np.eye(d) #np.zeros((d,d))#np.eye(d)#np.array([[1,0,0],[1,0,0], [1,0,0]])#np.array([[1,2], [-1,-2]]) #np.zeros((d,d))#np.array([[0.11,0.22], [-0.11,-0.22]])#np.array([[1,-0.5], [0.5,5]])
G_list.append(G)

A = np.zeros((d,d))
A_list.append(A)
G = np.eye(d)#np.zeros((d,d))#np.eye(d)#np.array([[1,0,0],[1,0,0], [1,0,0]])#np.array([[1,2], [-1,-2]]) #np.zeros((d,d))#np.array([[0.11,0.22], [-0.11,-0.22]])#np.array([[1,-0.5], [0.5,5]])
G_list.append(G)
# cons = np.array([[0, 1], [-1, 0]]) #np.array([[0, 1], [-1, 0]]) #np.array([[0, 1], [-1, 0]])#np.array([[0, 7], [-1, 0]])# -2* np.eye(d)
# A = matrix + cons #np.array([[1/3, 4/3], [2/3, -1/3]])
# A_list.append(A)
#
# G =np.eye(d) #np.array([[1,2], [-1,-2]]) #np.eye(d) #np.array([[0.11,0.22], [-0.11,-0.22]])#np.array([[1,0], [0,5]])
# G_list.append(G)
# D = np.matmul(G, np.transpose(G))


N = 100
T = 10
dt = 0.05
dt_EM = dt#0.001
alpha = 0 #0.6
scale = 10
max_its = 100
stationary_start = True
hist_time_jump = 5
plot_trajectory_graph = False
estimate_parameters = False
OT_solver = True
plot_histograms = True
plot_3d = False
save_histograms = False
unkilled = True

X_measured_list = []

count = 1
for i in range(len(A_list)):
    A = A_list[i]
    print('true A:', A)
    G = G_list[i]
    D = np.matmul(G, G.T)
    print('true D:', D)
    # avg = D.sum()/np.count_nonzero(D)

    n_measured_times = int(T / dt)
    if unkilled:
        np.random.seed(0)
        if stationary_start:
            if count < 2:
                beps = False
            else:
                beps = False
            X_measured = true_multi_ou_process(N, d, T, dt_EM, dt, A, G,
                                                   X0=None, beps = beps)
            count += 1
        else:
            # if count < 2:
            #     X0_prob_dist = [
            #         (np.array([1, -1]), 1/2),
            #         (np.array([-1, 1]), 1 / 2)
            # #     ]
            # else:
            #     X0_prob_dist = [
            #         (np.array([1, 0]), 1/2),
            #         (np.array([-1, 0]), 1 / 2)
            #     ]
            # count += 1
            points = [np.array([1,-1])]
            X0_prob_dist = [(point, 1/len(points)) for point in points]

            # mean = np.array([0, 0])
            # cov = np.array([[1, 0], [0, 1]])
            # N = 100  # Number of points to generate
            #
            # # Generate N points from the 2D Gaussian distribution
            # gaussian_points = np.random.multivariate_normal(mean, cov, size=N)
            #
            # # Assign equal probabilities to each point
            # X0_prob_dist = [(point, 1 / N) for point in gaussian_points]
            X_measured = true_multi_ou_process(N, d, T, dt_EM, dt, A, G, X0_prob_dist = X0_prob_dist)
        X_measured_list.append(X_measured)
        # est_A_bib = estimate_A_bibbona(X_measured, dt)
        # print(f'estimated A (bibbona) = {est_A_bib}')
        if estimate_parameters:
            its = 1
            if OT_solver:
                # X_OT = create_OT_traj(X_measured, entropy_reg = avg*dt, dt=dt, metric='sqeuclidean',
                #                   frac_other_time_samples=alpha)
                X_OT_ = create_OT_traj_md(X_measured, D=np.eye(d), dt=dt,
                                      frac_other_time_samples=alpha)
            else:
                X_OT = X_measured
            # est_A_exp = estimate_A_exp(X_OT, dt)
            # print(f'estimated A (from package) iteration {its} = {est_A_exp}')
            # est_D = estimate_GGT(X_OT, T, est_A=est_A_exp)
            # print(f'estimated D (from package) iteration {its} = {est_D}')
            est_A_exp_ = estimate_A_exp(X_OT_, dt)
            print(f'estimated A (multi-dim Sinkhorn) iteration {its} = {est_A_exp_}')
            est_D_ = estimate_GGT(X_OT_, T, est_A=est_A_exp_)
            print(f'estimated D (multi-dim Sinkhorn)) iteration {its} = {est_D_}')
            while its < max_its:
                # X_OT = create_OT_traj(X_measured, entropy_reg = avg*dt, dt=dt, metric='sqeuclidean',
                #                       frac_other_time_samples=alpha, cur_est_A=est_A_exp, linearization=True)
                X_OT_ = create_OT_traj_md(X_measured, D= est_D_, dt=dt,
                                      frac_other_time_samples=alpha, cur_est_A=est_A_exp_, linearization=True)
                est_A_exp_ = estimate_A_exp(X_OT_, dt)
                its += 1
                # print(f'estimated A (from package) iteration {its} = {est_A_exp}')
                # est_A_exp_ = estimate_A_exp(X_OT_, dt)
                # est_D = estimate_GGT(X_OT, T, est_A=est_A_exp)
                # print(f'estimated D (from package) iteration {its} = {est_D}')
                print(f'estimated A (multi-dim Sinkhorn) iteration {its} = {est_A_exp_}')
                est_D_ = estimate_GGT(X_OT_, T, est_A=est_A_exp_)
                print(f'estimated D (multi-dim Sinkhorn) iteration {its} = {est_D_}')
        if plot_trajectory_graph:
            plot_trajectories(X_measured, T, dt, save_file=False, N_truncate=5)
    else:
        np.random.seed(0)
        if stationary_start:
            X_measured = generate_maximal_dataset_cell_measurement_death(N, T, dt, d, dt_EM, A, G, stationary=True)
        else:
            X_measured = generate_maximal_dataset_cell_measurement_death(N, T, dt, d, dt_EM, A, G, stationary=False, X0=scale*np.ones(d))
        X_measured_list.append(X_measured)
        if estimate_parameters:
            its = 1
            X_OT = create_OT_traj_md(X_measured, np.matmul(G, np.transpose(G)), dt=dt, frac_other_time_samples=alpha)
            # est_A_bib = estimate_A_bibbona(X_OT, dt)
            # print(f'estimated A (bibbona) = {est_A_bib}')
            est_A_exp = estimate_A_exp(X_OT, dt)
            print(f'estimated A (exp) iteration {its} = {est_A_exp}')
            # est_D = estimate_GGT_split(X_OT, T, est_A=None)
            # print(f'estimated D split = {est_D}')
            est_D = estimate_GGT(X_OT, T, est_A = est_A_exp)
            print(f'estimated D iteration {its} = {est_D}')
            while its < max_its:
                X_OT = create_OT_traj_md(X_measured, np.matmul(G, np.transpose(G)), dt=dt,
                                      frac_other_time_samples=alpha, cur_est_A=est_A_exp, linearization=True)
                est_A_exp = estimate_A_exp(X_OT, dt)
                its += 1
                print(f'estimated A (exp) iteration {its} = {est_A_exp}')
                est_D = estimate_GGT(X_OT, T, est_A=est_A_exp)
                print(f'estimated D iteration {its} = {est_D}')

            if plot_trajectory_graph:
                plot_trajectories(X_OT, T, dt, save_file=False, N_truncate=5)


# Helper function to compute Jensen-Shannon Divergence
def compute_jsd(P, Q):
    # Flatten histograms and add a small epsilon to avoid division by zero
    epsilon = 1e-10
    P = P.flatten() + epsilon
    Q = Q.flatten() + epsilon

    # Normalize histograms
    P /= np.sum(P)
    Q /= np.sum(Q)

    return jensenshannon(P, Q) ** 2


# Boolean flag to control whether to plot in 3D or 2D
num_steps = int(T / dt)
scaling_factor = 1  # Adjust this scaling factor if needed
confidence_interval = 0.99  # confidence interval for filtering outliers

def ensure_positive_definite(cov_matrix, epsilon=1e-6):
    """
    Ensures that the covariance matrix is positive definite by adding a small value to the diagonal.
    """
    # Try to perform Cholesky decomposition to check if the matrix is positive definite
    try:
        np.linalg.cholesky(cov_matrix)
    except np.linalg.LinAlgError:
        # If it's not positive definite, add a small value to the diagonal
        cov_matrix += np.eye(cov_matrix.shape[0]) * epsilon
    return cov_matrix

def filter_outliers(data, confidence_interval):
    """
    Filters outliers beyond a given confidence interval.
    Returns a boolean mask indicating which rows are within the range.
    """
    lower_bound = np.percentile(data, (1 - confidence_interval) * 50, axis=0)
    upper_bound = np.percentile(data, confidence_interval * 100 + (1 - confidence_interval) * 50, axis=0)
    mask = np.all((data >= lower_bound) & (data <= upper_bound), axis=1)
    return mask

x_min, x_max, y_min, y_max = None, None, None, None

# First pass: determine global axis limits across all datasets and time steps
for i in range(0, num_steps, hist_time_jump):
    for m in range(len(X_measured_list)):
        time_marginal = X_measured_list[m][:, i, :]

        # Filter outliers based on the 99% confidence interval
        mask = filter_outliers(time_marginal, confidence_interval)
        filtered_time_marginal = time_marginal[mask]

        # Determine min and max across all datasets for consistent axis limits
        x_min_current, x_max_current = np.min(filtered_time_marginal[:, 0]), np.max(filtered_time_marginal[:, 0])
        y_min_current, y_max_current = np.min(filtered_time_marginal[:, 1]), np.max(filtered_time_marginal[:, 1])

        if x_min is None or x_min_current < x_min:
            x_min = x_min_current
        if x_max is None or x_max_current > x_max:
            x_max = x_max_current
        if y_min is None or y_min_current < y_min:
            y_min = y_min_current
        if y_max is None or y_max_current > y_max:
            y_max = y_max_current

# Main loop for creating histograms and plots
for i in range(0, num_steps, hist_time_jump):
    all_histograms = []

    # First pass: determine global axis limits
    for m in range(len(X_measured_list)):
        time_marginal = X_measured_list[m][:, i, :]

        # Filter outliers based on the confidence interval
        mask = filter_outliers(time_marginal, confidence_interval)
        filtered_time_marginal = time_marginal[mask]

        # Determine min and max across all datasets for consistent axis limits
        x_min_current, x_max_current = np.min(filtered_time_marginal[:, 0]), np.max(filtered_time_marginal[:, 0])
        y_min_current, y_max_current = np.min(filtered_time_marginal[:, 1]), np.max(filtered_time_marginal[:, 1])

        if x_min is None or x_min_current < x_min:
            x_min = x_min_current
        if x_max is None or x_max_current > x_max:
            x_max = x_max_current
        if y_min is None or y_min_current < y_min:
            y_min = y_min_current
        if y_max is None or y_max_current > y_max:
            y_max = y_max_current

    x_max = max(x_max, y_max)
    y_max = max(x_max, y_max)
    x_min = min(x_min, y_min)
    y_min = min(x_min, y_min)

    # Prepare the figure and gridspec layout
    fig = plt.figure(figsize=(12, 6))
    gs = gridspec.GridSpec(1, len(X_measured_list) + 1, width_ratios=[1] * len(X_measured_list) + [0.05])

    axes = [fig.add_subplot(gs[0, m]) for m in range(len(X_measured_list))]

    for m, ax in enumerate(axes):
        time_marginal = X_measured_list[m][:, i, :]

        # Filter outliers
        mask = filter_outliers(time_marginal, confidence_interval)
        filtered_time_marginal = time_marginal[mask]

        # Add artificial points at the corners to ensure no white space
        artificial_points = np.array([
            [x_min, y_min],
            [x_min, y_max],
            [x_max, y_min],
            [x_max, y_max]
        ])
        extended_data = np.vstack([filtered_time_marginal, artificial_points])

        # Create a 2D histogram of the marginal data
        H, xedges, yedges = np.histogram2d(extended_data[:, 0], extended_data[:, 1], bins=(100, 100), density=True)
        all_histograms.append(H)

        # Create meshgrid for the plot
        x_midpoints = (xedges[:-1] + xedges[1:]) / 2
        y_midpoints = (yedges[:-1] + yedges[1:]) / 2
        X, Y = np.meshgrid(x_midpoints, y_midpoints)

        # Plot the filled contour to fill the background
        contour = ax.contourf(X, Y, H.T, levels=100, cmap='viridis')

        # Set consistent axis limits using global limits
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.axhline(0, color='white', linewidth=.5)
        ax.axvline(0, color='white', linewidth=.5)

        ax.set_xlabel('X axis')
        ax.set_ylabel('Y axis')

        # A = [[Fraction(item).limit_denominator() for item in row] for row in A]


        # Set the title using the formatted LaTeX string
        ax.set_title(f'$dX_t = ${A_list[m]} $X_tdt$ + {G_list[m]} $dW_t$', fontsize=10)
        # ax.set_title(f'dX_t = {A_list[m]}X_tdt + [{G_list[m]}]dW_t', fontsize=10)

        # Calculate the empirical mean and covariance
        empirical_mean = np.mean(filtered_time_marginal, axis=0)
        empirical_covariance = np.cov(filtered_time_marginal, rowvar=False)

        # Ensure the covariance matrix is positive definite
        empirical_covariance = ensure_positive_definite(empirical_covariance)

        # Create a grid for the Gaussian PDF
        pos = np.dstack((X, Y))

        # Fit a multivariate normal distribution based on empirical mean and covariance
        rv = multivariate_normal(empirical_mean, empirical_covariance, allow_singular=True)
        Z = rv.pdf(pos)

        # Overlay the Gaussian PDF as a contour plot with a few contour levels
        num_contour_levels = 5  # Choose the number of contour levels (e.g., 5 for a few rings)
        contour_levels = np.linspace(Z.min(), Z.max(), num_contour_levels)

        # Ensure that contour levels are strictly increasing and within the range of the data
        contour_levels = np.sort(np.unique(contour_levels))
        contour_levels = contour_levels[contour_levels > Z.min()]  # Remove levels at or below Z.min()
        if i > 0 and len(contour_levels) > 0:
            ax.contour(X, Y, Z, levels=contour_levels, colors='red')

    # Create a shared color bar for all subplots
    cbar_ax = fig.add_subplot(gs[0, -1])  # Last column for color bar
    cbar = fig.colorbar(contour, cax=cbar_ax)
    cbar.ax.tick_params(labelsize=10)  # Adjust the tick label size if needed

    plt.subplots_adjust(top=0.8)
    # Set the title
    plt.suptitle(
        f'Marginal at time {round(i * dt, 2)} given $X_0 \sim \\text{{unif}}\{{(1, 0), (0, 1)\}}$',
        fontsize=14,
        y=0.95
    )
    # Save or show the plot
    os.makedirs('FP_graphs', exist_ok=True)
    os.makedirs(os.path.join('FP_graphs', f'{A_list[m]}'), exist_ok=True)
    if save_histograms:
        plot_filename = os.path.join(f'FP_graphs/{A_list[m]}', f"plot_step-{i}.png")
        if os.path.isfile(plot_filename):
            os.remove(plot_filename)
        plt.savefig(plot_filename)
    if plot_histograms:
        plt.show()

    # Compute Earth Mover's Distance (EMD) between the two histograms
    if len(X_measured_list) == 2:
        H1, H2 = all_histograms[0], all_histograms[1]
        if plot_histograms:
            # Flatten the histograms and compute EMD
            emd = wasserstein_distance(H1.flatten(), H2.flatten())
            print(f"Earth Mover's Distance (EMD) at time {round(i * dt, 2)}: {emd}")

            # Calculate Jensen-Shannon Divergence
            jsd = compute_jsd(H1, H2)
            print(f"Jensen-Shannon Divergence at time {round(i * dt, 2)}: {jsd}")

            # Compute the JSD between the marginal and the N(0, I) reference
            jsd_value = compute_jsd(H1, Z)
            print(f"Jensen-Shannon Divergence to Standard Gaussian at time {round(i * dt, 2)}: {jsd_value}")


# # x_min, x_max, y_min, y_max, z_min, z_max = None, None, None, None, None, None
# scaling_factor = 1.5  # Adjust this scaling factor for the Gaussian bell curve in 3D
#
# for i in range(0, num_steps, hist_time_jump):
#     all_histograms = []
#
#     # First pass: determine global axis limits
#     for m in range(len(X_measured_list)):
#         time_marginal = X_measured_list[m][:, i, :]
#
#         # Filter outliers based on the 99% confidence interval
#         mask = filter_outliers(time_marginal, confidence_interval)
#         filtered_time_marginal = time_marginal[mask]
#
#         # Determine min and max across all datasets for consistent axis limits
#         x_min_current, x_max_current = np.min(filtered_time_marginal[:, 0]), np.max(filtered_time_marginal[:, 0])
#         y_min_current, y_max_current = np.min(filtered_time_marginal[:, 1]), np.max(filtered_time_marginal[:, 1])
#
#         if x_min is None or x_min_current < x_min:
#             x_min = x_min_current
#         if x_max is None or x_max_current > x_max:
#             x_max = x_max_current
#         if y_min is None or y_min_current < y_min:
#             y_min = y_min_current
#         if y_max is None or y_max_current > y_max:
#             y_max = y_max_current
#
#     x_max = max(x_max, y_max)
#     y_max = max(x_max, y_max)
#     x_min = min(x_min, y_min)
#     y_min = min(x_min, y_min)
#
#
#     # Prepare the figure and axes
#     fig, axes = plt.subplots(1, len(X_measured_list), figsize=(12, 6), subplot_kw={'projection': '3d'} if plot_3d else {}, sharex=True, sharey=True)
#
#     for m in range(len(X_measured_list)):
#         time_marginal = X_measured_list[m][:, i, :]
#
#         # Filter outliers based on the 99% confidence interval
#         mask = filter_outliers(time_marginal, confidence_interval)
#         filtered_time_marginal = time_marginal[mask]
#
#         # Create a 2D histogram of the marginal data
#         H, xedges, yedges = np.histogram2d(filtered_time_marginal[:, 0], filtered_time_marginal[:, 1], bins=(100, 100), density=True)
#         all_histograms.append(H)
#
#         A = A_list[m]
#         eigvals, eigvecs = np.linalg.eig(A)
#
#         # Extract the real part of the eigenvectors (since the eigenvalues are complex)
#         eigvec1 = np.real(eigvecs[:, 0])  # Corresponds to the rotation
#         eigvec2 = np.real(eigvecs[:, 1])  # Orthogonal direction to the rotation
#
#
#
#         # Create meshgrid for the plot
#         x_midpoints = (xedges[:-1] + xedges[1:]) / 2
#         y_midpoints = (yedges[:-1] + yedges[1:]) / 2
#         X, Y = np.meshgrid(x_midpoints, y_midpoints)
#
#         ax = axes[m]
#
#         if plot_3d:
#             # Plot a 3D surface for the histogram
#             histogram_surface = ax.plot_surface(X, Y, H.T, cmap='viridis', alpha=0.6)
#
#             # Define the mean and covariance for the Gaussian distribution
#             empirical_mean = np.mean(filtered_time_marginal, axis=0)
#             empirical_covariance = np.cov(filtered_time_marginal, rowvar=False)
#
#             # Ensure the covariance matrix is positive definite
#             empirical_covariance = ensure_positive_definite(empirical_covariance)
#
#             # Apply the scaling factor to the covariance matrix
#             empirical_covariance *= scaling_factor
#
#             # Create a grid for the Gaussian PDF
#             pos = np.dstack((X, Y))
#             rv = multivariate_normal(empirical_mean, empirical_covariance)
#             Z = rv.pdf(pos)
#
#             # Normalize Z to ensure the Gaussian is scaled properly
#             Z /= Z.max()  # Optional normalization
#
#             # Overlay the Gaussian PDF as a surface plot
#             gaussian_surface = ax.plot_surface(X, Y, Z, color='red', alpha=0.4)
#
#             # Plot the eigenvectors as arrows on the contour plot
#             ax.quiver(empirical_mean[0], empirical_mean[1], eigvec1[0], eigvec1[1], color='blue', scale=5, width=0.005)
#             ax.quiver(empirical_mean[0], empirical_mean[1], eigvec2[0], eigvec2[1], color='green', scale=5, width=0.005)
#
#             # Determine z_min and z_max for consistent Z-axis scaling
#             z_min_current, z_max_current = np.min(Z), np.max(Z)
#
#             if z_min is None or z_min_current < z_min:
#                 z_min = z_min_current
#             if z_max is None or z_max_current > z_max:
#                 z_max = z_max_current
#
#             ax.set_zlabel('Probability Density')
#
#         else:
#             # Add artificial points at the corners to ensure no white space
#             artificial_points = np.array([
#                 [x_min, y_min],
#                 [x_min, y_max],
#                 [x_max, y_min],
#                 [x_max, y_max]
#             ])
#             extended_data = np.vstack([filtered_time_marginal, artificial_points])
#
#             # Create a 2D histogram of the marginal data
#             H, xedges, yedges = np.histogram2d(extended_data[:, 0], extended_data[:, 1],
#                                                bins=(100, 100), density=True)
#             all_histograms.append(H)
#
#             # Create meshgrid for the plot
#             x_midpoints = (xedges[:-1] + xedges[1:]) / 2
#             y_midpoints = (yedges[:-1] + yedges[1:]) / 2
#             X, Y = np.meshgrid(x_midpoints, y_midpoints)
#
#             ax = axes[m]
#
#             # Plot the filled contour to fill the background
#             contour = ax.contourf(X, Y, H.T, levels=100, cmap='viridis')
#
#             # Set consistent axis limits using global limits
#             ax.set_xlim(x_min, x_max)
#             ax.set_ylim(y_min, y_max)
#             ax.axhline(0, color='white', linewidth=.5)
#             ax.axvline(0, color='white', linewidth=.5)
#
#             ax.set_xlabel('X axis')
#             ax.set_ylabel('Y axis')
#             # ax.set_title(f'Marginal at time {round(i * dt, 2)} for $dX_t = {A}X_tdt + {G_list[m]}dW_t$ given $X_0 = {point}$ ')
#             ax.set_title(f'dX_t = {A}X_tdt + [{G_list[m]}]dW_t', fontsize=8)
#
#
#             # Calculate the empirical mean and covariance
#             empirical_mean = np.mean(filtered_time_marginal, axis=0)
#             empirical_covariance = np.cov(filtered_time_marginal, rowvar=False)
#
#             # Ensure the covariance matrix is positive definite
#             empirical_covariance = ensure_positive_definite(empirical_covariance)
#
#             # Create a grid for the Gaussian PDF
#             pos = np.dstack((X, Y))
#
#             # Fit a multivariate normal distribution based on empirical mean and covariance
#             rv = multivariate_normal(empirical_mean, empirical_covariance, allow_singular=True)
#             Z = rv.pdf(pos)
#
#             # Overlay the Gaussian PDF as a contour plot with a few contour levels
#             num_contour_levels = 5  # Choose the number of contour levels (e.g., 5 for a few rings)
#             contour_levels = np.linspace(Z.min(), Z.max(), num_contour_levels)
#
#             # Ensure that contour levels are strictly increasing and within the range of the data
#             contour_levels = np.sort(np.unique(contour_levels))
#             contour_levels = contour_levels[contour_levels > Z.min()]  # Remove levels at or below Z.min()
#             if i > 0:
#                 # Plot the contour with the selected levels
#                 if len(contour_levels) > 0:  # Ensure there are valid contour levels
#                     ax.contour(X, Y, Z, levels=contour_levels, colors='red')
#     # Create a shared color bar for all subplots in the 2D case
#     if not plot_3d:
#         fig.colorbar(contour, ax=axes.ravel().tolist(), orientation='vertical', fraction=0.05)
#         # fig.colorbar(contour, ax=axes.ravel().tolist(), orientation='vertical', fraction=0.05)
#
#     os.makedirs('FP_graphs', exist_ok=True)
#     os.makedirs(os.path.join('FP_graphs', f'{A_list[m]}'), exist_ok=True)
#     if stationary_start:
#         plot_filename = os.path.join(f'FP_graphs/{A_list[m]}',
#                                      f"ellipse_unkilled-{unkilled}_X0-normal_{A_list[m]}_step-{i}.png")
#     else:
#         plot_filename = os.path.join(f'FP_graphs/{A_list[m]}',
#                                      f"unkilled-{unkilled}_X0_scale-{scale}_{A_list[m]}_step-{i}.png")
#     if save_histograms:
#         plt.suptitle(f'Marginal at time {round(i * dt, 2)} given X_0 = {point}', fontsize=14)
#         if os.path.isfile(plot_filename):
#             os.remove(plot_filename)
#         plt.savefig(plot_filename)
#     if plot_histograms:
#         plt.suptitle(f'Marginal at time {round(i * dt, 2)} given X_0 = {point}', fontsize=14)
#         plt.tight_layout(pad=1.08)
#         plt.show()
