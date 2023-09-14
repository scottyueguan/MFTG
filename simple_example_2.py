import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import CSS4_COLORS as mcolors
import pickle as pkl
import os

from simple_example.examples import SimpleExample2
from simple_example.solver import Solver
from simple_example.visualizers import visualize_value, get_intersection
from utils import ROOT_PATH

from simple_example.emp_dist import empirical_dist

if __name__ == "__main__":
    SOLVE_COR = False
    PLOT_VALUE = False
    SOLVE_DEVIATE = False
    PLOT_DEVIATE = True

    res = 500
    rho = 0.375

    # save directories
    data_path = ROOT_PATH / "data"
    figure_path = ROOT_PATH / "figures"

    # set up save directory
    if not os.path.exists(data_path):
        os.makedirs(data_path)
    if not os.path.exists(figure_path):
        os.makedirs(figure_path)

    example = SimpleExample2(rho=rho)
    solver = Solver(game=example, blue_resolution_list=[res, res, res], red_resolution_list=[res, res, res])

    # solve the game
    if SOLVE_COR:
        solver.solve()
        solver.save(data_path)

    # load the game solution
    data = solver.load(data_path)

    # plot value functions
    if PLOT_VALUE:
        ax0 = visualize_value(value=data["maxmin_value"][0],
                              blue_mesh=data["mesh_lists"][0][0], red_mesh=data["mesh_lists"][1][0], elev=30, azim=130)
        ax0.set_zlabel("$J^{\\rho \star}_{\mathrm{cor}, 0}$", rotation=0)
        ax0.set_xlabel("$\mu^\\rho_0(x^1)$")
        ax0.set_ylabel("$\\nu^\\rho_0(y^1)$")
        plt.savefig(figure_path / 'simple_example2_J0.svg', format='svg', dpi=800)

        ax1 = visualize_value(value=data["maxmin_value"][1],
                              blue_mesh=data["mesh_lists"][0][1], red_mesh=data["mesh_lists"][1][1], elev=30, azim=130)
        ax1.set_zlabel("$J^{\\rho \star}_{\mathrm{cor}, 1}$", rotation=0)
        ax1.set_xlabel("$\mu^\\rho_1(x^1)$")
        ax1.set_ylabel("$\\nu^\\rho_1(y^1)$")
        p_list, q_list = [1 / np.sqrt(2) for _ in range(51)], np.linspace(0.0, 1.0, 51)
        z_list = get_intersection(x_list=p_list, y_list=q_list, surf_list=data["maxmin_value"][1],
                                  mesh_x=data["mesh_lists"][0][1], mesh_y=data["mesh_lists"][1][1])
        ax1.plot(p_list, q_list, z_list, mcolors['navy'], alpha=0.8, linestyle="dashed")
        plt.savefig(figure_path / 'simple_example2_J1.svg', format='svg', dpi=800)

        ax2 = visualize_value(value=data["maxmin_value"][2],
                              blue_mesh=data["mesh_lists"][0][2], red_mesh=data["mesh_lists"][1][2], elev=30, azim=130)
        ax2.set_zlabel("$J^{\\rho \star}_{\mathrm{cor}, 2}$", rotation=0)
        ax2.set_xlabel("$\mu^\\rho_2(x^1)$")
        ax2.set_ylabel("$\\nu^\\rho_2(y^1)$")
        plt.savefig(figure_path / 'simple_example2_J2.svg', format='svg', dpi=800)

        plt.show()

    # compute finite-population difference
    if SOLVE_DEVIATE:
        mu0 = [1.0, 0.0]
        nu0 = [0.4, 0.6]
        n_blue_list = np.array([3, 6, 12, 30, 39, 48, 54, 60, 75, 90, 105, 120, 135, 150, 165])
        n_list = (n_blue_list / rho).astype(int)
        n_red_list = ((1 - rho) * n_list).astype(int)

        # find coordinator game value
        # blue_index = np.where(abs(data["mesh_lists"][0][0] - mu0[0]) < 1e-5)
        # red_index = np.where(abs(data["mesh_lists"][1][0] - nu0[0]) < 1e-5)
        # coord_game_value = data["maxmin_value"][0][blue_index, red_index]
        coord_game_value = -nu0[0]

        # compute finite population value
        blue_best_config_list = []
        best_transition_list = []
        J_N_Opt_list = []  # value for both blue identical and non-identical against non-identical red
        J_N_Opt_blue_list = []  # value for identical red against non-identical blue
        blue_1_config_list = []
        blue_identical_policy = [1 / np.sqrt(2), 1 - 1 / np.sqrt(2)]
        for n_blue in n_blue_list:
            best_p1 = np.floor(n_blue / np.sqrt(2)) / n_blue
            best_p2 = np.ceil(n_blue / np.sqrt(2)) / n_blue
            diff1 = (best_p1 - (1 / np.sqrt(2))) ** 2
            diff2 = (best_p2 - (1 / np.sqrt(2))) ** 2
            if diff1 > diff2:
                best_p = best_p2
            else:
                best_p = best_p1
            assert (best_p - np.round(n_blue / np.sqrt(2)) / n_blue) < 1e-5

            blue_best_config_list.append([best_p, 1 - best_p])
            best_transition_matrix = example.red_transition_matrix(policy=[[0, 1], [0, 1]],
                                                                    nu=nu0, mu=[best_p, 1 - best_p], t=1)
            best_transition_list.append(best_transition_matrix[1, :])
            blue_1_config_list.append(empirical_dist(N=n_blue, p=blue_identical_policy))

        for n_red, transition, blue_1_config in zip(n_red_list, best_transition_list, blue_1_config_list):
            n_red_on_0 = nu0[0] * n_red
            n_red_on_1 = n_red - n_red_on_0

            # compute red against non-identical blue
            dist = empirical_dist(N=n_red_on_1, p=transition)
            value = 0
            for node in dist:
                n_red_on_0_prime = n_red_on_0 + node.n_list[0]
                value -= node.prob * (n_red_on_0_prime / n_red)
            J_N_Opt_list.append(value)

            # compute red against identical blue
            value = 0
            for blue_node in blue_1_config:
                mu1 = blue_node.emp_dist
                transition = example.red_transition_matrix(policy=[[0, 1], [0, 1]],
                                                            mu=mu1, nu=nu0)[1, :]

                red_dist = empirical_dist(N=n_red_on_1, p=transition)
                for red_node in red_dist:
                    n_red_on_0_prime = n_red_on_0 + red_node.n_list[0]
                    value -= blue_node.prob * red_node.prob * (n_red_on_0_prime / n_red)
            J_N_Opt_blue_list.append(value)

        with open(data_path / "simple_example2_{}_deviation.pkl".format(rho), "wb") as f:
            pkl.dump([[mu0, nu0], coord_game_value, n_blue_list, J_N_Opt_list, J_N_Opt_blue_list], f)

    if PLOT_DEVIATE:
        with open(data_path / "simple_example2_{}_deviation.pkl".format(rho), "rb") as f:
            data = pkl.load(f)

        mu0, nu0 = data[0]
        coord_game_value = data[1]
        n_blue_list = np.array(data[2])
        J_N_Opt_list, J_N_Opt_blue_list = data[3], data[4]
        # Lipschitz constant for the value function 5 is the ratio in the max function within the dynamics
        L_J = 4 * np.sqrt(2 * 5)

        fig, ax = plt.subplots(figsize=(6.8, 4.2))
        ax.set_facecolor('white')
        # plt.loglog(np.array(n_blue_list),  coord_game_value - np.array(J_N_Opt_list), "ko-",
        #            label="$J^{N \star} - J^{\\rho \star}_{\mathrm{cor}}$")
        plt.loglog(np.array(n_blue_list),   np.array(J_N_Opt_list) - np.array(J_N_Opt_blue_list), "ro-",
                   label="$J^{N \star} - \min_{\phi^{N_1}} J^{N, \phi^{N_1}, \\beta^*}$")
        plt.loglog(np.array(n_blue_list), L_J / np.sqrt(n_blue_list * rho), 'k--')
        ax.set_ylabel("Difference")
        ax.set_xlabel("N Blue agents")
        plt.savefig(figure_path / 'simple_example2_deviate.svg', format='svg', dpi=800)

        plt.show()
    print("done!")
