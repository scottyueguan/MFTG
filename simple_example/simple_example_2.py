import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import CSS4_COLORS as mcolors
import pickle as pkl

from examples import SimpleExample2
from solver import Solver
from visualizers import visualize_value, get_intersection
from utils import ROOT_PATH

from emp_dist import empirical_dist

if __name__ == "__main__":
    SOLVE_COR = True
    PLOT_VALUE = True
    SOLVE_DEVIATE = True
    PLOT_DEVIATE = True

    res = 200
    rho = 0.625

    # save directories
    data_path = ROOT_PATH / "simple_example/data"
    figure_path = ROOT_PATH / "simple_example/figures"

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
                              blue_mesh=data["mesh_lists"][0][0], red_mesh=data["mesh_lists"][1][0], elev=30, azim=25)
        ax0.set_zlabel("$J^{\\rho \star}_{\mathrm{cor}, 0}$", rotation=0)
        plt.savefig(figure_path / 'simple_example2_J0.svg', format='svg', dpi=800)

        ax1 = visualize_value(value=data["maxmin_value"][1],
                              blue_mesh=data["mesh_lists"][0][1], red_mesh=data["mesh_lists"][1][1], elev=30, azim=25)
        ax1.set_zlabel("$J^{\\rho \star}_{\mathrm{cor}, 1}$", rotation=0)
        p_list, q_list = np.linspace(0.0, 1.0, 51), [1 / np.sqrt(2) for _ in range(51)]
        z_list = get_intersection(x_list=p_list, y_list=q_list, surf_list=data["maxmin_value"][1],
                                  mesh_x=data["mesh_lists"][0][1], mesh_y=data["mesh_lists"][1][1])
        ax1.plot(p_list, q_list, z_list, mcolors['navy'], alpha=0.8, linestyle="dashed")
        plt.savefig(figure_path / 'simple_example2_J1.svg', format='svg', dpi=800)

        ax2 = visualize_value(value=data["maxmin_value"][2],
                              blue_mesh=data["mesh_lists"][0][2], red_mesh=data["mesh_lists"][1][2], elev=30, azim=25)
        ax2.set_zlabel("$J^{\\rho \star}_{\mathrm{cor}, 2}$", rotation=0)
        plt.savefig(figure_path / 'simple_example2_J2.svg', format='svg', dpi=800)

        plt.show()

    # compute finite-population difference
    if SOLVE_DEVIATE:
        mu0 = [0.6, 0.4]
        nu0 = [1.0, 0.0]
        n_blue_list = np.array([5, 10, 20, 50, 65, 80, 90, 100, 125, 150, 175, 200, 225, 250, 275])
        n_list = (n_blue_list / rho).astype(int)
        n_red_list = ((1 - rho) * n_list).astype(int)

        # find coordinator game value
        # blue_index = np.where(abs(data["mesh_lists"][0][0] - mu0[0]) < 1e-5)
        # red_index = np.where(abs(data["mesh_lists"][1][0] - nu0[0]) < 1e-5)
        # coord_game_value = data["maxmin_value"][0][blue_index, red_index]
        coord_game_value = mu0[1]

        # compute finite population value
        red_best_config_list = []
        best_transition_list = []
        J_N_Opt_list = []  # value for both blue identical and non-identical against non-identical red
        J_N_Opt_red_list = []  # value for identical red against non-identical blue
        red_1_config_list = []
        red_identical_policy = [1 / np.sqrt(2), 1 - 1 / np.sqrt(2)]
        for n_red in n_red_list:
            best_q1 = np.floor(n_red / np.sqrt(2)) / n_red
            best_q2 = np.ceil(n_red / np.sqrt(2)) / n_red
            diff1 = (best_q1 - (1 / np.sqrt(2))) ** 2
            diff2 = (best_q2 - (1 / np.sqrt(2))) ** 2
            if diff1 > diff2:
                best_q = best_q2
            else:
                best_q = best_q1
            assert (best_q - np.round(n_red / np.sqrt(2)) / n_red) < 1e-5

            red_best_config_list.append([best_q, 1 - best_q])
            best_transition_matrix = example.blue_transition_matrix(policy=[[0, 1], [1, 0]],
                                                                    mu=mu0, nu=[best_q, 1 - best_q])
            best_transition_list.append(best_transition_matrix[0, :])
            red_1_config_list.append(empirical_dist(N=n_red, p=red_identical_policy))

        for n_blue, transition, red_1_config in zip(n_blue_list, best_transition_list, red_1_config_list):
            n_blue_on_0 = mu0[0] * n_blue
            n_blue_on_1 = n_blue - n_blue_on_0

            # compute blue against non-identical red
            dist = empirical_dist(N=n_blue_on_0, p=transition)
            value = 0
            for node in dist:
                n_blue_on_1_prime = n_blue_on_1 + node.n_list[1]
                value += node.prob * (n_blue_on_1_prime / n_blue)
            J_N_Opt_list.append(value)

            # compute blue against identical red
            value = 0
            for red_node in red_1_config:
                nu1 = red_node.emp_dist
                transition = example.blue_transition_matrix(policy=[[0, 1], [1, 0]],
                                                            mu=mu0, nu=nu1)[0, :]

                blue_dist = empirical_dist(N=n_blue_on_0, p=transition)
                for blue_node in blue_dist:
                    n_blue_on_1_prime = n_blue_on_1 + blue_node.n_list[1]
                    value += red_node.prob * blue_node.prob * (n_blue_on_1_prime / n_blue)
            J_N_Opt_red_list.append(value)

        with open(data_path / "simple_example2_{}_deviation.pkl".format(rho), "wb") as f:
            pkl.dump([[mu0, nu0], coord_game_value, n_blue_list, J_N_Opt_list, J_N_Opt_red_list], f)

    if PLOT_DEVIATE:
        with open(data_path / "simple_example2_{}_deviation.pkl".format(rho), "rb") as f:
            data = pkl.load(f)

        mu0, nu0 = data[0]
        coord_game_value = data[1]
        n_blue_list = np.array(data[2])
        J_N_Opt_list, J_N_Opt_red_list = data[3], data[4]
        # Lipschitz constant for the value function 5 is the ratio in the max function within the dynamics
        L_J = 4 * np.sqrt(2 * 5)

        fig, ax = plt.subplots()
        ax.set_facecolor('white')
        plt.loglog(np.array(n_blue_list), np.array(J_N_Opt_list) - coord_game_value, "ko-",
                   label="$J^{N \star} - J^{\\rho \star}_{\mathrm{cor}}$")
        plt.loglog(np.array(n_blue_list), np.array(J_N_Opt_red_list) - np.array(J_N_Opt_list), "ro-",
                   label="$J^{N \star} - \min_{\phi^{N_1}} J^{N, \phi^{N_1}, \\beta^*}$")
        plt.loglog(np.array(n_blue_list), L_J / np.sqrt(n_blue_list * rho), 'k--')
        ax.set_ylabel("Difference")
        ax.set_xlabel("N Blue agents")
        plt.savefig(figure_path / 'simple_example2_deviate.svg', format='svg', dpi=800)

    print("done!")
