import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import CSS4_COLORS as mcolors
import os

from simple_example.examples import SimpleExample1
from simple_example.solver import Solver, linear_approximation_2d
from simple_example.visualizers import visualize_value, get_intersection, visualize_blue_Rset, visualize_red_Rset, \
    visualize_Rset, add_point
from utils import ROOT_PATH

if __name__ == "__main__":
    SOLVE_COR = False
    PLOT_VALUE = True
    PLOT_RSet = False

    res = 200
    rho = 0.6

    # visualize point optimization
    p = 0.96
    q = 0.04

    # save directories
    data_path = ROOT_PATH / "test_data"
    figure_path = ROOT_PATH / "figures"

    # set up save directory
    if not os.path.exists(data_path):
        os.makedirs(data_path)
    if not os.path.exists(figure_path):
        os.makedirs(figure_path)

    example = SimpleExample1(rho=rho)
    solver = Solver(game=example, blue_resolution_list=[[res, res, res]], red_resolution_list=[[res, res, res]])

    # solve the game
    if SOLVE_COR:
        solver.solve()
        solver.save(data_path)

    # load the game solution
    data = solver.load(data_path)

    if PLOT_VALUE:
        ############################# value at t = 0 #############################
        ax0 = visualize_value(value=data["maxmin_value"][0],
                              blue_mesh=data["mesh_lists"][0][0][0], red_mesh=data["mesh_lists"][1][0][0],
                              elev=30, azim=44, roll=0)
        # visualize_value(value=data["minmax_value"][0], blue_mesh=data["mesh_lists"][0][0],
        #                 red_mesh=data["mesh_lists"][1][0], ax=ax0)

        maxmin_z = linear_approximation_2d(p=p, q=q, mesh_p=data["mesh_lists"][0][0], mesh_q=data["mesh_lists"][1][0],
                                           surf=data["maxmin_value"][0])
        minmax_z = linear_approximation_2d(p=p, q=q, mesh_p=data["mesh_lists"][0][0], mesh_q=data["mesh_lists"][1][0],
                                           surf=data["minmax_value"][0])
        # plot point of interest
        # add_point(ax=ax0, x=p, y=q, z=maxmin_z, radius=0.02, color=mcolors['lime'])
        # add_point(ax=ax0, x=p, y=q, z=minmax_z, radius=0.02, color=mcolors['yellow'])

        ax0.set_xlabel("$\mu^\\rho_0(x^1)$")
        ax0.set_ylabel("$\\nu^\\rho_0(y^1)$")
        ax0.set_zlabel("$J^{\\rho \star}_{\mathrm{cor}, 0}$", rotation=0)

        plt.show()

        plt.savefig(figure_path / 'simple_example1_J0.svg', format='svg', dpi=800)

        ############################# value at t = 1 #############################
        value_1 = data["maxmin_value"][1]
        mesh_p1, mesh_q1 = data["mesh_lists"][0][1], data["mesh_lists"][1][1]

        ax1 = visualize_value(value=value_1,
                              blue_mesh=mesh_p1, red_mesh=mesh_q1, elev=30, azim=44, roll=0)

        # plot reachable set box
        RSet_blue_verts = example.generate_blue_Rset(mu=[p, 1 - p], nu=[q, 1 - q], t=1)
        RSet_red_verts = example.generate_red_Rset(mu=[p, 1 - p], nu=[q, 1 - q], t=1)
        h0 = np.min(value_1)
        h1 = 0.7 * np.min(value_1) + 0.3 * np.max(value_1)
        ax1.plot([RSet_blue_verts[0], RSet_blue_verts[1]], [RSet_red_verts[1], RSet_red_verts[1]],
                 [h0, h0], mcolors['blue'], alpha=0.5)
        ax1.plot([RSet_blue_verts[0], RSet_blue_verts[1]], [RSet_red_verts[0], RSet_red_verts[0]],
                 [h0, h0], mcolors['blue'], alpha=0.5)
        ax1.plot([RSet_blue_verts[0], RSet_blue_verts[0]], [RSet_red_verts[0], RSet_red_verts[1]],
                 [h0, h0], mcolors['red'], alpha=0.5)
        ax1.plot([RSet_blue_verts[1], RSet_blue_verts[1]], [RSet_red_verts[0], RSet_red_verts[1]],
                 [h0, h0], mcolors['red'], alpha=0.5)

        # plot reachable set walls
        cutX, cutY = np.meshgrid([RSet_blue_verts[0]], [RSet_red_verts[0], RSet_red_verts[1]])
        cut_surf_p1 = ax1.plot_surface(cutX, cutY,
                                       np.array([[h0, h1],
                                                 [h0, h1]]),
                                       color='r', alpha=0.1)
        cutX, cutY = np.meshgrid([RSet_blue_verts[1]], [RSet_red_verts[0], RSet_red_verts[1]])
        cut_surf_p2 = ax1.plot_surface(cutX, cutY,
                                       np.array([[h0, h1],
                                                 [h0, h1]]),
                                       color='r', alpha=0.1)
        cutX, cutY = np.meshgrid([RSet_blue_verts[0], RSet_blue_verts[1]], [RSet_red_verts[0]])
        cut_surf_q1 = ax1.plot_surface(cutX, cutY,
                                       np.array([[h0, h0],
                                                 [h1, h1]]),
                                       color='b', alpha=0.1)
        cutX, cutY = np.meshgrid([RSet_blue_verts[0], RSet_blue_verts[1]], [RSet_red_verts[1]])
        cut_surf_q2 = ax1.plot_surface(cutX, cutY,
                                       np.array([[h0, h0],
                                                 [h1, h1]]),
                                       color='b', alpha=0.1)

        # plot intersection between RSet wall and the surface
        p_list = np.linspace(RSet_blue_verts[0], RSet_blue_verts[1], 50)
        q_list = [RSet_red_verts[0] for _ in range(len(p_list))]
        z_list = get_intersection(x_list=p_list, y_list=q_list, surf_list=value_1, mesh_x=mesh_p1, mesh_y=mesh_q1)
        ax1.plot(p_list, q_list, z_list, mcolors['navy'], alpha=0.7, linestyle="dashed")

        p_list = np.linspace(RSet_blue_verts[0], RSet_blue_verts[1], 50)
        q_list = [RSet_red_verts[1] for _ in range(len(p_list))]
        z_list = get_intersection(x_list=p_list, y_list=q_list, surf_list=value_1, mesh_x=mesh_p1, mesh_y=mesh_q1)
        ax1.plot(p_list, q_list, z_list, mcolors['navy'], alpha=0.7, linestyle="dashed")

        q_list = np.linspace(RSet_red_verts[0], RSet_red_verts[1], 50)
        p_list = [RSet_blue_verts[0] for _ in range(len(q_list))]
        z_list = get_intersection(x_list=p_list, y_list=q_list, surf_list=value_1, mesh_x=mesh_p1, mesh_y=mesh_q1)
        ax1.plot(p_list, q_list, z_list, mcolors['navy'], alpha=0.7, linestyle="dashed")

        q_list = np.linspace(RSet_red_verts[0], RSet_red_verts[1], 50)
        p_list = [RSet_blue_verts[1] for _ in range(len(q_list))]
        z_list = get_intersection(x_list=p_list, y_list=q_list, surf_list=value_1, mesh_x=mesh_p1, mesh_y=mesh_q1)
        ax1.plot(p_list, q_list, z_list, mcolors['navy'], alpha=0.7, linestyle="dashed")

        # set labels
        ax1.set_xlabel("$\mu^\\rho_1(x^1)$")
        ax1.set_ylabel("$\\nu^\\rho_1(y^1)$")
        ax1.set_zlabel("$J^{\\rho *}_{\mathrm{cor},1}$", rotation=0)
        plt.savefig(figure_path / 'simple_example1_J1.svg', format='svg', dpi=800)

        ############################# value at t = 2 #############################
        value_2 = data["maxmin_value"][2]
        mesh_p2, mesh_q2 = data["mesh_lists"][0][2], data["mesh_lists"][1][2]
        ax2 = visualize_value(value=value_2,
                              blue_mesh=mesh_p2, red_mesh=mesh_q2, elev=30, azim=44, roll=0)
        ax2.set_xlabel("$\mu^\\rho_2(x^1)$")
        ax2.set_ylabel("$\\nu^\\rho_2(y^1)$")
        ax2.set_zlabel("$J^{\\rho *}_{\mathrm{cor},2}$", rotation=0)
        plt.savefig(figure_path / 'simple_example1_J2.svg', format='svg', dpi=800)

        ############################# visualize local value function at t=0 #############################
        local_mesh_p, local_mesh_q, value_1_local = solver.construct_value_matrix(blue_RSet_list=RSet_blue_verts,
                                                                                  red_RSet_list=RSet_red_verts,
                                                                                  blue_mesh_prime=mesh_p1,
                                                                                  red_mesh_prime=mesh_q1,
                                                                                  value_prime=value_1, t=0)
        ax3 = visualize_value(value=value_1_local,
                              blue_mesh=local_mesh_p, red_mesh=local_mesh_q, elev=40, azim=50, roll=0, zorder=3.1)
        ax3.computed_zorder = False

        # plot point of interest
        maxmin_index = np.where(value_1_local == maxmin_z)
        minmax_index = np.where(value_1_local == minmax_z)
        maxmin_x, maxmin_y = local_mesh_p[maxmin_index[0][0]], local_mesh_q[maxmin_index[1][0]]
        minmax_x, minmax_y = local_mesh_p[minmax_index[0][0]], local_mesh_q[minmax_index[1][0]]

        add_point(ax=ax3, x=maxmin_x, y=maxmin_y, z=maxmin_z, radius=0.01, color=mcolors['lime'])
        add_point(ax=ax3, x=minmax_x, y=minmax_y, z=minmax_z, radius=0.01, color=mcolors['yellow'])

        ax3.plot(local_mesh_p, [np.min(local_mesh_q) - 0.05 for _ in range(len(local_mesh_p))],
                 np.min(value_1_local, axis=1), color='k', alpha=0.8)
        add_point(ax=ax3, x=maxmin_x, y=np.min(local_mesh_q) - 0.05, z=maxmin_z, radius=0.01, color=mcolors['lime'],
                  alpha=1.0, zorder=4.1)
        ax3.plot([maxmin_x, maxmin_x], [np.min(local_mesh_q) - 0.05, maxmin_y], [maxmin_z, maxmin_z], 'g--', alpha=0.3)
        ax3.plot([np.min(local_mesh_p) - 0.05 for _ in range(len(local_mesh_q))], local_mesh_q,
                 np.max(value_1_local, axis=0), color='k', alpha=0.8)
        add_point(ax=ax3, x=np.min(local_mesh_p) - 0.05, y=minmax_y, z=minmax_z, radius=0.01, color=mcolors['yellow'],
                  alpha=1.0, zorder=4.1)
        ax3.plot([np.min(local_mesh_p) - 0.05, minmax_x], [minmax_y, minmax_y], [minmax_z, minmax_z], 'y--', alpha=0.3)

        # plot intersections
        p_list = np.linspace(RSet_blue_verts[0], RSet_blue_verts[1], 50)
        q_list = [RSet_red_verts[0] for _ in range(len(p_list))]
        z_list = get_intersection(x_list=p_list, y_list=q_list, surf_list=value_1, mesh_x=mesh_p1, mesh_y=mesh_q1)
        ax3.plot(p_list, q_list, z_list, mcolors['navy'], alpha=0.3, linestyle="dashed", zorder=3.1)

        p_list = np.linspace(RSet_blue_verts[0], RSet_blue_verts[1], 50)
        q_list = [RSet_red_verts[1] for _ in range(len(p_list))]
        z_list = get_intersection(x_list=p_list, y_list=q_list, surf_list=value_1, mesh_x=mesh_p1, mesh_y=mesh_q1)
        ax3.plot(p_list, q_list, z_list, mcolors['navy'], alpha=0.3, linestyle="dashed", zorder=3.1)

        q_list = np.linspace(RSet_red_verts[0], RSet_red_verts[1], 50)
        p_list = [RSet_blue_verts[0] for _ in range(len(q_list))]
        z_list = get_intersection(x_list=p_list, y_list=q_list, surf_list=value_1, mesh_x=mesh_p1, mesh_y=mesh_q1)
        ax3.plot(p_list, q_list, z_list, mcolors['navy'], alpha=0.3, linestyle="dashed", zorder=3.1)

        q_list = np.linspace(RSet_red_verts[0], RSet_red_verts[1], 50)
        p_list = [RSet_blue_verts[1] for _ in range(len(q_list))]
        z_list = get_intersection(x_list=p_list, y_list=q_list, surf_list=value_1, mesh_x=mesh_p1, mesh_y=mesh_q1)
        ax3.plot(p_list, q_list, z_list, mcolors['navy'], alpha=0.3, linestyle="dashed", zorder=3.1)


        ax3.plot([RSet_blue_verts[0], RSet_blue_verts[1]], [RSet_red_verts[1], RSet_red_verts[1]],
                 [h0, h0], mcolors['blue'], alpha=0.5, zorder=2.1)
        ax3.plot([RSet_blue_verts[0], RSet_blue_verts[1]], [RSet_red_verts[0], RSet_red_verts[0]],
                 [h0, h0], mcolors['blue'], alpha=0.5, zorder=2.1)
        ax3.plot([RSet_blue_verts[0], RSet_blue_verts[0]], [RSet_red_verts[0], RSet_red_verts[1]],
                 [h0, h0], mcolors['red'], alpha=0.5, zorder=2.1)
        ax3.plot([RSet_blue_verts[1], RSet_blue_verts[1]], [RSet_red_verts[0], RSet_red_verts[1]],
                 [h0, h0], mcolors['red'], alpha=0.5, zorder=2.1)

        ax3.zaxis.set_rotate_label(False)
        ax3.set_xlabel("$\mu^\\rho_1(x^1)$")
        ax3.set_ylabel("$\\nu^\\rho_1(y^1)$")
        ax3.set_zlabel("$J^{\\rho *}_{\mathrm{cor},1}$", rotation=0)
        plt.savefig(figure_path / 'simple_example1_local_value.svg', format='svg', dpi=800)
        plt.show()

    if PLOT_RSet:
        visualize_blue_Rset(game=example, mu=[p, 1 - p], nu=[q, 1 - q], t=0, visualize=True, save_path=figure_path)
        visualize_red_Rset(game=example, mu=[p, 1 - p], nu=[q, 1 - q], t=0, visualize=True, save_path=figure_path)
        visualize_Rset(game=example, mu=[p, 1 - p], nu=[q, 1 - q], t=0, offset=0.02,visualize=True, save_path=figure_path)

    print("done!")
