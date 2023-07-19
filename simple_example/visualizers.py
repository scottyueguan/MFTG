from simple_example.examples import MFTG
from simple_example.solver import linear_approximation_2d
import matplotlib.pyplot as plt
from matplotlib.colors import CSS4_COLORS as mcolors
from matplotlib import cm
import numpy as np


def visualize_blue_Rset(game: MFTG, mu, nu, t, visualize=False):
    p_vertices = game.generate_blue_Rset(mu=mu, nu=nu, t=t)
    plt.figure()
    plt.plot(np.linspace(0, 1, 50), 1 - np.linspace(0, 1, 50), 'k')
    plt.plot(mu[0], mu[1], 'bo', markersize=7)
    plt.plot(p_vertices, 1 - p_vertices, 'b', linewidth=3)
    plt.legend(["Simplex", "Initial Distribution", "Reachable Set"])
    plt.xlabel("$\mu(x^1)$")
    plt.ylabel("$\mu(x^2)$")
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.gca().set_aspect("equal")
    plt.savefig('figures/simple_example_Blue_RSet.svg', format='svg', dpi=800)

    if visualize:
        plt.show()


def visualize_red_Rset(game: MFTG, mu, nu, t, visualize=False):
    q_vertices = game.generate_red_Rset(mu=mu, nu=nu, t=t)
    plt.figure()
    plt.plot(np.linspace(0, 1, 50), 1 - np.linspace(0, 1, 50), 'k')
    plt.plot(nu[0], nu[1], 'ro', markersize=7)
    plt.plot(q_vertices, 1 - q_vertices, mcolors['red'], linewidth=3)
    plt.legend(["Simplex", "Initial Distribution", "Reachable Set"])
    plt.xlabel("$\\nu(y^1)$")
    plt.ylabel("$\\nu(y^2)$")

    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.gca().set_aspect("equal")
    plt.savefig('figures/simple_example_Red_RSet.svg', format='svg', dpi=800)

    if visualize:
        plt.show()


def visualize_value(value, blue_mesh, red_mesh, elev=30, azim=44, roll=0, zorder=None, ax=None):
    if ax is None:
        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
        ax.view_init(elev=elev, azim=azim, roll=roll)
        ax.computed_zorder = False
        ax.set_facecolor('white')
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
    X, Y = np.meshgrid(blue_mesh, red_mesh)
    surf = ax.plot_surface(X, Y, value.transpose(), cmap=cm.coolwarm,
                           linewidth=0, antialiased=False, zorder=zorder)
    ax.zaxis.set_rotate_label(False)

    return ax


def get_intersection(x_list, y_list, surf_list, mesh_x, mesh_y):
    z_list = []
    for x, y in zip(x_list, y_list):
        z = linear_approximation_2d(p=x, q=y, mesh_p=mesh_x, mesh_q=mesh_y, surf=surf_list)
        z_list.append(z)
    return z_list


def add_point(ax, x, y, z, radius=0.005, color='r', alpha=1.0, zorder=5.0):
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)

    axis_length = [x[1] - x[0] for x in [ax.get_xbound(), ax.get_ybound(), ax.get_zbound()]]
    ratio = [length / max(axis_length) for length in axis_length]

    x_ = ratio[0] * radius * np.outer(np.cos(u), np.sin(v)) + x
    y_ = ratio[1] * radius * np.outer(np.sin(u), np.sin(v)) + y
    z_ = ratio[2] * radius * np.outer(np.ones(np.size(u)), np.cos(v)) + z
    ax.plot_surface(x_, y_, z_, color=color, alpha=alpha, zorder=zorder)
