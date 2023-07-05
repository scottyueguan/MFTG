import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import pickle as pkl
from mpl_toolkits.mplot3d import Axes3D, art3d
from matplotlib.patches import Circle, Ellipse
from matplotlib.colors import CSS4_COLORS as mcolors

"""
Two state systems, each state has two actions: 
action-0 intends to stay at the same state
action-1 intends to move to the other state
"""

extreme_policy_list = [np.array([[0.5 + p1, 0.5 - p1], [0.5 + p2, 0.5 - p2]]) for p1 in [-0.5, 0.5]
                       for p2 in [-0.5, 0.5]]


def blue_dynamics(x, a, x_prime, mu, nu, rho):
    p = 0.0
    if a == 0:
        if x == x_prime:
            p = 0.5 * (1 + (rho * mu[x] - (1 - rho) * nu[x]))
        else:
            p = 0.5 * (1 - (rho * mu[x] - (1 - rho) * nu[x]))
    else:
        if x == x_prime:
            p = 0.5 * (1 - 0.3 * (rho * mu[x] - (1 - rho) * nu[x]))
        else:
            p = 0.5 * (1 + 0.3 * (rho * mu[x] - (1 - rho) * nu[x]))

    return p


def blue_transition_matrix(policy, mu, nu, rho):
    F = np.zeros((2, 2))
    for x in range(2):
        for a in range(2):
            prob_a = policy[x, a]
            for x_prime in range(2):
                F[x, x_prime] += blue_dynamics(x, a, x_prime, mu, nu, rho) * prob_a
    assert (abs(np.sum(F, axis=1) - 1.0) < 1e-5).all()
    return F


def red_transition_matrix(policy, mu, nu, rho):
    mu_, nu_, rho_ = nu, mu, 1 - rho
    G = blue_transition_matrix(policy, mu_, nu_, rho_)
    return G


def generate_blue_Rset(mu, nu, rho):
    p_min, p_max = 2, -2
    for extreme_policy in extreme_policy_list:
        F = blue_transition_matrix(policy=extreme_policy, mu=mu, nu=nu, rho=rho)
        mu_prime = np.matmul(mu, F)
        if mu_prime[0] < p_min:
            p_min = mu_prime[0]
        if mu_prime[0] > p_max:
            p_max = mu_prime[0]
    return np.array([p_min, p_max])


def generate_red_Rset(mu, nu, rho):
    mu_, nu_, rho_ = nu, mu, 1 - rho
    return generate_blue_Rset(mu_, nu_, rho_)


def visualize_blue_Rset(mu, nu, rho):
    p_vertices = generate_blue_Rset(mu=mu, nu=nu, rho=rho)
    plt.figure()
    plt.plot(np.linspace(0, 1, 50), 1 - np.linspace(0, 1, 50), 'k')
    plt.plot(mu[0], mu[1], 'bo', markersize=15)
    plt.plot(p_vertices, 1 - p_vertices, 'b', linewidth=5)

    plt.xlabel("$\mu(x^1)$")
    plt.ylabel("$\mu(x^2)$")
    plt.show()


def visualize_red_Rset(mu, nu, rho):
    q_vertices = generate_red_Rset(mu=mu, nu=nu, rho=rho)
    plt.figure()
    plt.plot(np.linspace(0, 1, 50), 1 - np.linspace(0, 1, 50), 'k')
    plt.plot(nu[0], nu[1], 'ro', markersize=15)
    plt.plot(q_vertices, 1 - q_vertices, mcolors['red'], linewidth=5)

    plt.xlabel("$\\nu(x^1)$")
    plt.ylabel("$\\nu(x^2)$")
    plt.show()


def reward(mu, nu, rho):
    return mu[2]


def linear_approximation(point, mesh, v_list_list):
    start_index = np.where(mesh - point > 0)[0][0]
    coeff = (point - mesh[start_index - 1]) / (mesh[start_index] - mesh[start_index - 1])
    assert (0 <= coeff < 1)
    v_list = (1 - coeff) * v_list_list[start_index - 1, :] + coeff * v_list_list[start_index, :]

    return start_index, v_list


def linear_approximation_2d(p, q, mesh_p, mesh_q, surf):
    _, v_list = linear_approximation(point=p, mesh=mesh_p, v_list_list=surf)
    _, v_list = linear_approximation(point=q, mesh=mesh_q, v_list_list=np.expand_dims(v_list, axis=0).transpose())
    return v_list[0]


def max_min(value_matrix):
    """
    :param value_matrix: row player maximize, column player minimize
    :return: value
    """
    f_min = np.min(value_matrix, axis=1)
    opt_value = np.max(f_min)
    return opt_value


def compute_v1(res_p1: int, res_q1: int, rho):
    mesh_p = np.linspace(0, 1, res_p1 + 1)
    mesh_q = np.linspace(0, 1, res_q1 + 1)

    value_1 = np.zeros((res_p1 + 1, res_q1 + 1))

    for i, p in enumerate(mesh_p):
        for j, q in enumerate(mesh_q):
            RSet_verts = generate_blue_Rset(mu=[p, 1 - p], nu=[q, 1 - q], rho=rho)
            value_1[i, j] = 1 - RSet_verts[0]

    return mesh_p, mesh_q, value_1


def construct_value_matrix(p_start, p_end, q_start, q_end, mesh_p, mesh_q, original_value_matrix):
    # concstruct mesh for the RSets
    blue_start_index, blue_start_row = linear_approximation(point=p_start, mesh=mesh_p,
                                                            v_list_list=original_value_matrix)

    blue_end_index, blue_end_row = linear_approximation(point=p_end, mesh=mesh_p,
                                                        v_list_list=original_value_matrix)

    mesh_p_tmp = [p_start] + list(mesh_p)[blue_start_index: blue_end_index] + [p_end]
    value_temp = np.array([blue_start_row] + list(value_1)[blue_start_index:blue_end_index] + [blue_end_row])

    red_start_index, red_start_row = linear_approximation(point=q_start, mesh=mesh_q,
                                                          v_list_list=value_temp.transpose())
    red_end_index, red_end_row = linear_approximation(point=q_end, mesh=mesh_q,
                                                      v_list_list=value_temp.transpose())

    mesh_q_tmp = [q_start] + list(mesh_q)[red_start_index: red_end_index] + [q_end]
    value_temp = np.array([red_start_row] + list(value_temp.transpose())[red_start_index:red_end_index] + [
        red_end_row]).transpose()

    assert value_temp.shape[0] == len(mesh_p_tmp) and value_temp.shape[1] == len(mesh_q_tmp)
    return mesh_p_tmp, mesh_q_tmp, value_temp


def compute_v0(value_info_1: list, res_p0: int, res_q0: int, rho):
    mesh_p1, mesh_q1, value_1 = value_info_1
    mesh_p = np.linspace(0, 1, res_p0 + 1)
    mesh_q = np.linspace(0, 1, res_q0 + 1)

    maxmin_value_0 = np.zeros((res_p0 + 1, res_q0 + 1))
    minmax_value_0 = np.zeros((res_p0 + 1, res_q0 + 1))

    for i, p in enumerate(mesh_p):
        for j, q in enumerate(mesh_q):
            RSet_blue_verts = generate_blue_Rset(mu=[p, 1 - p], nu=[q, 1 - q], rho=rho)
            RSet_red_verts = generate_red_Rset(mu=[p, 1 - p], nu=[q, 1 - q], rho=rho)

            # concstruct mesh for the RSets
            _, _, value_1_temp = construct_value_matrix(p_start=RSet_blue_verts[0], p_end=RSet_blue_verts[1],
                                                        q_start=RSet_red_verts[0], q_end=RSet_red_verts[1],
                                                        mesh_p=mesh_p1, mesh_q=mesh_q1,
                                                        original_value_matrix=value_1)

            maxmin_value_0[i, j] = max_min(value_matrix=value_1_temp)
            minmax_value_0[i, j] = -max_min(value_matrix=-value_1_temp.transpose())

            diff = abs(minmax_value_0[i, j] - maxmin_value_0[i, j])
            # if diff > 1e-3:
            #     print("value difference {}".format(diff))

    return mesh_p, mesh_q, maxmin_value_0, minmax_value_0


# def add_point(ax, x, y, z, fc = None, ec = None, radius = 0.005):
#    xy_len, z_len = ax.get_figure().get_size_inches()
#    axis_length = [x[1] - x[0] for x in [ax.get_xbound(), ax.get_ybound(), ax.get_zbound()]]
#    axis_rotation =  {'z': ((x, y, z), axis_length[1]/axis_length[0]),
#                      mcolors['yellow']: ((x, z, y), axis_length[2]/axis_length[0]*xy_len/z_len),
#                      'x': ((y, z, x), axis_length[2]/axis_length[1]*xy_len/z_len)}
#    for a, ((x0, y0, z0), ratio) in axis_rotation.items():
#        p = Ellipse((x0, y0), width = radius, height = radius*ratio, fc=fc, ec=ec)
#        ax.add_patch(p)
#        art3d.pathpatch_2d_to_3d(p, z=z0, zdir=a)

def add_point(ax, x, y, z, radius=0.005, color='r', alpha=1.0):
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)

    axis_length = [x[1] - x[0] for x in [ax.get_xbound(), ax.get_ybound(), ax.get_zbound()]]
    ratio = [length / max(axis_length) for length in axis_length]

    x_ = ratio[0] * radius * np.outer(np.cos(u), np.sin(v)) + x
    y_ = ratio[1] * radius * np.outer(np.sin(u), np.sin(v)) + y
    z_ = ratio[2] * radius * np.outer(np.ones(np.size(u)), np.cos(v)) + z
    ax.plot_surface(x_, y_, z_, color=color, alpha=alpha)


if __name__ == "__main__":
    resolution = 500
    rho = 0.6

    # Discretize and solve
    # X1, Y1, value_1 = compute_v1(res_p1=resolution, res_q1=resolution, rho=rho)
    # X0, Y0, maxmin_value, minmax_value = compute_v0(value_info_1=[X1, Y1, value_1],
    #                                                 res_p0=resolution, res_q0=resolution, rho=rho)
    # data = [[X1, Y1, value_1], [X0, Y0, maxmin_value, minmax_value]]
    # with open("data/simple_example_{}_{}.pkl".format(rho, resolution), "wb") as f:
    #     pkl.dump(data, f)

    # Load data
    with open("data/simple_example_{}_{}.pkl".format(rho, resolution), "rb") as f:
        data = pkl.load(f)
        X1, Y1, value_1 = data[0]
        X0, Y0, maxmin_value, minmax_value = data[1]

    # visualize point optimization
    p = 0.96
    q = 0.04

    RSet_blue_verts = generate_blue_Rset(mu=[p, 1 - p], nu=[q, 1 - q], rho=rho)
    RSet_red_verts = generate_red_Rset(mu=[p, 1 - p], nu=[q, 1 - q], rho=rho)

    local_mesh_p, local_mesh_q, value_1_local = construct_value_matrix(p_start=RSet_blue_verts[0],
                                                                       p_end=RSet_blue_verts[1],
                                                                       q_start=RSet_red_verts[0],
                                                                       q_end=RSet_red_verts[1],
                                                                       mesh_p=X1, mesh_q=Y1,
                                                                       original_value_matrix=value_1)

    # Visualize value function t=1
    fig1, ax1 = plt.subplots(subplot_kw={"projection": "3d"})
    ax1.view_init(elev=30, azim=44, roll=0)
    ax1.computed_zorder = False
    ax1.set_facecolor('white')
    ax1.xaxis.pane.fill = False
    ax1.yaxis.pane.fill = False
    ax1.zaxis.pane.fill = False
    X, Y = np.meshgrid(X1, Y1)
    surf_maxmin = ax1.plot_surface(X, Y, value_1.transpose(), cmap=cm.coolwarm,
                                   linewidth=0, antialiased=False)

    h0 = np.min(value_1)
    h1 = 0.7 * np.min(value_1) + 0.3 * np.max(value_1)
    cutX, cutY = np.meshgrid([RSet_blue_verts[0]], [RSet_red_verts[0], RSet_red_verts[1]])
    cut_surf_p1 = ax1.plot_surface(cutX, cutY,
                                   np.array([[h0, h1],
                                             [h0, h1]]),
                                   color='r', alpha=0.3)
    cutX, cutY = np.meshgrid([RSet_blue_verts[1]], [RSet_red_verts[0], RSet_red_verts[1]])
    cut_surf_p2 = ax1.plot_surface(cutX, cutY,
                                   np.array([[h0, h1],
                                             [h0, h1]]),
                                   color='r', alpha=0.3)
    cutX, cutY = np.meshgrid([RSet_blue_verts[0], RSet_blue_verts[1]], [RSet_red_verts[0]])
    cut_surf_q1 = ax1.plot_surface(cutX, cutY,
                                   np.array([[h0, h0],
                                             [h1, h1]]),
                                   color='r', alpha=0.3)
    cutX, cutY = np.meshgrid([RSet_blue_verts[0], RSet_blue_verts[1]], [RSet_red_verts[1]])
    cut_surf_q1 = ax1.plot_surface(cutX, cutY,
                                   np.array([[h0, h0],
                                             [h1, h1]]),
                                   color='r', alpha=0.3)
    ax1.plot([RSet_blue_verts[0], RSet_blue_verts[1], RSet_blue_verts[1], RSet_blue_verts[0], RSet_blue_verts[0]],
             [RSet_red_verts[1], RSet_red_verts[1], RSet_red_verts[0], RSet_red_verts[0], RSet_red_verts[1]],
             [h0, h0, h0, h0, h0], mcolors['red'], alpha=0.7)

    ax1.set_xlabel("$\mu_1(x^1)$")
    ax1.set_ylabel("$\\nu_1(y^1)$")
    ax1.zaxis.set_rotate_label(False)
    ax1.set_zlabel("$J^{\\rho *}_1$", rotation=0)
    plt.savefig('figures/simple_example_J1.svg', format='svg', dpi=800)

    # Visualize value function t=0
    fig0, ax0 = plt.subplots(subplot_kw={"projection": "3d"})
    ax0.view_init(elev=30, azim=44, roll=0)
    ax0.computed_zorder = False
    ax0.set_facecolor('white')
    ax0.xaxis.pane.fill = False
    ax0.yaxis.pane.fill = False
    ax0.zaxis.pane.fill = False
    X, Y = np.meshgrid(X0, Y0)
    surf_maxmin = ax0.plot_surface(X, Y, maxmin_value.transpose(), cmap=cm.coolwarm,
                                   linewidth=0, antialiased=False)
    surf_minmax = ax0.plot_surface(X, Y, minmax_value.transpose(), cmap=cm.coolwarm,
                                   linewidth=0, antialiased=False)

    # plot point of interest
    maxmin_z = linear_approximation_2d(p=p, q=q, mesh_p=X0, mesh_q=Y0, surf=maxmin_value)
    minmax_z = linear_approximation_2d(p=p, q=q, mesh_p=X0, mesh_q=Y0, surf=minmax_value)

    add_point(ax=ax0, x=p, y=q, z=maxmin_z, radius=0.02, color=mcolors['lime'])
    add_point(ax=ax0, x=p, y=q, z=minmax_z, radius=0.02, color=mcolors['yellow'])
    # ax0.plot([p, p], [q, q], [maxmin_z, minmax_z], 'b--')

    ax0.set_xlabel("$\mu_0(x^1)$")
    ax0.set_ylabel("$\\nu_0(y^1)$")
    ax0.zaxis.set_rotate_label(False)
    ax0.set_zlabel("$J^{\\rho *}_0$", rotation=0)
    plt.savefig('figures/simple_example_J0.svg', format='svg', dpi=800)

    # Visualize local value function t=0
    fig2, ax2 = plt.subplots(subplot_kw={"projection": "3d"})
    ax2.view_init(elev=35, azim=65, roll=0)
    ax2.computed_zorder = False
    ax2.set_facecolor('white')
    ax2.xaxis.pane.fill = False
    ax2.yaxis.pane.fill = False
    ax2.zaxis.pane.fill = False
    X, Y = np.meshgrid(local_mesh_p, local_mesh_q)
    surf_local = ax2.plot_surface(X, Y, value_1_local.transpose(), cmap=cm.coolwarm,
                                  linewidth=0, antialiased=False)

    # plot point of interest
    maxmin_index = np.where(value_1_local == maxmin_z)
    minmax_index = np.where(value_1_local == minmax_z)
    maxmin_x, maxmin_y = local_mesh_p[maxmin_index[0][0]], local_mesh_q[maxmin_index[1][0]]
    minmax_x, minmax_y = local_mesh_p[minmax_index[0][0]], local_mesh_q[minmax_index[1][0]]

    add_point(ax=ax2, x=maxmin_x, y=maxmin_y, z=maxmin_z, radius=0.01, color=mcolors['lime'])
    add_point(ax=ax2, x=minmax_x, y=minmax_y, z=minmax_z, radius=0.01, color=mcolors['yellow'])

    ax2.plot(local_mesh_p, [np.min(local_mesh_q) - 0.05 for _ in range(len(local_mesh_p))],
             np.min(value_1_local, axis=1), color='k', alpha=0.4)
    add_point(ax=ax2, x=maxmin_x, y=np.min(local_mesh_q) - 0.05, z=maxmin_z, radius=0.01, color=mcolors['lime'], alpha=1.0)
    ax2.plot([maxmin_x, maxmin_x], [np.min(local_mesh_q) - 0.05, maxmin_y], [maxmin_z, maxmin_z], 'g--', alpha=1.0)
    ax2.plot([np.min(local_mesh_p) - 0.05 for _ in range(len(local_mesh_q))], local_mesh_q,
             np.max(value_1_local, axis=0), color='k', alpha=1.0)
    add_point(ax=ax2, x=np.min(local_mesh_p) - 0.05, y=minmax_y, z=minmax_z, radius=0.01, color=mcolors['yellow'], alpha=1.0)
    ax2.plot([np.min(local_mesh_p) - 0.05, minmax_x], [minmax_y, minmax_y], [minmax_z, minmax_z], 'y--', alpha=0.2)

    ax2.set_xlabel("$\mu_0(x^1)$")
    ax2.set_ylabel("$\\nu_0(y^1)$")
    ax2.zaxis.set_rotate_label(False)
    ax2.set_zlabel("$J^{\\rho *}_1$", rotation=0)
    plt.savefig('figures/simple_example_local_value.svg', format='svg', dpi=800)

    plt.show()

    print("done!")
