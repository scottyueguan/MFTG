import pickle as pkl
import numpy as np


def project_mesh_point(point, mesh):
    index = np.where(mesh - point >= 0)[0][0]
    return index


if __name__ == "__main__":
    policy_class = "maxmin"
    resolution = 500
    rho = 0.6
    T = 2
    mu_0, nu_0 = [0.96, 0.04], [0.04, 0.96]

    config_list = [[mu_0, nu_0]]

    with open("data/simple_example_{}_{}.pkl".format(rho, resolution), "rb") as f:
        data = pkl.load(f)
        X1, Y1, value_1, maxmin_policies_1, minmax_policies_1 = data[0]
        X0, Y0, maxmin_value_0, minmax_value_0, maxmin_policies_0, minmax_policies_0 = data[1]

    mesh_list = [[X0, Y0], [X1, Y1]]

    if policy_class == "minmax":
        policy_list = [minmax_policies_0, minmax_policies_1]
        value = minmax_value_0
    else:
        policy_list = [maxmin_policies_0, maxmin_policies_1]
        value = maxmin_value_0

    for t in range(0, T):
        mu_t = config_list[t][0]
        nu_t = config_list[t][1]
        mesh_p_t, mesh_q_t = mesh_list[t]
        mu_index = project_mesh_point(point=mu_t[0], mesh=mesh_list[t][0])
        nu_index = project_mesh_point(point=nu_t[0], mesh=mesh_list[t][1])

        mu_new = policy_list[t][0][mu_index][nu_index]
        nu_new = policy_list[t][1][mu_index][nu_index]

        config_list.append([mu_new, nu_new])

    print(config_list)
    print(value[project_mesh_point(point=mu_0[0], mesh=mesh_list[0][0]),
                         project_mesh_point(point=nu_0[0], mesh=mesh_list[0][1])])

    print("done!")
