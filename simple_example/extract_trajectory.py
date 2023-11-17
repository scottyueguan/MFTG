import pickle as pkl
import numpy as np


def project_mesh_point(point, mesh):
    diff = 1e5
    best_point, best_index = None, None
    for i, x in enumerate(mesh):
        if len(x) < len(point):
            assert len(x) == len(point) - 1
            x = list(x)
            x.append(1-sum(x))
        x = np.array(x)
        diff_ = np.linalg.norm(x - point)
        if diff_ < diff:
            best_point = x
            best_index = i
            diff = diff_
    return best_point, best_index


if __name__ == "__main__":
    policy_class = "maxmin_policies"
    resolution = 500
    mu_0, nu_0 = [[0., 1.0], [0.5, 0.5]], [[1., 0.]]

    config_list = [[mu_0, nu_0]]

    with open("../test_data/TwoNodePerimeter_10.pkl", "rb") as f:
        data = pkl.load(f)

        game = data["game"]
        blue_mesh, red_mesh = data["mesh_lists"]
        blue_index_map, red_index_map = data["index_maps"]
        blue_policy, red_policy = data[policy_class]

    T = len(blue_mesh[0])

    for t in range(0, T-1):
        mu_t_list = config_list[t][0]
        nu_t_list = config_list[t][1]

        mu_index_list, nu_index_list = [], []
        for mu_t, mesh in zip(mu_t_list, blue_mesh):
            _, mu_index = project_mesh_point(point=mu_t, mesh=mesh[t])
            mu_index_list.append(mu_index)

        for nu_t, mesh in zip(nu_t_list, red_mesh):
            _, nu_index = project_mesh_point(point=nu_t, mesh=mesh[t])
            nu_index_list.append(nu_index)

        blue_index = blue_index_map[t][tuple(mu_index_list)]
        red_index = red_index_map[t][tuple(nu_index_list)]

        mu_new_list = blue_policy[t][blue_index][red_index]
        nu_new_list = red_policy[t][blue_index][red_index]

        config_list.append([mu_new_list, nu_new_list])


    blue_bar_list, red_bar_list = [[] for _ in range(game.n_blue_types)], [[] for _ in range(game.n_red_types)]
    for config in config_list:
        mu_list, nu_list = config[0], config[1]
        for i, mu in enumerate(mu_list):
            blue_bar_list[i].append([])
            for p in mu:
                blue_bar_list[i][-1].append(200 * p * game.blue_rho_list[i])

        for j, nu in enumerate(nu_list):
            red_bar_list[j].append([])
            for p in nu:
                red_bar_list[j][-1].append(200 * p * game.red_rho_list[j])






    print(config_list)

    print(blue_bar_list)

    print(red_bar_list)

    print("done!")
