import numpy as np
from simple_example.examples import MFTG
from typing import List
import pickle as pkl
import itertools
from simple_example.mftg_utils import RSet


def max_min(value_matrix):
    """
    :param value_matrix: row player maximize, column player minimize
    :return: value
    """
    f_min_indices = np.argmin(value_matrix, axis=1)
    f_min = [value_matrix[i, min_index] for i, min_index in enumerate(f_min_indices)]
    argmax_index = np.argmax(f_min)
    argmin_index = f_min_indices[argmax_index]
    opt_value = f_min[argmax_index]
    return opt_value, argmax_index, argmin_index


def linear_approximation(point, mesh, v_list_list):
    start_index = np.where(mesh - point >= 0)[0][0]
    coeff = (point - mesh[start_index - 1]) / (mesh[start_index] - mesh[start_index - 1])
    assert (0 <= coeff <= 1)
    v_list = (1 - coeff) * v_list_list[start_index - 1, :] + coeff * v_list_list[start_index, :]

    return start_index, v_list


def linear_approximation_2d(p, q, mesh_p, mesh_q, surf):
    _, v_list = linear_approximation(point=p, mesh=mesh_p, v_list_list=surf)
    _, v_list = linear_approximation(point=q, mesh=mesh_q, v_list_list=np.expand_dims(v_list, axis=0).transpose())
    return v_list[0]


class Solver:
    def __init__(self, game: MFTG, blue_resolution_list: List[List[int]], red_resolution_list: List[List[int]],
                 solve_red=False):
        self.game = game
        self.blue_resolution_list = blue_resolution_list
        self.red_resolution_list = red_resolution_list
        self.Tf = self.game.Tf
        assert len(blue_resolution_list[0]) == len(red_resolution_list[0]) == self.Tf + 1

        self.solve_red = solve_red

        self.blue_mesh_list = self._generate_mesh_list(self.blue_resolution_list, agent_type="blue")
        self.red_mesh_list = self._generate_mesh_list(self.red_resolution_list, agent_type="red")
        self.maxmin_value_list = []
        self.blue_maxmin_strategy, self.red_maxmin_strategy = [], []

        if self.solve_red:
            self.blue_minmax_strategy, self.red_minmax_strategy = [], []
            self.minmax_value_list = []
        else:
            self.blue_minmax_strategy, self.red_minmax_strategy, self.minmax_value_list = None, None, None

        self._initialize_index_mapping()

    def solve(self):
        blue_mesh_tf = [mesh[-1] for mesh in self.blue_mesh_list]
        red_mesh_tf = [mesh[-1] for mesh in self.red_mesh_list]
        maxmin_value_tf = np.zeros((self.blue_prod_mesh_size_list[-1], self.red_prod_mesh_size_list[-1]))

        if self.solve_red:
            minmax_value_tf = np.zeros((len(blue_mesh_tf), len(red_mesh_tf)))

        # p_list and q_list are of one dimension smaller than the mu_list and nu_list, due to the simplex constraint
        for i, p_list in enumerate(self.cartesian_product(blue_mesh_tf)):
            for j, q_list in enumerate(self.cartesian_product(red_mesh_tf)):
                maxmin_value_tf[i, j] = self.game.reward(mu_list=p_list, nu_list=q_list, t=self.Tf)
                if self.solve_red:
                    minmax_value_tf[i, j] = self.game.reward(mu=p_list, nu=q_list, t=self.Tf)
        self.maxmin_value_list.append(maxmin_value_tf)

        if self.solve_red:
            self.minmax_value_list.append(minmax_value_tf)

        for t in reversed(range(0, self.Tf)):
            print("Solving stage {}/{}".format(t, self.Tf))
            blue_mesh_prime = [self.blue_mesh_list[i][t + 1] for i in range(self.game.n_blue_types)]
            red_mesh_prime = [self.red_mesh_list[j][t + 1] for j in range(self.game.n_red_types)]
            maxmin_value_prime = self.maxmin_value_list[-1]

            if self.solve_red:
                minmax_value_prime = self.minmax_value_list[-1]

            blue_mesh, red_mesh = [blue_mesh[t] for blue_mesh in self.blue_mesh_list], \
                                  [red_mesh[t] for red_mesh in self.red_mesh_list]

            maxmin_value = np.zeros((self.blue_prod_mesh_size_list[t], self.red_prod_mesh_size_list[t]))
            blue_maxmin_policy, red_maxmin_policy = [[] for _ in range(self.blue_prod_mesh_size_list[t])], \
                                                    [[] for _ in range(self.blue_prod_mesh_size_list[t])]

            if self.solve_red:
                minmax_value = np.zeros((self.blue_prod_mesh_size_list[t], self.red_prod_mesh_size_list[t]))
                blue_minmax_policy, red_minmax_policy = [[] for _ in range(self.blue_prod_mesh_size_list[t])], \
                                                        [[] for _ in range(self.blue_prod_mesh_size_list[t])]

            for i, p_list in enumerate(self.cartesian_product(blue_mesh)):
                print(i)
                for j, q_list in enumerate(self.cartesian_product(red_mesh)):
                    RSet_blue_list, RSet_red_list = self.game.generate_Rset(mu_list=p_list, nu_list=q_list, t=t)

                    # construct mesh for the RSets
                    p_mesh_tmp, q_mesh_tmp, maxmin_value_temp = self.construct_value_matrix(RSet_blue_list,
                                                                                            RSet_red_list,
                                                                                            blue_mesh_prime,
                                                                                            red_mesh_prime,
                                                                                            maxmin_value_prime, t)

                    maxmin_value[i, j], maxmin_max_index, maxmin_min_index = max_min(
                        value_matrix=maxmin_value_temp)

                    blue_maxmin_policy[i].append(p_mesh_tmp[maxmin_max_index])
                    red_maxmin_policy[i].append(q_mesh_tmp[maxmin_min_index])

                    if self.solve_red:
                        p_mesh_tmp, q_mesh_tmp, minmax_value_temp = self.construct_value_matrix(RSet_blue_list,
                                                                                                RSet_red_list,
                                                                                                blue_mesh_prime,
                                                                                                red_mesh_prime,
                                                                                                minmax_value_prime, t)
                        tmp, minmax_min_index, minmax_max_index = max_min(value_matrix=-minmax_value_temp.transpose())
                        minmax_value[i, j] = - tmp
                        blue_minmax_policy[i].append([p_mesh_tmp[minmax_max_index], 1 - p_mesh_tmp[minmax_max_index]])
                        red_maxmin_policy[i].append([q_mesh_tmp[minmax_min_index], 1 - q_mesh_tmp[minmax_min_index]])

            self.maxmin_value_list.append(maxmin_value)
            self.blue_maxmin_strategy.append(blue_maxmin_policy)
            self.blue_maxmin_strategy.append(red_maxmin_policy)

            if self.solve_red:
                self.minmax_value_list.append(minmax_value)
                self.blue_minmax_strategy.append(blue_minmax_policy)
                self.red_minmax_strategy.append(red_minmax_policy)

        self.maxmin_value_list.reverse()
        self.blue_maxmin_strategy.reverse()
        self.red_maxmin_strategy.reverse()

        if self.solve_red:
            self.minmax_value_list.reverse()
            self.blue_minmax_strategy.reverse()
            self.red_minmax_strategy.reverse()

    # def construct_value_matrix_2d(self, blue_RSet_verts: List[RSet], red_Reset_verts: List[RSet], blue_mesh_prime, red_mesh_prime, value_prime):
    #     blue_start_index, blue_start_row = linear_approximation(point=blue_RSet_verts[0], mesh=blue_mesh_prime,
    #                                                             v_list_list=value_prime)
    #     blue_end_index, blue_end_row = linear_approximation(point=blue_RSet_verts[1], mesh=blue_mesh_prime,
    #                                                         v_list_list=value_prime)
    #     blue_mesh_tmp = [blue_RSet_verts[0]] + list(blue_mesh_prime)[blue_start_index: blue_end_index] + [
    #         blue_RSet_verts[1]]
    #     value_temp = np.array([blue_start_row] + list(value_prime)[blue_start_index:blue_end_index] + [blue_end_row])
    #
    #     red_start_index, red_start_row = linear_approximation(point=red_Reset_verts[0], mesh=red_mesh_prime,
    #                                                           v_list_list=value_temp.transpose())
    #     red_end_index, red_end_row = linear_approximation(point=red_Reset_verts[1], mesh=red_mesh_prime,
    #                                                       v_list_list=value_temp.transpose())
    #
    #     red_mesh_tmp = [red_Reset_verts[0]] + list(red_mesh_prime)[red_start_index: red_end_index] + [
    #         red_Reset_verts[1]]
    #     value_temp = np.array([red_start_row] + list(value_temp.transpose())[red_start_index:red_end_index] + [
    #         red_end_row]).transpose()
    #
    #     assert value_temp.shape[0] == len(blue_mesh_tmp) and value_temp.shape[1] == len(red_mesh_tmp)
    #     return blue_mesh_tmp, red_mesh_tmp, value_temp

    def construct_value_matrix(self, blue_RSet_list: List[RSet], red_RSet_list: List[RSet], blue_mesh_prime,
                               red_mesh_prime, value_prime, t):

        blue_index_list_list, blue_point_list_list = [], []
        for blue_RSet_vert, blue_mesh in zip(blue_RSet_list, blue_mesh_prime):
            index_list, point_list = blue_RSet_vert.get_in_points(blue_mesh)
            blue_index_list_list.append(index_list)
            blue_point_list_list.append(point_list)

        red_index_list_list, red_point_list_list = [], []
        for red_RSet_vert, red_mesh in zip(red_RSet_list, red_mesh_prime):
            index_list, point_list = red_RSet_vert.get_in_points(red_mesh)
            red_index_list_list.append(index_list)
            red_point_list_list.append(point_list)

        value_tmp = np.zeros((self.cartesian_size(blue_index_list_list), self.cartesian_size(red_index_list_list)))
        blue_mesh_tmp = itertools.product(*blue_point_list_list)
        red_mesh_tmp = itertools.product(*red_point_list_list)

        i = 0
        for blue_index in itertools.product(*blue_index_list_list):
            j = 0
            for red_index in itertools.product(*red_index_list_list):
                b_indx = self.get_blue_global_index(blue_index, t)
                r_indx = self.get_red_global_index(red_index, t)
                value_tmp[i, j] = value_prime[b_indx, r_indx]
                j += 1
            i += 1

        return list(blue_mesh_tmp), list(red_mesh_tmp), value_tmp

    def _generate_mesh_list(self, resolution_list_list: List[List[int]], agent_type: str):
        if agent_type == "blue":
            n_state_list = self.game.n_blue_states_list
        else:
            n_state_list = self.game.n_red_states_list

        mesh_list = []
        for resolution_list, n_states in zip(resolution_list_list, n_state_list):
            mesh = []
            for resolution in resolution_list:
                mesh.append(list(itertools.product(*[np.linspace(0, 1, resolution + 1) for _ in range(n_states - 1)])))
            mesh_list.append(mesh)
        return mesh_list

    def save(self, data_path):
        data = {"maxmin_value": self.maxmin_value_list,
                "minmax_value": self.minmax_value_list,
                "maxmin_policies": [self.blue_maxmin_strategy, self.red_maxmin_strategy],
                "minmax_policies": [self.blue_minmax_strategy, self.red_minmax_strategy],
                "mesh_lists": [self.blue_mesh_list, self.red_mesh_list]}
        with open(data_path / "{}_{}.pkl".format(self.game.name, max(self.blue_resolution_list)),
                  "wb") as f:
            pkl.dump(data, f)

    def load(self, data_path):
        with open(data_path / "{}_{}.pkl".format(self.game.name, max(self.blue_resolution_list)),
                  "rb") as f:
            data = pkl.load(f)
        return data

    def cartesian_product(self, list_list: List[List]):
        if len(list_list) == 1:
            return [[item] for item in list_list[0]]
        return itertools.product(*list_list)

    def cartesian_size(self, list_list: List[List]):
        size = 1
        for a in list_list:
            size *= len(a)
        return size

    def get_blue_global_index(self, individual_index_list, t):
        return self.blue_map_list[t][individual_index_list]

    def get_red_global_index(self, individual_Index_list, t):
        return self.red_map_list[t][individual_Index_list]

    def _initialize_index_mapping(self):
        self.blue_map_list, self.red_map_list = [], []
        self.blue_prod_mesh_size_list, self.red_prod_mesh_size_list = [], []

        for t in range(self.Tf + 1):
            blue_res = [len(self.blue_mesh_list[i][t]) for i in range(self.game.n_blue_types)]
            red_res = [len(self.red_mesh_list[i][t]) for i in range(self.game.n_red_types)]
            blue_len = np.prod(np.array(blue_res))
            red_len = np.prod(np.array(red_res))

            self.blue_prod_mesh_size_list.append(blue_len)
            self.red_prod_mesh_size_list.append(red_len)

            blue_map = np.array(list(range(blue_len))).reshape(tuple(blue_res))
            red_map = np.array(list(range(red_len))).reshape(tuple(red_res))
            self.blue_map_list.append(blue_map)
            self.red_map_list.append(red_map)
