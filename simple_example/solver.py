import numpy as np
from examples import MFTG
from typing import List
import pickle as pkl
from utils import ROOT_PATH


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
    def __init__(self, game: MFTG, blue_resolution_list: List[int], red_resolution_list: List[int]):
        self.game = game
        self.blue_resolution_list = blue_resolution_list
        self.red_resolution_list = red_resolution_list
        self.Tf = self.game.Tf
        assert len(blue_resolution_list) == len(red_resolution_list) == self.Tf + 1

        self.blue_mesh_list = self._generate_mesh_list(self.blue_resolution_list)
        self.red_mesh_list = self._generate_mesh_list(self.red_resolution_list)

        self.maxmin_value_list, self.minmax_value_list = [], []
        self.blue_minmax_strategy, self.blue_maxmin_strategy = [], []
        self.red_minmax_strategy, self.red_maxmin_strategy = [], []

    def solve(self):
        blue_mesh_tf, red_mesh_tf = self.blue_mesh_list[-1], self.red_mesh_list[-1]
        maxmin_value_tf = np.zeros((len(blue_mesh_tf), len(red_mesh_tf)))
        minmax_value_tf = np.zeros((len(blue_mesh_tf), len(red_mesh_tf)))

        for i, p in enumerate(blue_mesh_tf):
            for j, q in enumerate(red_mesh_tf):
                mu, nu = [p, 1 - p], [q, 1 - q]
                maxmin_value_tf[i, j] = self.game.reward(mu=mu, nu=nu, t=self.Tf)
                minmax_value_tf[i, j] = self.game.reward(mu=mu, nu=nu, t=self.Tf)
        self.maxmin_value_list.append(maxmin_value_tf)
        self.minmax_value_list.append(minmax_value_tf)

        for t in reversed(range(0, self.Tf)):
            print("Solving stage {}/{}".format(t, self.Tf))
            blue_mesh_prime, red_mesh_prime = self.blue_mesh_list[t + 1], self.red_mesh_list[t + 1]
            maxmin_value_prime, minmax_value_prime = self.maxmin_value_list[-1], self.minmax_value_list[-1]

            blue_mesh, red_mesh = self.blue_mesh_list[t], self.red_mesh_list[t]

            maxmin_value = np.zeros((self.blue_resolution_list[t] + 1, self.red_resolution_list[t] + 1))
            minmax_value = np.zeros((self.blue_resolution_list[t] + 1, self.red_resolution_list[t] + 1))

            blue_maxmin_policy, blue_minmax_policy = [[] for _ in range(self.blue_resolution_list[t] + 2)], \
                                                     [[] for _ in range(self.blue_resolution_list[t] + 2)]
            red_maxmin_policy, red_minmax_policy = [[] for _ in range(self.red_resolution_list[t] + 2)], \
                                                   [[] for _ in range(self.red_resolution_list[t] + 2)]

            for i, p in enumerate(blue_mesh):
                for j, q in enumerate(red_mesh):
                    mu, nu = [p, 1 - p], [q, 1 - q]
                    RSet_blue_verts = self.game.generate_blue_Rset(mu=mu, nu=nu, t=t)
                    RSet_red_verts = self.game.generate_red_Rset(mu=mu, nu=nu, t=t)

                    # construct mesh for the RSets
                    p_mesh_tmp, q_mesh_tmp, maxmin_value_temp = self.construct_value_matrix(RSet_blue_verts,
                                                                                            RSet_red_verts,
                                                                                            blue_mesh_prime,
                                                                                            red_mesh_prime,
                                                                                            maxmin_value_prime)

                    p_mesh_tmp, q_mesh_tmp, minmax_value_temp = self.construct_value_matrix(RSet_blue_verts,
                                                                                            RSet_red_verts,
                                                                                            blue_mesh_prime,
                                                                                            red_mesh_prime,
                                                                                            minmax_value_prime)

                    maxmin_value[i, j], maxmin_max_index, maxmin_min_index = max_min(
                        value_matrix=maxmin_value_temp)
                    tmp, minmax_min_index, minmax_max_index = max_min(value_matrix=-minmax_value_temp.transpose())
                    minmax_value[i, j] = - tmp

                    blue_maxmin_policy[i].append([p_mesh_tmp[maxmin_max_index], 1 - p_mesh_tmp[maxmin_max_index]])
                    red_maxmin_policy[i].append([q_mesh_tmp[maxmin_min_index], 1 - q_mesh_tmp[maxmin_min_index]])
                    blue_minmax_policy[i].append([p_mesh_tmp[minmax_max_index], 1 - p_mesh_tmp[minmax_max_index]])
                    red_maxmin_policy[i].append([q_mesh_tmp[minmax_min_index], 1 - q_mesh_tmp[minmax_min_index]])

            self.maxmin_value_list.append(maxmin_value)
            self.minmax_value_list.append(minmax_value)
            self.blue_minmax_strategy.append(blue_minmax_policy)
            self.blue_maxmin_strategy.append(blue_maxmin_policy)
            self.red_minmax_strategy.append(red_minmax_policy)
            self.blue_maxmin_strategy.append(red_maxmin_policy)

        self.maxmin_value_list.reverse()
        self.minmax_value_list.reverse()
        self.blue_maxmin_strategy.reverse()
        self.blue_minmax_strategy.reverse()
        self.red_maxmin_strategy.reverse()
        self.red_minmax_strategy.reverse()

    def construct_value_matrix(self, blue_RSet_verts, red_Reset_verts, blue_mesh_prime, red_mesh_prime, value_prime):
        blue_start_index, blue_start_row = linear_approximation(point=blue_RSet_verts[0], mesh=blue_mesh_prime,
                                                                v_list_list=value_prime)
        blue_end_index, blue_end_row = linear_approximation(point=blue_RSet_verts[1], mesh=blue_mesh_prime,
                                                            v_list_list=value_prime)
        blue_mesh_tmp = [blue_RSet_verts[0]] + list(blue_mesh_prime)[blue_start_index: blue_end_index] + [
            blue_RSet_verts[1]]
        value_temp = np.array([blue_start_row] + list(value_prime)[blue_start_index:blue_end_index] + [blue_end_row])

        red_start_index, red_start_row = linear_approximation(point=red_Reset_verts[0], mesh=red_mesh_prime,
                                                              v_list_list=value_temp.transpose())
        red_end_index, red_end_row = linear_approximation(point=red_Reset_verts[1], mesh=red_mesh_prime,
                                                          v_list_list=value_temp.transpose())

        red_mesh_tmp = [red_Reset_verts[0]] + list(red_mesh_prime)[red_start_index: red_end_index] + [
            red_Reset_verts[1]]
        value_temp = np.array([red_start_row] + list(value_temp.transpose())[red_start_index:red_end_index] + [
            red_end_row]).transpose()

        assert value_temp.shape[0] == len(blue_mesh_tmp) and value_temp.shape[1] == len(red_mesh_tmp)
        return blue_mesh_tmp, red_mesh_tmp, value_temp

    def _generate_mesh_list(self, resolution_list: List[int]):
        mesh_list = []
        for resolution in resolution_list:
            mesh_list.append(np.linspace(0, 1, resolution + 1))
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
