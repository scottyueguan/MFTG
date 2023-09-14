import numpy as np
from typing import List
from simple_example.mftg_utils import RSet


def generate_2D_Rset(mu, transition_matrix_list):
    p_min, p_max = 2, -2
    for F in transition_matrix_list:
        mu_prime = np.matmul(mu, F)
        if mu_prime[0] < p_min:
            p_min = mu_prime[0]
        if mu_prime[0] > p_max:
            p_max = mu_prime[0]
    rset = RSet(vertices=[np.array([p_min, 1 - p_min]), np.array([p_max, 1 - p_max])])
    return rset


class MFTG:
    def __init__(self, name, n_blue_states_list: List, n_red_states_list: List,
                 n_blue_actions_list: List, n_red_actions_list: List, rho_list: List, Tf):
        self.name = name
        self.n_blue_states_list, self.n_red_states_list = n_blue_states_list, n_red_states_list
        self.n_blue_actions_list, self.n_red_actions_list = n_blue_actions_list, n_red_actions_list

        self.n_blue_types, self.n_red_types = len(self.n_blue_states_list), len(self.n_red_states_list)

        self.blue_extreme_policies_list = self._generate_extreme_policies(n_blue_states_list, n_blue_actions_list)
        self.red_extreme_policies_list = self._generate_extreme_policies(n_red_states_list, n_red_actions_list)
        self.blue_rho_list, self.red_rho_list = rho_list[0:self.n_blue_types], rho_list[self.n_blue_types:]
        self.Tf = Tf

    def blue_dynamics(self, x: int, u: int, x_prime: int,
                      agent_type: int, mu_list: List, nu_list: List,
                      blue_rho_list: List, red_rho_list, t=None) -> float:
        raise NotImplemented

    def blue_type_l_transition_matrix(self, blue_type_l_policy, agent_type: int, mu_list: List, nu_list: List, t=None):
        n_blue_states, n_blue_actions = self.n_blue_states_list[agent_type], self.n_blue_actions_list[agent_type]
        F_t = np.zeros((n_blue_states, n_blue_states))
        for x in range(n_blue_states):
            for u in range(n_blue_actions):
                prob_u = blue_type_l_policy[x][u]
                for x_prime in range(n_blue_states):
                    F_t[x, x_prime] += self.blue_dynamics(x, u, x_prime, agent_type, mu_list, nu_list,
                                                          self.blue_rho_list, self.red_rho_list, t) * prob_u
        assert (abs(np.sum(F_t, axis=1) - 1.0) < 1e-5).all()
        assert (F_t >= -1e-5).all()
        return F_t

    def red_dynamics(self, y: int, v: int, y_prime: int,
                     agent_type: int, mu_list: List, nu_list: List,
                     blue_rho_list: List, red_rho_list, t=None) -> float:
        raise NotImplemented

    def red_type_m_transition_matrix(self, red_type_m_policy, agent_type: int, mu_list: List, nu_list: List, t=None):
        n_red_states, n_red_actions = self.n_red_states_list[agent_type], self.n_red_actions_list[agent_type]
        G_t = np.zeros((n_red_states, n_red_states))
        for y in range(n_red_states):
            for v in range(n_red_actions):
                prob_u = red_type_m_policy[y][v]
                for y_prime in range(n_red_states):
                    G_t[y, y_prime] += self.red_dynamics(y, v, y_prime, agent_type, mu_list, nu_list,
                                                         self.blue_rho_list, self.red_rho_list, t) * prob_u
        assert (abs(np.sum(G_t, axis=1) - 1.0) < 1e-5).all()
        return G_t

    def reward(self, mu_list: List, nu_list: List, t=None):
        raise NotImplemented

    def generate_Rset(self, mu_list: List, nu_list: List, t=None) -> List[List[RSet]]:
        mu_list = self._reconfig_mu_list(mu_list)
        nu_list = self._reconfig_nu_list(nu_list)
        blue_Rset = self.generate_blue_Rset(mu_list, nu_list, t)
        red_Rset = self.generate_red_Rset(mu_list, nu_list, t)
        return [blue_Rset, red_Rset]

    def generate_blue_Rset(self, mu_list: List, nu_list: List, t=None) -> List[RSet]:
        R_set_list = []
        for l in range(self.n_blue_types):
            transition_matrix_list = [self.blue_type_l_transition_matrix(policy, l, mu_list, nu_list, t) for policy
                                      in self.blue_extreme_policies_list[l]]
            mu = np.array(mu_list[l])
            vertices_list = [np.matmul(mu, transition_matrix_list[0])]
            for k in range(1, len(transition_matrix_list)):
                vertex = np.matmul(mu, transition_matrix_list[k])
                diff = np.array([np.linalg.norm(vertex - point) for point in vertices_list])
                if (diff > 1e-5).all():
                    vertices_list.append(vertex)
            R_set_list.append(RSet(vertices=vertices_list))
        return R_set_list

    def generate_red_Rset(self, mu_list: List, nu_list: List, t=None) -> List[RSet]:
        R_set_list = []
        for m in range(self.n_red_types):
            transition_matrix_list = [self.red_type_m_transition_matrix(policy, m, mu_list, nu_list, t) for policy
                                      in self.red_extreme_policies_list[m]]
            nu = np.array(nu_list[m])
            vertices_list = [np.matmul(nu, transition_matrix_list[0])]
            for k in range(1, len(transition_matrix_list)):
                vertex = np.matmul(nu, transition_matrix_list[k])
                diff = np.array([np.linalg.norm(vertex - point) for point in vertices_list])
                if (diff > 1e-5).all():
                    vertices_list.append(vertex)
            R_set_list.append(RSet(vertices=vertices_list))
        return R_set_list

    @staticmethod
    def _generate_extreme_policies(n_states_list: List, n_actions_list: List):
        extreme_policy_list = []
        for n_states, n_actions in zip(n_states_list, n_actions_list):
            extreme_policy = [[]]

            while len(extreme_policy[0]) < n_states:
                tmp = extreme_policy.pop(0)
                for a in range(n_actions):
                    tmp_ = tmp.copy()
                    dist = [0.0 for _ in range(n_actions)]
                    dist[a] = 1.0
                    tmp_.append(dist)
                    extreme_policy.append(tmp_)
            extreme_policy_list.append(extreme_policy)
        return extreme_policy_list

    def _reconfig_mu_list(self, mu_list: List):
        if len(mu_list[0]) < self.n_blue_states_list[0]:
            mu_list_ = []
            for p in mu_list:
                p = list(p)
                p.append(1 - np.sum(p))
                mu_list_.append(p)
            return mu_list_
        else:
            return mu_list

    def _reconfig_nu_list(self, nu_list: List):
        if len(nu_list[0]) < self.n_red_states_list[0]:
            nu_list_ = []
            for q in nu_list:
                q = list(q)
                q.append(1 - np.sum(q))
                nu_list_.append(q)
            return nu_list_
        else:
            return nu_list


class MulticlassExample1(MFTG):
    def __init__(self, blue_rho, red_rho):
        super().__init__(name="MulticlassExample1", n_blue_states_list=[2, 3], n_red_states_list=[2],
                         n_blue_actions_list=[3, 2], n_red_actions_list=[2],
                         rho_list=blue_rho + red_rho, Tf=2)

    def blue_dynamics(self, x: int, u: int, x_prime: int,
                      agent_type: int, mu_list: List, nu_list: List,
                      blue_rho_list: List, red_rho_list, t=None) -> float:
        p = 0

        if agent_type == 0:
            if u == 0:
                p = 0.5
            elif u == 1:
                if x_prime != x:
                    p = 0.7
                else:
                    p = 0.3
            elif u == 2:
                if x_prime == 0:
                    p = 0.2
                else:
                    p = 0.8
            else:
                raise Exception("type 1 blue wrong!")

        if agent_type == 1:
            if u == 0 and x == x_prime:
                p = 1
            elif u == 1 and x != x_prime:
                p = 0.5
            else:
                p = 0
        return p

    def red_dynamics(self, y: int, v: int, y_prime: int,
                     agent_type: int, mu_list: List, nu_list: List,
                     blue_rho_list: List, red_rho_list, t=None) -> float:

        if agent_type == 0:
            q = 1 - mu_list[0][0] * blue_rho_list[0]
            if v == 0:
                if y == y_prime:
                    return q
                else:
                    return 1 - q
            if v == 1:
                if y != y_prime:
                    return q
                else:
                    return 1 - q

        if agent_type == 1:
            q = 1 - mu_list[1][1] * blue_rho_list[1]
            if v == 1:
                if y == y_prime:
                    return q
                else:
                    return 1 - q
            if v == 0:
                if y != y_prime:
                    return q
                else:
                    return 1 - q

    def reward(self, mu_list: List, nu_list: List, t=None):
        mu_list = self._reconfig_mu_list(mu_list)
        nu_list = self._reconfig_nu_list(nu_list)

        r = self.blue_rho_list[0] * mu_list[0][1] + self.blue_rho_list[1] * mu_list[1][2] - \
            self.red_rho_list[0] * nu_list[0][1]

        return r


class SimpleExample1(MFTG):
    def __init__(self, rho):
        super().__init__(name="SimpleExample1", n_blue_states_list=[2], n_red_states_list=[2],
                         n_blue_actions_list=[2], n_red_actions_list=[2],
                         rho_list=[rho, 1 - rho], Tf=2)

    def blue_dynamics(self, x, u, x_prime, agent_type, mu_list, nu_list, blue_rho_list, red_rho_list, t=None):
        p = 0.0
        rho = blue_rho_list[0]
        mu, nu = mu_list[0], nu_list[0]
        if u == 0:
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

    def red_dynamics(self, y, v, y_prime, agent_type, mu_list, nu_list, blue_rho_list, red_rho_list, t=None):
        q = 0.0
        rho = blue_rho_list[0]
        mu, nu = mu_list[0], nu_list[0]
        if v == 0:
            if y == y_prime:
                q = 0.5 * (1 + ((1 - rho) * nu[y] - rho * mu[y]))
            else:
                q = 0.5 * (1 - ((1 - rho) * nu[y] - rho * mu[y]))
        else:
            if y == y_prime:
                q = 0.5 * (1 - 0.3 * ((1 - rho) * nu[y] - rho * mu[y]))
            else:
                q = 0.5 * (1 + 0.3 * ((1 - rho) * nu[y] - rho * mu[y]))

        return q

    def reward(self, mu_list, nu_list, t=None):
        mu_list = self._reconfig_mu_list(mu_list)
        nu_list = self._reconfig_nu_list(nu_list)

        return mu_list[0][1]

    def generate_blue_Rset(self, mu_list: List, nu_list: List, t=None):
        mu_list = self._reconfig_mu_list(mu_list)
        nu_list = self._reconfig_nu_list(nu_list)
        transition_matrix_list = [self.blue_type_l_transition_matrix(policy, 0, mu_list, nu_list, t) for policy
                                  in self.blue_extreme_policies_list[0]]
        RSet = [generate_2D_Rset(mu=mu_list[0], transition_matrix_list=transition_matrix_list)]
        return RSet

    def generate_red_Rset(self, mu_list: List, nu_list: List, t=None):
        mu_list = self._reconfig_mu_list(mu_list)
        nu_list = self._reconfig_nu_list(nu_list)
        transition_matrix_list = [self.red_type_m_transition_matrix(policy, 0, mu_list, nu_list, t=t) for policy
                                  in self.red_extreme_policies_list[0]]
        RSet = [generate_2D_Rset(mu=nu_list[0], transition_matrix_list=transition_matrix_list)]
        return RSet

#
# class SimpleExample2(MFTG):
#     def __init__(self, rho):
#         super().__init__(name="simple_example2_{}".format(rho),
#                          n_blue_states=2, n_red_states=2,
#                          n_blue_actions=2, n_red_actions=2,
#                          rho=rho, Tf=2)
#
#     def red_dynamics(self, y, v, y_prime, mu, nu, t=None):
#         if t == 0:
#             if y_prime == y:
#                 return 1.0
#             else:
#                 return 0.0
#         else:
#             if v == 0:
#                 if y_prime == y:
#                     return 1.0
#                 else:
#                     return 0.0
#             elif v == 1:
#                 if y == 1:
#                     prob = min(5 * ((mu[0] - 1 / np.sqrt(2)) ** 2 + (mu[1] - (1 - 1 / np.sqrt(2))) ** 2), 1)
#                     if y_prime == 1:
#                         return 1 - prob
#                     else:
#                         return prob
#                 else:
#                     if y_prime == 1:
#                         return 1.0
#                     else:
#                         return 0.0
#
#     def blue_dynamics(self, x, u, x_prime, mu, nu, t=None):
#         if u == 0:
#             if x == x_prime:
#                 return 1.0
#             else:
#                 return 0.0
#         else:
#             if x == x_prime:
#                 return 0.0
#             else:
#                 return 1.0
#
#     def reward(self, mu, nu, t=None):
#         if t <= 1:
#             return 0.0
#         else:
#             return -nu[0]
#
#     def generate_red_Rset(self, mu, nu, t=None):
#         if t == 0:
#             return np.array([nu[0], nu[0]])
#         else:
#             transition_matrix_list = [self.red_transition_matrix(policy=policy, mu=mu, nu=nu, t=t) for policy
#                                       in self.red_extreme_policies]
#             RSet = generate_2D_Rset(mu=nu, transition_matrix_list=transition_matrix_list)
#             return RSet
#
#     def generate_blue_Rset(self, mu, nu, t=None):
#         transition_matrix_list = [self.blue_transition_matrix(,
#         for policy
#         in self.blue_extreme_policies]
#         RSet = generate_2D_Rset(mu=mu, transition_matrix_list=transition_matrix_list)
#         return RSet
#
#
# class SimpleExample2_bak(MFTG):
#     def __init__(self, rho):
#         super().__init__(name="simple_example2_{}".format(rho),
#                          n_blue_states=2, n_red_states=2,
#                          n_blue_actions=2, n_red_actions=2,
#                          rho=rho, Tf=2)
#
#     def blue_dynamics(self, x, u, x_prime, mu, nu, t=None):
#         if t == 0:
#             if x_prime == x:
#                 return 1.0
#             else:
#                 return 0.0
#         else:
#             if u == 0:
#                 if x_prime == x:
#                     return 1.0
#                 else:
#                     return 0.0
#             elif u == 1:
#                 if x == 0:
#                     prob = min(5 * ((nu[0] - 1 / np.sqrt(2)) ** 2 + (nu[1] - (1 - 1 / np.sqrt(2))) ** 2), 1)
#                     if x_prime == 0:
#                         return 1 - prob
#                     else:
#                         return prob
#                 else:
#                     if x_prime == 0:
#                         return 1.0
#                     else:
#                         return 0.0
#
#     def red_dynamics(self, y, v, y_prime, mu, nu, t=None):
#         if v == 0:
#             if y == y_prime:
#                 return 1.0
#             else:
#                 return 0.0
#         else:
#             if y == y_prime:
#                 return 0.0
#             else:
#                 return 1.0
#
#     def reward(self, mu, nu, t=None):
#         if t <= 1:
#             return 0.0
#         else:
#             return mu[1]
#
#     def generate_blue_Rset(self, mu, nu, t=None):
#         if t == 0:
#             return np.array([mu[0], mu[0]])
#         else:
#             transition_matrix_list = [self.blue_transition_matrix(,
#             for policy
#             in self.blue_extreme_policies]
#             RSet = generate_2D_Rset(mu=mu, transition_matrix_list=transition_matrix_list)
#             return RSet
#
#     def generate_red_Rset(self, mu, nu, t=None):
#         transition_matrix_list = [self.red_transition_matrix(policy=policy, mu=mu, nu=nu, t=t) for policy
#                                   in self.red_extreme_policies]
#         RSet = generate_2D_Rset(mu=nu, transition_matrix_list=transition_matrix_list)
#         return RSet
