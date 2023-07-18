import numpy as np
from scipy.spatial import ConvexHull


def generate_2D_Rset(mu, transition_matrix_list):
    p_min, p_max = 2, -2
    for F in transition_matrix_list:
        mu_prime = np.matmul(mu, F)
        if mu_prime[0] < p_min:
            p_min = mu_prime[0]
        if mu_prime[0] > p_max:
            p_max = mu_prime[0]
    return np.array([p_min, p_max])


class MFTG:
    def __init__(self, name, n_blue_states, n_red_states, n_blue_actions, n_red_actions, rho, Tf):
        self.name = name
        self.n_blue_states, self.n_red_states = n_blue_states, n_red_states
        self.n_blue_actions, self.n_red_actions = n_blue_actions, n_red_actions
        self.blue_extreme_policies = self._generate_extreme_policies(n_blue_states, n_blue_actions)
        self.red_extreme_policies = self._generate_extreme_policies(n_red_states, n_red_actions)
        self.rho = rho
        self.Tf = Tf

    def blue_dynamics(self, x, u, x_prime, mu, nu, t=None):
        raise NotImplemented

    def blue_transition_matrix(self, policy, mu, nu, t=None):
        F_t = np.zeros((self.n_blue_states, self.n_blue_states))
        for x in range(self.n_blue_states):
            for u in range(self.n_blue_actions):
                prob_u = policy[x][u]
                for x_prime in range(self.n_blue_states):
                    F_t[x, x_prime] += self.blue_dynamics(x, u, x_prime, mu, nu, t) * prob_u
        assert (abs(np.sum(F_t, axis=1) - 1.0) < 1e-5).all()
        assert (F_t >= -1e-5).all()
        return F_t

    def red_dynamics(self, y, v, y_prime, mu, nu, t=None):
        raise NotImplemented

    def red_transition_matrix(self, policy, mu, nu, t=None):
        G_t = np.zeros((self.n_red_states, self.n_red_states))
        for y in range(self.n_red_states):
            for v in range(self.n_red_actions):
                prob_u = policy[y][v]
                for y_prime in range(self.n_red_states):
                    G_t[y, y_prime] += self.red_dynamics(y, v, y_prime, mu, nu, t) * prob_u
        assert (abs(np.sum(G_t, axis=1) - 1.0) < 1e-5).all()
        return G_t

    def reward(self, mu, nu, t=None):
        raise NotImplemented

    def generate_blue_Rset(self, mu, nu, t=None):
        raise NotImplemented

    def generate_red_Rset(self, mu, nu, t=None):
        raise NotImplemented

    @staticmethod
    def _generate_extreme_policies(n_states, n_actions):
        extreme_policy_list = [[]]

        while len(extreme_policy_list[0]) < n_states:
            tmp = extreme_policy_list.pop(0)
            for a in range(n_actions):
                tmp_ = tmp.copy()
                dist = [0.0 for _ in range(n_actions)]
                dist[a] = 1.0
                tmp_.append(dist)
                extreme_policy_list.append(tmp_)
        return extreme_policy_list


class SimpleExample1(MFTG):
    def __init__(self, rho):
        super().__init__(name="SimpleExample1", n_blue_states=2, n_red_states=2, n_blue_actions=2, n_red_actions=2,
                         rho=rho, Tf=2)

    def blue_dynamics(self, x, u, x_prime, mu, nu, t=None):
        p = 0.0
        if u == 0:
            if x == x_prime:
                p = 0.5 * (1 + (self.rho * mu[x] - (1 - self.rho) * nu[x]))
            else:
                p = 0.5 * (1 - (self.rho * mu[x] - (1 - self.rho) * nu[x]))
        else:
            if x == x_prime:
                p = 0.5 * (1 - 0.3 * (self.rho * mu[x] - (1 - self.rho) * nu[x]))
            else:
                p = 0.5 * (1 + 0.3 * (self.rho * mu[x] - (1 - self.rho) * nu[x]))

        return p

    def red_dynamics(self, y, v, y_prime, mu, nu, t=None):
        q = 0.0
        if v == 0:
            if y == y_prime:
                q = 0.5 * (1 + ((1 - self.rho) * nu[y] - self.rho * mu[y]))
            else:
                q = 0.5 * (1 - ((1 - self.rho) * nu[y] - self.rho * mu[y]))
        else:
            if y == y_prime:
                q = 0.5 * (1 - 0.3 * ((1 - self.rho) * nu[y] - self.rho * mu[y]))
            else:
                q = 0.5 * (1 + 0.3 * ((1 - self.rho) * nu[y] - self.rho * mu[y]))

        return q

    def reward(self, mu, nu, t=None):
        return mu[1]

    def generate_blue_Rset(self, mu, nu, t=None):
        transition_matrix_list = [self.blue_transition_matrix(policy=policy, mu=mu, nu=nu, t=t) for policy
                                  in self.blue_extreme_policies]
        RSet = generate_2D_Rset(mu=mu, transition_matrix_list=transition_matrix_list)
        return RSet

    def generate_red_Rset(self, mu, nu, t=None):
        transition_matrix_list = [self.red_transition_matrix(policy=policy, mu=mu, nu=nu, t=t) for policy
                                  in self.red_extreme_policies]
        RSet = generate_2D_Rset(mu=nu, transition_matrix_list=transition_matrix_list)
        return RSet


class SimpleExample2(MFTG):
    def __init__(self, rho):
        super().__init__(name="simple_example2_{}".format(rho),
                         n_blue_states=2, n_red_states=2,
                         n_blue_actions=2, n_red_actions=2,
                         rho=rho, Tf=2)

    def blue_dynamics(self, x, u, x_prime, mu, nu, t=None):
        if t == 0:
            if x_prime == x:
                return 1.0
            else:
                return 0.0
        else:
            if u == 0:
                if x_prime == x:
                    return 1.0
                else:
                    return 0.0
            elif u == 1:
                if x == 0:
                    prob = min(5 * ((nu[0] - 1 / np.sqrt(2)) ** 2 + (nu[1] - (1 - 1 / np.sqrt(2))) ** 2), 1)
                    if x_prime == 0:
                        return 1 - prob
                    else:
                        return prob
                else:
                    if x_prime == 0:
                        return 1.0
                    else:
                        return 0.0

    def red_dynamics(self, y, v, y_prime, mu, nu, t=None):
        if v == 0:
            if y == y_prime:
                return 1.0
            else:
                return 0.0
        else:
            if y == y_prime:
                return 0.0
            else:
                return 1.0

    def reward(self, mu, nu, t=None):
        if t <= 1:
            return 0.0
        else:
            return mu[1]

    def generate_blue_Rset(self, mu, nu, t=None):
        if t == 0:
            return np.array([mu[0], mu[0]])
        else:
            transition_matrix_list = [self.blue_transition_matrix(policy=policy, mu=mu, nu=nu, t=t) for policy
                                      in self.blue_extreme_policies]
            RSet = generate_2D_Rset(mu=mu, transition_matrix_list=transition_matrix_list)
            return RSet

    def generate_red_Rset(self, mu, nu, t=None):
        transition_matrix_list = [self.red_transition_matrix(policy=policy, mu=mu, nu=nu, t=t) for policy
                                  in self.red_extreme_policies]
        RSet = generate_2D_Rset(mu=nu, transition_matrix_list=transition_matrix_list)
        return RSet
