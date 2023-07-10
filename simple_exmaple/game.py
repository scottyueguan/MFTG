import numpy as np


class MFTG:
    def __init__(self, n_blue_states, n_red_states, n_blue_actions, n_red_actions):
        self.n_blue_states, self.n_red_states = n_blue_states, n_red_states
        self.n_blue_actions, self.n_red_actions = n_blue_actions, n_red_actions

    def blue_dynamics(self, x, u, x_prime, mu, nu, rho, t=None):
        raise NotImplemented

    def blue_transition_matrix(self, policy, mu, nu, rho, t=None):
        F_t = np.zeros((self.n_blue_states, self.n_blue_states))
        for x in range(self.n_blue_states):
            for u in range(self.n_blue_actions):
                prob_u = policy[x, u]
                for x_prime in range(self.n_blue_states):
                    F_t[x, x_prime] += self.blue_dynamics(x, u, x_prime, mu, nu, rho, t) * prob_u
        assert (abs(np.sum(F_t, axis=1) - 1.0) < 1e-5).all()
        return F_t

    def red_dynamics(self, y, v, x_prime, mu, nu, rho, t=None):
        raise NotImplemented

    def red_transition_matrix(self, policy, mu, nu, rho, t=None):
        G_t = np.zeros((self.n_red_states, self.n_red_states))
        for y in range(self.n_red_states):
            for v in range(self.n_red_actions):
                prob_u = policy[y, v]
                for y_prime in range(self.n_red_states):
                    G_t[y, y_prime] += self.red_dynamics(y, v, y_prime, mu, nu, rho, t) * prob_u
        assert (abs(np.sum(G_t, axis=1) - 1.0) < 1e-5).all()
        return G_t