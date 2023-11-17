import numpy as np
from simple_example.examples import MFTG
from typing import List


class PerimeterDefenseGame(MFTG):
    def __init__(self, blue_rho=[0.2, 0.3], red_rho=[0.5], alpha: float = 0.6,beta: float = 0.1, Tf=5):
        super().__init__(name="TwoNodePerimeter", n_blue_states_list=[2, 2], n_red_states_list=[2],
                         n_blue_actions_list=[2, 2], n_red_actions_list=[2],
                         rho_list=blue_rho + red_rho, Tf=Tf)

        self.alpha, self.beta = alpha, beta

    def blue_dynamics(self, x: int, u: int, x_prime: int,
                      agent_type: int, mu_list: List, nu_list: List,
                      blue_rho_list: List, red_rho_list, t=None) -> float:

        if agent_type == 0:
            if x == 0:
                if x_prime == 0:
                    return 1.0
                else:
                    return 0.0
            if x == 1:
                if u == 0:
                    if x_prime == 1:
                        return 1.0
                    else:
                        return 0
                else:
                    if x_prime == 0:
                        return 1.0
                    else:
                        return 0.0

        # if agent_type == 1:
        #     if x == 0:
        #         if u == 0:
        #             if x_prime == 0:
        #                 return 1.0
        #             else:
        #                 return 0.0
        #         if u == 1:
        #             if x_prime == 1:
        #                 return 1.0
        #             else:
        #                 return 0.0
        #     if x == 1:
        #         if u == 0:
        #             if x_prime == 0:
        #                 return 1.0
        #             else:
        #                 return 0.0
        #         if u == 1:
        #             if x_prime == 2:
        #                 return 1.0
        #             else:
        #                 return 0.0
        #     if x == 2:
        #         if u == 0:
        #             if x_prime == 2:
        #                 return 1.0
        #             else:
        #                 return 0.0
        #         if u == 1:
        #             if x_prime == 1:
        #                 return 1.0
        #             else:
        #                 return 0.0

        if agent_type == 1:
            if u == 0:
                if x == x_prime:
                    return 1.0
                else:
                    return 0.0
            elif u == 1:
                if x != x_prime:
                    return 1.0
                else:
                    return 0.0

    def red_dynamics(self, y: int, v: int, y_prime: int,
                     agent_type: int, mu_list: List, nu_list: List,
                     blue_rho_list: List, red_rho_list, t=None) -> float:
        if y == 0:
            if v == 0:
                if y_prime == 0:
                    return 1.0
                else:
                    return 0.0

            elif v == 1:
                p = (red_rho_list[0] * nu_list[0][0]
                     - self.alpha * blue_rho_list[0] * mu_list[0][0] - (1 - self.alpha) * blue_rho_list[1] * mu_list[1][
                         0])
                p = min(max(p, 0.1), 1)
                if y_prime == 1:
                    return p
                elif y_prime == 0:
                    return 1 - p
                else:
                    return 0.0

            else:
                raise Exception("red action error!")

        if y == 1:
            if v == 1:
                if y_prime == 0:
                    return 1.0
                else:
                    return 0.0

            if v == 0:
                p = (red_rho_list[0] * nu_list[0][1]
                     - self.alpha * blue_rho_list[0] * mu_list[0][1] - (1 - self.alpha) * blue_rho_list[1] * mu_list[1][
                         1])
                p = min(0.3*max(p, 0.6), 1)
                if y_prime == 1:
                    return p
                elif y_prime == 0:
                    return 1 - p

    def reward(self, mu_list: List, nu_list: List, t=None) -> float:
        if t == None:
            raise Exception("no t input for reward!")

        reward = -(self.red_rho_list[0] * (1 - nu_list[0][0])
        - self.beta * (self.blue_rho_list[0] * (1- mu_list[0][0])
        + self.blue_rho_list[1] * (1-mu_list[1][0])))

        return reward
