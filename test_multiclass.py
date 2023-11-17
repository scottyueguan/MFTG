import numpy as np

from simple_example.examples import MulticlassExample1
from simple_example.perimeter_defense import PerimeterDefenseGame
from simple_example.solver import Solver, linear_approximation_2d
from utils import ROOT_PATH
import os

if __name__ == "__main__":

    # save directories
    data_path = ROOT_PATH / "test_data"

    # set up save directory
    if not os.path.exists(data_path):
        os.makedirs(data_path)

    rho_blue = [0.1, 0.4]
    rho_red = [0.5]

    T = 2

    example = PerimeterDefenseGame(Tf=T, blue_rho=rho_blue, red_rho=rho_red)
    solver = Solver(game=example, blue_resolution_list=[[10 for _ in range(T+1)], [20 for _ in range(T+1)]],
                    red_resolution_list=[[100 for _ in range(T+1)]])

    solver.solve()
    solver.save(data_path)



