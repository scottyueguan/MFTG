import numpy as np

from simple_example.examples import MulticlassExample1
from simple_example.solver import Solver, linear_approximation_2d
from utils import ROOT_PATH

if __name__ == "__main__":
    res = 50
    rho_blue = [0.3, 0.2]
    rho_red = [0.5]

    example = MulticlassExample1(blue_rho=rho_blue, red_rho=rho_red)
    solver = Solver(game=example, blue_resolution_list=[[10, 10, 10], [5, 5, 5]], red_resolution_list=[[4, 4, 4]])

    solver.solve()
