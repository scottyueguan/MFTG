import numpy as np
from scipy.optimize import linprog


# def in_hull(points, x):
#     n_points = len(points)
#     n_dim = len(x)
#     c = np.zeros(n_points)
#     A = np.r_[points.T, np.ones((1, n_points))]
#     b = np.r_[x, np.ones(1)]
#     lp = linprog(c, A_eq=A, b_eq=b)
#     return lp.success


class RSet:
    def __init__(self, vertices):
        self.size = len(vertices)
        self.vertices = np.array(vertices)
        self.dim = len(vertices[0])
        self.A = np.r_[self.vertices.T, np.ones((1, self.size))]
        self.c = np.zeros(self.size)
        self.over_constrained = (self.size <= self.dim)

    def get_in_points(self, points):
        index_list, point_list = [], []
        for i, point in enumerate(points):
            if len(point) < self.dim:
                assert len(point) == self.dim - 1
                point = list(point)
                point.append(1 - sum(point))
            if self.in_hull(x=np.array(point)):
                index_list.append(i)
                point_list.append(point)
        return index_list, point_list

    def in_hull(self, x):
        b = np.r_[x, np.ones(1)]
        c_, res, _, _ = np.linalg.lstsq(self.A, b)
        if (c_ >= 0).all() and res < 1e-4:
            return True
        elif self.over_constrained:
            return False
        else:
            lp = linprog(self.c, A_eq=self.A, b_eq=b)
            return lp.success
