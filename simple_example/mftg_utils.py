import numpy as np
from scipy.optimize import linprog
from typing import List


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

        if self.dim > 2:
            self.A = np.r_[self.vertices.T, np.ones((1, self.size))]
            self.c = np.zeros(self.size)
            self.over_constrained = (self.size <= self.dim)

        else:
            p_list = [v[0] for v in vertices]
            self.min_p, self.max_p = min(p_list), max(p_list)

    def get_in_points(self, points):
        if self.dim == 2:
            start_index = np.where(points - self.min_p >= 0)[0][0]
            end_index = np.where(points - self.max_p >= 0)[0][0]
            index_list = list(range(start_index, end_index))
            point_list = [(points[i][0], 1-points[i][0]) for i in index_list]
            return index_list, point_list


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
        if self.dim == 2:
            if self.min_p <= x[0] <= self.max_p:
                return True
            else:
                return False


        b = np.r_[x, np.ones(1)]
        c_, res, _, _ = np.linalg.lstsq(self.A, b)
        if (c_ >= 0).all() and res < 1e-4:
            return True
        elif self.over_constrained:
            return False
        else:
            lp = linprog(self.c, A_eq=self.A, b_eq=b)
            return lp.success


if __name__ == "__main__":
    rset = RSet(vertices=[np.array([0.1, 0.9, 0.0]), np.array([0, 0.9, 0.1]), np.array([0.0951, 0.905, 0.0])])
