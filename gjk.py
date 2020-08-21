import numpy as np
import IPython
import matplotlib.pyplot as plt


class Polygon:
    ''' Convex polygonal shape. '''
    def __init__(self, vertices):
        self.vertices = vertices

    def support(self, d):
        ''' Return the point in the shape that has the highest dot product with
            unit vector d. '''
        # this is always one of the vertices
        idx = np.argmax([d.dot(v) for v in self.vertices])
        return self.vertices[idx]


class Circle:
    def __init__(self, c, r):
        self.c = c
        self.r = r

    def support(self, d):
        return self.c + self.r * d


def simplex_closest(S, x):
    ''' Return true if the simplex S contains the point x. '''
    # interior of the simplex is complex combination of vertices
    n = S.shape[0]
    A = np.vstack((S.T, np.ones(n)))
    b = np.append(x, 1)
    a = np.linalg.solve(A.T @ A, A.T @ b)
    p = A @ a

    IPython.embed()


# a1*s1 + a2*s2 + a3*s3 = x
# a1    + a2    + a3    = 1

# s1 s2 s3 * a = x
# 1  1  1      = 1

'''
given: point x, simplex s

min_a  ||x-y||^2
s.t.   y in simplex <=> y = S*a, \sum a = 1
       a >= 0

'''


# need to determine:
# 1. if simplex contains the origin
# 2. if not, what is the closest simplex and normal vector to the origin


def closest_simplex():
    pass


def gjk(shapes):
    pass


# def dx(Y):
#     ncol = Y.shape[1]
#     if ncol == 1:
#         return 1
#     y0 = Y[:, 0]
#     ym = Y[:, -1]
#     return dx(Y[:, :-1]) * (y0 - ym).dot(y0)


def dx(X, i):
    ncol = X.shape[1]
    if ncol == 1:
        return 1

    # remove ith column
    xi = X[:, i]
    Xi = np.delete(X, i, axis=1)

    # arbitrary which column of Xi is chosen, so take the first one
    x0 = Xi[:, 0]

    # compute dx for adding the ith element back into the simplex
    return np.sum([ dx(Xi, j) * (x0 - xi).dot(Xi[:, j]) for j in range(ncol-1)])

B = np.array([
    [1, 1, 1],  # 7
    [0, 1, 1],  # 3
    [1, 0, 1],  # 5
    [1, 1, 0],  # 6
    [0, 0, 1],  # 1
    [0, 1, 0],  # 2
    [1, 0, 0]], # 4
    dtype=np.bool)


def closest_point(Y):
    # b is 3-dim binary array
    for i in range(B.shape[0]):
        b = B[i, :]
        n = np.sum(b)
        X = Y[:, b]
        Z = Y[:, ~b]

        # need to check dXi and dXj - what are dXi?

        dxs = np.array([dx(X, i) for i in range(n)])
        # dzs = 

        if np.all(dxs[b] > 0) and np.all(dxs[~b] <= 0):
            return b, v
    IPython.embed()

    # # if there is only one point, then we can't recurse further
    # if np.sum(b) == 1:
    #     return b, None
    #
    # # recursively try smaller combinations
    # idx = np.nonzero(b)
    # for i in idx:
    #     bi = np.copy(b)
    #     bi[i] = False
    #     b, v = closest_point(Y, bi)
    #     if v is not None:
    #         return b, 


def johnson(Y):
    ''' Unrolled 2D version of Johnson's distance algorithm. '''
    y1 = Y[:, 0]
    y2 = Y[:, 1]
    y3 = Y[:, 2]

    # check vertices
    d1_y1 = 1
    d2_y1y2 = d1_y1 * (y1 - y2).dot(y1)
    d3_y1y3 = d1_y1 * (y1 - y3).dot(y1)

    if d2_y1y2 <= 0 and d3_y1y3 <= 0:
        return y1

    d2_y2 = 1
    d1_y1y2 = d2_y2 * (y2 - y1).dot(y2)
    d3_y2y3 = d2_y2 * (y2 - y3).dot(y2)

    if d1_y1y2 <= 0 and d3_y2y3 <= 0:
        return y2

    d3_y3 = 1
    d1_y1y3 = d3_y3 * (y3 - y1).dot(y3)
    d2_y2y3 = d3_y3 * (y3 - y2).dot(y3)

    if d1_y1y3 <= 0 and d2_y2y3 <= 0:
        return y3

    # check edges
    d1_y1y2y3 = d2_y2y3 * (y2 - y1).dot(y2) + d3_y2y3 * (y2 - y1).dot(y3)
    d2_y1y2y3 = d1_y1y3 * (y1 - y2).dot(y1) + d3_y1y3 * (y1 - y2).dot(y3)
    d3_y1y2y3 = d1_y1y2 * (y1 - y3).dot(y1) + d2_y1y2 * (y1 - y3).dot(y2)

    if d2_y2y3 > 0 and d3_y2y3 > 0 and d1_y1y2y3 <= 0:
        dX = d2_y2y3 + d3_y2y3
        v = (d2_y2y3 * y2 + d3_y2y3 * y3) / dX
        return v

    if d1_y1y3 > 0 and d3_y1y3 > 0 and d2_y1y2y3 <= 0:
        dX = d1_y1y3 + d3_y1y3
        v = (d1_y1y3 * y1 + d3_y1y3 * y3) / dX
        return v

    if d1_y1y2 > 0 and d2_y1y2 > 0 and d3_y1y2y3 <= 0:
        dX = d1_y1y2 + d2_y1y2
        v = (d1_y1y2 * y1 + d2_y1y2 * y2) / dX
        return v

    return np.zeros(2)


def main():
    # S = np.array([[0, 0], [0, 1], [1, 0]]) + np.array([0, 0.5])
    # X = S.T
    S = np.random.random((2, 3)) * 10 - 5
    print(S)
    v = johnson(S)

    fig, ax = plt.subplots(1)
    ax.set_xlim([-5, 5])
    ax.set_ylim([-5, 5])
    ax.add_patch(plt.Polygon(S.T, color='k', fill=False, closed=True))
    plt.plot([0, v[0]], [0, v[1]], '-o')
    plt.grid()
    plt.show()

    # IPython.embed()


# def main():
#     pygame.init()
#
#     screen = pygame.display.set_mode((240, 240))
#
#     # define a variable to control the main loop
#     running = True
#
#     # main loop
#     while running:
#         # event handling, gets all event from the event queue
#         for event in pygame.event.get():
#             # only do something if the event is of type QUIT
#             if event.type == pygame.QUIT:
#                 # change the value to False, to exit the main loop
#                 print('user exited')
#                 running = False
#

if __name__ == '__main__':
    main()
