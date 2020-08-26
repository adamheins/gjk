# reference/experimental code that likely does not work
import numpy as np


def dx(Y, b, i):
    ''' Recursive calculation delta_i^{b} '''
    if b <= 0:
        raise ValueError('b cannot be <= 0')

    # base case
    if b == 1 or b == 2 or b == 4:
        return 1

    # zero ith bit
    bi = unset_bit(b, i)

    # get the first member of the set
    k = find_first_set(bi)

    dy = Y[:, k] - Y[:, i]

    D[b, i] = np.sum([ dx2(Y, bi, j) * dy.dot(Y[:, j]) for j in range(3) if get_bit(bi, j) ])
    return D[b, i]


def johnson2(Y, index_set):
    # store deltas: row is indexed by the points in the set X; column indexed
    # by delta index
    # {y1} represented by 001, {y2} by 010, etc
    D = np.array((8, 3))

    # singleton sets are unity
    D[0b001, 0] = 1
    D[0b010, 1] = 1
    D[0b100, 2] = 1

    D[0b011, 1] = D[0b001, 0] * (Y[:, 0] - Y[:, 1]).dot(Y[:, 0])
    D[0b101, 2] = D[0b001, 0] * (Y[:, 0] - Y[:, 2]).dot(Y[:, 0])

    if D[0b011, 1] <= 0 and D[0b101, 2] <= 0:
        return Y[:, 0]

    D[0b011, 0] = D[0b010, 1] * (Y[:, 1] - Y[:, 0]).dot(Y[:, 1])
    D[0b110, 2] = D[0b010, 1] * (Y[:, 1] - Y[:, 2]).dot(Y[:, 1])

    if D[0b011, 0] <= 0 and D[0b110, 2] <= 0:
        return Y[:, 1]

    D[0b101, 0] = D[0b100, 2] * (Y[:, 2] - Y[:, 0]).dot(Y[:, 2])
    D[0b110, 1] = D[0b100, 2] * (Y[:, 2] - Y[:, 1]).dot(Y[:, 2])

    if D[0b101, 0] <= 0 and D[0b110, 1] <= 0:
        return Y[:, 2]

    D[0b111, 0] = D[0b110, 1] * (Y[:, 1] - Y[:, 0]).dot(Y[:, 1]) \
                + D[0b110, 2] * (Y[:, 1] - Y[:, 0]).dot(Y[:, 2])

    D[0b111, 1] = D[0b101, 0] * (Y[:, 0] - Y[:, 1]).dot(Y[:, 0]) \
                + D[0b101, 2] * (Y[:, 0] - Y[:, 1]).dot(Y[:, 2])

    D[0b111, 2] = D[0b011, 0] * (Y[:, 0] - Y[:, 2]).dot(Y[:, 0]) \
                + D[0b011, 1] * (Y[:, 0] - Y[:, 2]).dot(Y[:, 1])


def johnson(Y):
    ''' Unrolled 2D version of Johnson's distance algorithm.
        Returns:
            v  closest point to origin
            X  simplex supporting v
            True if Y contains the origin, false otherwise '''
    # b is a binary array denoting which members of Y are actually in the
    # simplex

    # TODO right now assuming origin is not exactly on a vertex or edge
    y1 = Y[:, 0]
    y2 = Y[:, 1]
    y3 = Y[:, 2]

    # check vertices
    d1_y1 = 1
    d2_y1y2 = d1_y1 * (y1 - y2).dot(y1)
    d3_y1y3 = d1_y1 * (y1 - y3).dot(y1)

    if d2_y1y2 <= 0 and d3_y1y3 <= 0:
        return y1, y1, False

    d2_y2 = 1
    d1_y1y2 = d2_y2 * (y2 - y1).dot(y2)
    d3_y2y3 = d2_y2 * (y2 - y3).dot(y2)

    if d1_y1y2 <= 0 and d3_y2y3 <= 0:
        return y2, y2, False

    d3_y3 = 1
    d1_y1y3 = d3_y3 * (y3 - y1).dot(y3)
    d2_y2y3 = d3_y3 * (y3 - y2).dot(y3)

    if d1_y1y3 <= 0 and d2_y2y3 <= 0:
        return y3, y3, False

    # check edges
    d1_y1y2y3 = d2_y2y3 * (y2 - y1).dot(y2) + d3_y2y3 * (y2 - y1).dot(y3)
    d2_y1y2y3 = d1_y1y3 * (y1 - y2).dot(y1) + d3_y1y3 * (y1 - y2).dot(y3)
    d3_y1y2y3 = d1_y1y2 * (y1 - y3).dot(y1) + d2_y1y2 * (y1 - y3).dot(y2)

    if d2_y2y3 > 0 and d3_y2y3 > 0 and d1_y1y2y3 <= 0:
        dX = d2_y2y3 + d3_y2y3
        v = (d2_y2y3 * y2 + d3_y2y3 * y3) / dX
        return v, np.array([y2, y3]).T, False

    if d1_y1y3 > 0 and d3_y1y3 > 0 and d2_y1y2y3 <= 0:
        dX = d1_y1y3 + d3_y1y3
        v = (d1_y1y3 * y1 + d3_y1y3 * y3) / dX
        return v, np.array([y1, y3]).T, False

    if d1_y1y2 > 0 and d2_y1y2 > 0 and d3_y1y2y3 <= 0:
        dX = d1_y1y2 + d2_y1y2
        v = (d1_y1y2 * y1 + d2_y1y2 * y2) / dX
        return v, np.array([y1, y2]).T, False

    # final check equivalent to
    # d1_y1y2y3 > 0 and d2_y1y2y3 > 0 and d3_y1y2y3 > 0
    # want to be able to do bitwise enabling

    return np.zeros(2), Y, True
