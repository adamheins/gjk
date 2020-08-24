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


def gjk(shape1, shape2, a):
    A = shape1.support(a) - shape2.support(-a)


# def dx(Y):
#     ncol = Y.shape[1]
#     if ncol == 1:
#         return 1
#     y0 = Y[:, 0]
#     ym = Y[:, -1]
#     return dx(Y[:, :-1]) * (y0 - ym).dot(y0)


# def dx(X, i):
#     ncol = X.shape[1]
#     if ncol == 1:
#         return 1
#
#     # remove ith column
#     xi = X[:, i]
#     Xi = np.delete(X, i, axis=1)
#
#     # arbitrary which column of Xi is chosen, so take the first one
#     x0 = Xi[:, 0]
#
#     # compute dx for adding the ith element back into the simplex
#     return np.sum([ dx(Xi, j) * (x0 - xi).dot(Xi[:, j]) for j in range(ncol-1)])


def dx(points, index_set, i):
    # base case: delta = 1 for singleton sets
    if np.sum(index_set) == 1:
        return 1

    # remove ith point
    index_set_no_i = np.copy(index_set)
    index_set_no_i[i] = False

    # arbitrary which remaining point is chosen, so take the first one
    nz_idx = np.nonzero(index_set_no_i)[0]
    k = nz_idx[0]

    dp = points[:, k] - points[:, i]

    # compute dx for adding the ith element back into the simplex
    return np.sum([ dx(points, index_set_i, j) * dp.dot(points[:, j]) for j in nz_idx ])


BITSETS = np.array([0b111, 0b011, 0b101, 0b110, 0b001, 0b010, 0b100])


def johnson5(points, this_bitset):
    Y = points

    # store deltas: row is indexed by the points in the set X; column indexed
    # by delta index
    # {y1} represented by 001, {y2} by 010, etc
    D = np.zeros((8, 3))

    # singleton sets are unity
    D[0b001, 0] = 1
    D[0b010, 1] = 1
    D[0b100, 2] = 1

    D[0b011, 1] = D[0b001, 0] * (Y[:, 0] - Y[:, 1]).dot(Y[:, 0])
    D[0b101, 2] = D[0b001, 0] * (Y[:, 0] - Y[:, 2]).dot(Y[:, 0])

    D[0b011, 0] = D[0b010, 1] * (Y[:, 1] - Y[:, 0]).dot(Y[:, 1])
    D[0b110, 2] = D[0b010, 1] * (Y[:, 1] - Y[:, 2]).dot(Y[:, 1])

    D[0b101, 0] = D[0b100, 2] * (Y[:, 2] - Y[:, 0]).dot(Y[:, 2])
    D[0b110, 1] = D[0b100, 2] * (Y[:, 2] - Y[:, 1]).dot(Y[:, 2])

    D[0b111, 0] = D[0b110, 1] * (Y[:, 1] - Y[:, 0]).dot(Y[:, 1]) \
                + D[0b110, 2] * (Y[:, 1] - Y[:, 0]).dot(Y[:, 2])

    D[0b111, 1] = D[0b101, 0] * (Y[:, 0] - Y[:, 1]).dot(Y[:, 0]) \
                + D[0b101, 2] * (Y[:, 0] - Y[:, 1]).dot(Y[:, 2])

    D[0b111, 2] = D[0b011, 0] * (Y[:, 0] - Y[:, 2]).dot(Y[:, 0]) \
                + D[0b011, 1] * (Y[:, 0] - Y[:, 2]).dot(Y[:, 1])

    for bitset in BITSETS:
        # only iterate bit sets that contain members of the current bitset
        if bitset & this_bitset != bitset:
            continue

        # check conditions for this bit set to contain the closest point
        contains_closest_pt = True
        for j in range(3):
            if get_bit(bitset, j) and D[bitset, j] <= 0:
                contains_closest_pt = False
            elif not get_bit(bitset, j) and D[set_bit(bitset, j), j] > 0:
                contains_closest_pt = False

        if contains_closest_pt:
            v = np.zeros(2)
            sv = 0
            for j in range(3):
                if get_bit(bitset, j):
                    v += D[bitset, j] * points[:, j]
                    sv += D[bitset, j]
            v = v / sv
            contains_origin = (bitset == 0b111)
            return v, bitset, contains_origin


def johnson4(points, index_set):
    contains_closest_pt = True
    for i, in_set in enumerate(index_set):
        index_superset = np.copy(index_set)
        index_superset[i] = True
        delta = dx(points, index_superset, i)

        # this simplex does not contain the closest point to origin if any
        # deltas with indices in the set are nonpositive, or any not in the set
        # are positive
        if in_set and delta <= 0:
            contains_closest_pt = False
        elif not in_set and delta > 0:
            contains_closest_pt = False

        # no point continuing if this simplex isn't the one
        if not contains_closest_pt:
            break

    if contains_closest_pt:
        v = deltas[index_set].dot(points[:, index_set]) / np.sum(deltas[index_set])
        contains_origin = (np.sum(index_set) == 3)
        return v, index_set, contains_origin

    if np.sum(index_set) == 1:
        return None, None, False

    for i in np.nonzero(index_set)[0]:
        index_subset = np.copy(index_set)
        index_subset[i] = False
        v, closet_simplex, contains_origin = johnson4(points, index_subset)
        if v is not None:
            return v, closet_simplex, contains_origin


def set_bit(b, i):
    ''' Set bit i in b to one. '''
    return b | (1 << i)


def unset_bit(b, i):
    ''' Set bit i in b to zero. '''
    return b & ~(1 << i)


def get_bit(b, i):
    return b & (1 << i)


def find_first_set(b):
    ''' Return the index of the first set bit (bit == 1) in b. '''
    return (b & -b).bit_length() - 1


def dx2(Y, b, i):
    ''' Calculate delta_i^{b} '''
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


B = np.array([
    [1, 1, 1],  # 7
    [0, 1, 1],  # 3
    [1, 0, 1],  # 5
    [1, 1, 0],  # 6
    [0, 0, 1],  # 1
    [0, 1, 0],  # 2
    [1, 0, 0]], # 4
    dtype=np.bool)




def johnson3(Y, b):
    # D[1 << 0, 0] = 1
    # D[1 << 1, 1] = 1
    # D[1 << 2, 2] = 1
    #
    # D[(1 << 0) + (1 << 1), 1] = D[1 << 0, 0] * (Y[:, 0] - Y[:, 1]).dot(Y[:, 0])
    # D[(1 << 0) + (1 << 2), 2] = D[1 << 0, 0] * (Y[:, 0] - Y[:, 2]).dot(Y[:, 0])
    #
    # D[(1 << 1) + (1 << 0), 0] = D[1 << 1, 1] * (Y[:, 1] - Y[:, 0]).dot(Y[:, 1])
    # D[(1 << 1) + (1 << 2), 2] = D[1 << 1, 1] * (Y[:, 1] - Y[:, 2]).dot(Y[:, 1])
    #
    # D[(1 << 2) + (1 << 0), 0] = D[1 << 2, 2] * (Y[:, 2] - Y[:, 0]).dot(Y[:, 2])
    # D[(1 << 2) + (1 << 1), 1] = D[1 << 2, 2] * (Y[:, 2] - Y[:, 1]).dot(Y[:, 2])
    #
    # D[(1 << 2) + (1 << 1) + (1 << 1), 0] = \
    #         D[(1 << 2) + (1 << 1), 1] * (Y[:, 1] - Y[:, 0]).dot(Y[:, 1]) \
    #       + D[(1 << 2) + (1 << 1), 2] * (Y[:, 1] - Y[:, 0]).dot(Y[:, 2])
    #
    # D[0b111, 1] = D[0b101, 0] * (Y[:, 0] - Y[:, 1]).dot(Y[:, 0]) \
    #             + D[0b101, 2] * (Y[:, 0] - Y[:, 1]).dot(Y[:, 2])
    #
    # D[0b111, 2] = D[0b011, 0] * (Y[:, 0] - Y[:, 2]).dot(Y[:, 0]) \
    #             + D[0b011, 1] * (Y[:, 0] - Y[:, 2]).dot(Y[:, 1])
    #
    #
    # D[0b111, 0] = D[0b110, 1] * (Y[:, 1] - Y[:, 0]).dot(Y[:, 1]) \
    #             + D[0b110, 2] * (Y[:, 1] - Y[:, 0]).dot(Y[:, 2])
    #
    #
    # # inputs
    # b = 0b111
    # i = 0
    #
    # # zero bit i in the index set
    # bi = b & ~(1 << i)
    #
    # # find the first index in the subsimplex
    # k = (bi & -bi).bit_length() - 1
    #
    # # recursive formula for \Delta_i^{b}
    # D[b, i] = np.sum([ D[bi, j] * (Y[:, k] - Y[:, i]).dot(Y[:, j]) for j in range(3) if bi & (1 << j) ])
    #
    # if D[b, 0] > 0 and D[b, 1] > 0 and D[b, 2] > 0:
    #     return

    D = np.array((8, 3))

    # singleton sets are unity
    D[0b001, 0] = 1
    D[0b010, 1] = 1
    D[0b100, 2] = 1

    # B = np.array([
    #     [1, 1, 1],  # 7
    #     [0, 1, 1],  # 3
    #     [1, 0, 1],  # 5
    #     [1, 1, 0],  # 6
    #     [0, 0, 1],  # 1
    #     [0, 1, 0],  # 2
    #     [1, 0, 0]], # 4
    #     dtype=np.bool)
    B = np.array([0b001, 0b010, 0b100, 0b011, 0b101, 0b110, 0b111])
    M = np.array([8, 4, 2, 1])

    # for idx in range(B.shape[0]):
    #     bidx = B[idx, :]
    #     for i in np.nonzero(bidx)[0]:
    #         bi = 

    # when calculating delta's, don't care about anything with a member we
    # don't have
    B_reduced = B[b | B == b]
    for idx in range(B_reduced[4:,:].shape[0]):
        bidx = B_reduced[4+idx, :]

        # iterate over members of bidx
        idx_in = (bidx & M) > 0
        for i in np.nonzero(idx_in)[0]:
            # zero bit i in the index set
            bi = bidx & ~(1 << i)

            # find the first index in the subsimplex
            k = (bi & -bi).bit_length() - 1

            D[bidx, i] = np.sum([ D[bi, j] * (Y[:, k] - Y[:, i]).dot(Y[:, j]) for j in range(3) if bi & (1 << j) ])

    IPython.embed()


    for i in range(B.shape[0]):
        b = B[i, :]
        # to calculate deltas, I need to iterate the current members of the set
        idx_in = (b & M) > 0
        idx_out = (b & M) <= 0
        if np.all(D[b, idx_in] > 0) and np.all(D[b, idx_out] < 0):
            return True


    # i = 0
    # bi = b & ~(1 << i)
    # if d2_y2y3 > 0 and d3_y2y3 > 0 and D[b, i] <= 0:
    #     dX = d2_y2y3 + d3_y2y3
    #     v = (d2_y2y3 * y2 + d3_y2y3 * y3) / dX
    #     return v, np.array([y2, y3]).T, False
    #
    # if d1_y1y3 > 0 and d3_y1y3 > 0 and d2_y1y2y3 <= 0:
    #     dX = d1_y1y3 + d3_y1y3
    #     v = (d1_y1y3 * y1 + d3_y1y3 * y3) / dX
    #     return v, np.array([y1, y3]).T, False
    #
    # if d1_y1y2 > 0 and d2_y1y2 > 0 and d3_y1y2y3 <= 0:
    #     dX = d1_y1y2 + d2_y1y2
    #     v = (d1_y1y2 * y1 + d2_y1y2 * y2) / dX
    #     return v, np.array([y1, y2]).T, False


def johnson2(Y, index_set):
    y1 = Y[:, 0]
    y2 = Y[:, 1]
    y3 = Y[:, 2]

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


def main():
    # S = np.array([[0, 0], [0, 1], [1, 0]]) + np.array([0, 0.5])
    # X = S.T
    S = np.random.random((2, 3)) * 10 - 5
    v1, X1, contains_origin1 = johnson(S)
    v2, X2, contains_origin2 = johnson5(S, 0b111)
    print(v1)
    print(v2)

    fig, ax = plt.subplots(1)
    ax.set_xlim([-5, 5])
    ax.set_ylim([-5, 5])
    ax.add_patch(plt.Polygon(S.T, color='k', fill=False, closed=True))
    plt.plot([0, v1[0]], [0, v1[1]], '-o', label='v1')
    plt.plot([0, v2[0]], [0, v2[1]], '-o', label='v2')
    plt.legend()
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
