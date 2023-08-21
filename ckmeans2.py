# a direct port of https://github.com/cran/Ckmeans.1d.dp/blob/f7f2920/src/Ckmeans.1d.dp.cpp
from typing import List, Any

# Whatever numeric data type you provide to ckmeans has to implement:
# - addition
# - subtraction
# - exponentiation
# - sorting
#
# Also you need to be able to get a zero value from the type, which
# I don't know of any way to type in python; whatever type I used
# I was stymied by having to return a literal zero, which python sees
# (correctly) as an integer and maybe not whatever type (float, complex, etc)
# you passed in
#
# Here are some generic types I tried:
#
# from numbers import Real
# T = TypeVar("T", int, Real)
# T = TypeVar("T", bound=int)
# T = TypeVar("T", bound=Real)
#
# but none of them worked adequately. Returning "Any" because I can't phrase
# zero as a generic type sucks
Vector = List
Matrix = List[List]


def dissimilarity(j: int, i: int, sum_x: List, sum_x_sq: List) -> Any:
    if j > 0:
        muji = (sum_x[i] - sum_x[j - 1]) / (i - j + 1)
        sji = sum_x_sq[i] - sum_x_sq[j - 1] - (i - j + 1) * muji**2
    else:
        sji = sum_x_sq[i] - sum_x[i] ** 2 / (i + 1)

    return 0 if sji < 0 else sji


def fill_row_q(
    imin: int,
    imax: int,
    q: int,
    S: Matrix,
    J: List[List[int]],
    sum_x: Vector,
    sum_x_sq: Vector,
):
    for i in range(imin, imax + 1):
        S[q][i] = S[q - 1][i - 1]
        J[q][i] = i
        jmin = max(q, J[q - 1][i])
        for j in range(i - 1, jmin - 1, -1):
            sj = S[q - 1][j - 1] + dissimilarity(j, i, sum_x, sum_x_sq)
            if sj < S[q][i]:
                S[q][i] = sj
                J[q][i] = j


# fill the dynamic programming matrix
#
# x: One dimension vector to be clustered, must be sorted (in any order).
# S: K x N matrix. S[q][i] is the sum of squares of the distance from each x[i]
#    to its cluster mean when there are exactly x[i] is the last point in
#    cluster q
# J: K x N backtrack matrix
def fill_dp_matrix(x: Vector, S: Matrix, J: List[List[int]]):
    K = len(S)
    N = len(S[0])

    # median. used to shift the values of x to improve numerical stability
    shift = x[N // 2]

    sum_x = [0.0] * N
    sum_x_sq = [0.0] * N

    sum_x[0] = x[0] - shift
    sum_x_sq[0] = sum_x[0] ** 2

    for i in range(1, N):
        sum_x[i] = sum_x[i - 1] + x[i] - shift
        sum_x_sq[i] = sum_x_sq[i - 1] + (x[i] - shift) ** 2

        # initialize for q = 0
        S[0][i] = dissimilarity(0, i, sum_x, sum_x_sq)
        J[0][i] = 0

    for q in range(1, K):
        imin = max(1, q) if q < K - 1 else N - 1

        # there are linear and log-linear methods in the source:
        # https://github.com/cran/Ckmeans.1d.dp/blob/f7f2920/src/dynamic_prog.cpp#L115-L118
        fill_row_q(imin, N - 1, q, S, J, sum_x, sum_x_sq)


# use the dynamic programming matrix to generate the clusters.
#
# - In the text, they are also returning each cluster's median and variance;
#   I've omitted that here and you can calculate it yourself if you'd like
#   https://github.com/cran/Ckmeans.1d.dp/blob/f7f2920/src/dynamic_prog.cpp#L176-L184
# - Given a large data array, we'll be allocating a lot here; it may make sense
#   to investigate better-performing approaches
def backtrack(x: Vector, J: List[List[int]]) -> Matrix:
    K = len(J)
    N = len(J[0])
    cluster_right = N - 1
    clusters = []

    for q in range(K - 1, -1, -1):
        cluster_left = J[q][cluster_right]
        clusters.append(x[cluster_left : cluster_right + 1])

        if q > 0:
            cluster_right = cluster_left - 1

    return clusters


# TODO replicate their code for choosing an optimal number for k; we're just
# skipping it here
# https://github.com/cran/Ckmeans.1d.dp/blob/f7f2920/src/Ckmeans.1d.dp.cpp#L237-L242
#
# ckmeans will return a Matrix of whatever type x is of, but actually typing it
# that way has eluded my ability
def ckmeans(x: Vector, k: int) -> Matrix:
    x.sort()

    # https://github.com/cran/Ckmeans.1d.dp/blob/f7f2920/src/Ckmeans.1d.dp.cpp#L199
    # I'm not sure what unequal weights are, so skip it for now

    # if x is large, this is probably not the fastest option. Benchmark.
    n_unique = len(set(x))

    k_max = n_unique if n_unique < k else k

    if n_unique > 1:
        # S and J are K x N matrices
        S = [[0.0] * len(x) for _ in range(k_max)]
        J = [[0] * len(x) for _ in range(k_max)]

        # again, we lack the not equally weighted branch
        # https://github.com/cran/Ckmeans.1d.dp/blob/f7f2920/src/Ckmeans.1d.dp.cpp#L229

        fill_dp_matrix(x, S, J)

        clusters = backtrack(x, J)

        return list(reversed(clusters))
    else:
        return [x]
