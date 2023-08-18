# a direct port of https://github.com/cran/Ckmeans.1d.dp/blob/f7f2920/src/Ckmeans.1d.dp.cpp
from typing import List, TypeVar

Vector = List[float]

T = TypeVar("T")
Matrix = List[List[T]]


def dissimilarity(j: int, i: int, sum_x: Vector, sum_x_sq: Vector) -> float:
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
    S: Matrix[float],
    J: Matrix[int],
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
def fill_dp_matrix(x: Vector, S: Matrix[float], J: Matrix[int]):
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


# TODO: loosen the "float" type to some sort of numeric type class. Going to
# start with strict types to match the source though
# TODO replicate their code for choosing an optimal number for k; we're just
# skipping it here
# https://github.com/cran/Ckmeans.1d.dp/blob/f7f2920/src/Ckmeans.1d.dp.cpp#L237-L242
def kmeans_1d_dp(x: Vector, k: int) -> Matrix[float]:
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

        # XXX: here's where I stopped for the day, translating
        # https://github.com/cran/Ckmeans.1d.dp/blob/f7f2920/src/Ckmeans.1d.dp.cpp#L271-L277
        #
        # I'm not quite sure:
        # - how to translate cluster_sorted, which passes an integer by
        # reference into `backtrack` - whether to implement `backtrack_L1` or
        # `backtrack`, or what the difference is - what `centers` and
        # `withinss` ought to be (they're arguments to the function in this
        # version, but what are they supposed to contain?)
        #
        # backtrack(x, J, cluster_sorted...
