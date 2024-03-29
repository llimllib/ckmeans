import numpy as np


def ssq(j, i, sum_x, sum_x_sq):
    if j > 0:
        muji = (sum_x[i] - sum_x[j - 1]) / (i - j + 1)
        sji = sum_x_sq[i] - sum_x_sq[j - 1] - (i - j + 1) * muji**2
    else:
        sji = sum_x_sq[i] - sum_x[i] ** 2 / (i + 1)

    return 0 if sji < 0 else sji


def fill_row_k(imin, imax, k, S, J, sum_x, sum_x_sq, N):
    if imin > imax:
        return

    i = (imin + imax) // 2
    S[k][i] = S[k - 1][i - 1]
    J[k][i] = i

    jlow = k

    if imin > k:
        jlow = int(max(jlow, J[k][imin - 1]))
    jlow = int(max(jlow, J[k - 1][i]))

    jhigh = i - 1
    if imax < N - 1:
        jhigh = int(min(jhigh, J[k][imax + 1]))

    for j in range(jhigh, jlow - 1, -1):
        sji = ssq(j, i, sum_x, sum_x_sq)

        if sji + S[k - 1][jlow - 1] >= S[k][i]:
            break

        # Examine the lower bound of the cluster border
        # compute s(jlow, i)
        sjlowi = ssq(jlow, i, sum_x, sum_x_sq)

        SSQ_jlow = sjlowi + S[k - 1][jlow - 1]

        if SSQ_jlow < S[k][i]:
            S[k][i] = SSQ_jlow
            J[k][i] = jlow

        jlow += 1

        SSQ_j = sji + S[k - 1][j - 1]
        if SSQ_j < S[k][i]:
            S[k][i] = SSQ_j
            J[k][i] = j

    fill_row_k(imin, i - 1, k, S, J, sum_x, sum_x_sq, N)
    fill_row_k(i + 1, imax, k, S, J, sum_x, sum_x_sq, N)


def fill_dp_matrix(data, S, J, K, N):
    sum_x = np.zeros(N, dtype=np.float_)
    sum_x_sq = np.zeros(N, dtype=np.float_)

    # median. used to shift the values of x to improve numerical stability
    shift = data[N // 2]

    for i in range(N):
        if i == 0:
            sum_x[0] = data[0] - shift
            sum_x_sq[0] = (data[0] - shift) ** 2
        else:
            sum_x[i] = sum_x[i - 1] + data[i] - shift
            sum_x_sq[i] = sum_x_sq[i - 1] + (data[i] - shift) ** 2

        S[0][i] = ssq(0, i, sum_x, sum_x_sq)
        J[0][i] = 0

    for k in range(1, K):
        if k < K - 1:
            imin = max(1, k)
        else:
            imin = N - 1

        fill_row_k(imin, N - 1, k, S, J, sum_x, sum_x_sq, N)


def ckmeans(data, n_clusters):
    if n_clusters <= 0:
        raise ValueError("Cannot classify into 0 or less clusters")
    if n_clusters > len(data):
        raise ValueError("Cannot generate more classes than there are data values")

    # if there's only one value, return it; there's no sensible way to split
    # it. This means that len(ckmeans([data], 2)) may not == 2. Is that OK?
    unique = len(set(data))
    if unique == 1:
        return [data]

    data.sort()
    n = len(data)

    S = np.zeros((n_clusters, n), dtype=np.float_)

    J = np.zeros((n_clusters, n), dtype=np.uint64)

    fill_dp_matrix(data, S, J, n_clusters, n)

    clusters = []
    cluster_right = n - 1

    for cluster in range(n_clusters - 1, -1, -1):
        cluster_left = int(J[cluster][cluster_right])
        clusters.append(data[cluster_left : cluster_right + 1])

        if cluster > 0:
            cluster_right = cluster_left - 1

    return list(reversed(clusters))


##
## HELPER CODE FOR TESTS
##

# partition recipe modified from
# http://wordaligned.org/articles/partitioning-with-python
from itertools import chain, combinations


def sliceable(xs):
    """Return a sliceable version of the iterable xs."""
    try:
        xs[:0]
        return xs
    except TypeError:
        return tuple(xs)


def partition_n(iterable, n):
    s = sliceable(iterable)
    l = len(s)
    b, mid, e = [0], list(range(1, l)), [l]
    splits = (d for _ in range(l) for d in combinations(mid, n - 1))
    return [[s[sl] for sl in map(slice, chain(b, d), chain(d, e))] for d in splits]


def squared_distance(part):
    mean = sum(part) / len(part)
    return sum((x - mean) ** 2 for x in part)


# given a partition, return the sum of the squared distances of each part
def sum_of_squared_distances(partition):
    return sum(squared_distance(part) for part in partition)


# brute force the correct answer by testing every partition.
def min_squared_distance(data, n):
    return min(
        (sum_of_squared_distances(partition), partition)
        for partition in partition_n(data, n)
    )


if __name__ == "__main__":

    def array_equal(a, b):
        if type(a) != type(b):
            return False
        if type(a) != list:
            return a == b
        if len(a) != len(b):
            return False
        for aa, bb in zip(a, b):
            if not array_equal(aa, bb):
                return False
        return True

    try:
        ckmeans([], 10)
        1 / 0
    except ValueError:
        pass

    tests = [
        (([1], 1), [[1]]),
        (([0, 3, 4], 2), [[0], [3, 4]]),
        (([-3, 0, 4], 2), [[-3, 0], [4]]),
        (([1, 1, 1, 1], 1), [[1, 1, 1, 1]]),
        (([1, 2, 3], 3), [[1], [2], [3]]),
        (([1, 2, 2, 3], 3), [[1], [2, 2], [3]]),
        (([1, 2, 2, 3, 3], 3), [[1], [2, 2], [3, 3]]),
        (([1, 2, 3, 2, 3], 3), [[1], [2, 2], [3, 3]]),
        (([3, 2, 3, 2, 1], 3), [[1], [2, 2], [3, 3]]),
        (([3, 2, 3, 5, 2, 1], 3), [[1, 2, 2], [3, 3], [5]]),
        (([0, 1, 2, 100, 101, 103], 2), [[0, 1, 2], [100, 101, 103]]),
        (([0, 1, 2, 50, 100, 101, 103], 3), [[0, 1, 2], [50], [100, 101, 103]]),
        (
            ([-1, 2, -1, 2, 4, 5, 6, -1, 2, -1], 3),
            [[-1, -1, -1, -1], [2, 2, 2], [4, 5, 6]],
        ),
    ]

    for test in tests:
        args, expected = test
        try:
            result = ckmeans(*args)
        except:
            print("✗ {}, {}".format(args[0], args[1], result))
            raise
        errormsg = "✗ ckmeans({}) = {} != {}\n{} > {}".format(
            args,
            result,
            expected,
            sum_of_squared_distances(result),
            sum_of_squared_distances(expected),
        )
        assert array_equal(result, expected), errormsg
        print("✓ {}".format(result))

    from hypothesis import given
    from hypothesis.strategies import lists, integers
    from numpy.testing import assert_approx_equal

    # can we set max higher? let's start with this number and see...
    for n in range(2, 10):

        @given(
            lists(
                integers(min_value=-100, max_value=100), min_size=n, max_size=20
            ).filter(lambda lst: len(set(lst)) > 1)
        )
        def test_ckmeans(data):
            result = ckmeans(data, n)

            data.sort()
            squared_distance = sum_of_squared_distances(result)

            brute_distance, brute_result = min_squared_distance(data, n)

            error_message = "ckmeans({}, {}) = {} != {}; {} > {}".format(
                data, n, result, brute_result, squared_distance, brute_distance
            )

            assert_approx_equal(squared_distance, brute_distance, err_msg=error_message)

        test_ckmeans()
