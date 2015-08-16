import numpy as np

def ckmeans(data, n_clusters):
    if n_clusters > len(data):
        raise ValueError("Cannot generate more classes than there are data values")

    # if there's only one value, return it; there's no sensible way to split
    # it. This means that len(ckmeans([data], 2)) may not == 2. Is that OK?
    if len(set(data)) == 1:
        return [data]

    data.sort()
    n = len(data)

    # The table we're filling up with withinss values
    D = np.zeros((n_clusters, n))

    # XXX: not clear on what goes in B?
    B = np.zeros((n_clusters, n), dtype=np.int)

    for cluster in range(n_clusters):
        first_cluster_mean = data[0]

        for data_idx in range(max(cluster, 1), n):
            if cluster == 0:
                squared_difference = (data[data_idx] - first_cluster_mean) ** 2
                D[cluster][data_idx] = D[cluster][data_idx - 1] + (data_idx / (data_idx + 1)) * squared_difference

                new_sum = data_idx * first_cluster_mean + data[data_idx]
                first_cluster_mean = new_sum / (data_idx + 1)
            else:
                sum_squared_distances = 0
                mean_xj = 0

                for j in range(data_idx, cluster-1, -1):
                    sum_squared_distances += (data_idx - j) / (data_idx - j + 1) * (data[j] - mean_xj)**2

                    mean_xj = (data[j] + ((data_idx - j) * mean_xj)) / (data_idx - j + 1)

                    if j == data_idx:
                        D[cluster][data_idx] = sum_squared_distances
                        B[cluster][data_idx] = j
                        if j > 0:
                            D[cluster][data_idx] += D[cluster - 1][j - 1]
                    else:
                        if j == 0:
                            if sum_squared_distances <= D[cluster][data_idx]:
                                D[cluster][data_idx] = sum_squared_distances
                                B[cluster][data_idx] = j
                        elif sum_squared_distances + D[cluster-1][j-1] < D[cluster][data_idx]:
                            D[cluster][data_idx] = sum_squared_distances + D[cluster-1][j-1]
                            B[cluster][data_idx] = j

    clusters = []
    cluster_right = len(B[0]) - 1

    for cluster in range(len(B)-1, -1, -1):
        cluster_left = B[cluster][cluster_right]
        clusters.insert(0, data[cluster_left:cluster_right+1])

        if cluster > 0:
            cluster_right = cluster_left - 1

    return clusters

##
## HELPER CODE FOR TESTS
##

# partition recipe modified from
# http://wordaligned.org/articles/partitioning-with-python
from itertools import chain, combinations

def sliceable(xs):
    '''Return a sliceable version of the iterable xs.'''
    try:
        xs[:0]
        return xs
    except TypeError:
        return tuple(xs)

def partition_n(iterable, n):
    s = sliceable(iterable)
    l = len(s)
    b, mid, e = [0], list(range(1, l)), [l]
    getslice = s.__getitem__
    splits = (d for i in range(l) for d in combinations(mid, n-1))
    return [[s[sl] for sl in map(slice, chain(b, d), chain(d, e))]
            for d in splits]

def squared_distance(part):
    mean = sum(part)/len(part)
    return sum((x-mean)**2 for x in part)

# given a partition, return the sum of the squared distances of each part
def sum_of_squared_distances(partition):
    return sum(squared_distance(part) for part in partition)

# brute force the correct answer by testing every partition.
def min_squared_distance(data, n):
    return min((sum_of_squared_distances(partition), partition)
                for partition in partition_n(data, n))


if __name__ == "__main__":
    try:
        ckmeans([], 10)
        1/0
    except ValueError:
        pass

    tests = [
        (([1], 1),                    [[1]]),
        (([0,3,4], 2),                [[0], [3,4]]),
        (([-3,0,4], 2),               [[-3,0], [4]]),
        (([1,1,1,1], 1),              [[1,1,1,1]]),
        (([1,2,3], 3),                [[1], [2], [3]]),
        (([1,2,2,3], 3),              [[1], [2,2], [3]]),
        (([1,2,2,3,3], 3),            [[1], [2,2], [3,3]]),
        (([1,2,3,2,3], 3),            [[1], [2,2], [3,3]]),
        (([3,2,3,2,1], 3),            [[1], [2,2], [3,3]]),
        (([3,2,3,5,2,1], 3),          [[1,2,2], [3,3], [5]]),
        (([0,1,2,100,101,103], 2),    [[0,1,2], [100,101,103]]),
        (([0,1,2,50,100,101,103], 3), [[0,1,2], [50], [100,101,103]]),
        (([-1,2,-1,2,4,5,6,-1,2,-1], 3),
            [[-1, -1, -1, -1], [2, 2, 2], [4, 5, 6]]),
    ]

    for test in tests:
        args, expected = test
        result = ckmeans(*args)
        errormsg = "ckmeans({}) = {} != {}\n{} > {}".format(
                args, result, expected,
                sum_of_squared_distances(result),
                sum_of_squared_distances(expected))
        assert np.array_equal(result, expected), errormsg

    from hypothesis import given
    from hypothesis.strategies import lists, integers, just, tuples

    # can we set max higher? let's start with this number and see...
    for n in range(2,10):
        @given(lists(integers(), min_size=n, max_size=20).filter(lambda lst: len(set(lst)) > 1))
        def test_ckmeans(data):
            result = ckmeans(data, n)

            data.sort()
            squared_distance = sum_of_squared_distances(result)

            brute_distance, brute_result = min_squared_distance(data, n)

            error_message = "ckmeans({}, {}) = {} != {}; {} > {}".format(
                data, n, result, brute_result, squared_distance, brute_distance)
            assert squared_distance == brute_distance, error_message

        test_ckmeans()
