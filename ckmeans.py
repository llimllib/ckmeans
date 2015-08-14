import numpy as np

def ckmeans(data, n_clusters):
    if n_clusters > len(data):
        raise ValueError("Cannot generate more classes than there are data values")

    n = len(data)

    # XXX: probably more efficient to leave this as an iterator, but let's get
    # it correct first
    sorted_data = list(sorted(data))
    unique_count = len(set(sorted_data))

    if unique_count == 1:
        return [sorted_data]

    matrix = np.zeros((n_clusters, n))
    backtrack_matrix = np.zeros((n_clusters, n), dtype=np.int)

    for cluster in range(n_clusters):
        first_cluster_mean = sorted_data[0]

        for sorted_idx in range(max(cluster, 1), n):
            if cluster == 0:
                squared_difference = (sorted_data[sorted_idx] - first_cluster_mean) ** 2
                matrix[cluster][sorted_idx] = matrix[cluster][sorted_idx - 1] + ((sorted_idx - 1) / sorted_idx) * squared_difference

                new_sum = sorted_idx * first_cluster_mean + sorted_data[sorted_idx]
                first_cluster_mean = new_sum / sorted_idx
            else:
                sum_squared_distances = 0
                mean_xj = 0

                for j in range(sorted_idx, cluster-1, -1):
                    sum_squared_distances += (sorted_idx - j) / (sorted_idx - j + 1) * (sorted_data[j] - mean_xj)**2

                    mean_xj = (sorted_data[j] + ((sorted_idx - j) * mean_xj)) / (sorted_idx - j + 1)

                    if j == sorted_idx:
                        matrix[cluster][sorted_idx] = sum_squared_distances
                        backtrack_matrix[cluster][sorted_idx] = j
                        if j > 0:
                            matrix[cluster][sorted_idx] += matrix[cluster - 1][j - 1]
                    else:
                        if j == 0:
                            if sum_squared_distances <= matrix[cluster][sorted_idx]:
                                matrix[cluster][sorted_idx] = sum_squared_distances
                                backtrack_matrix[cluster][sorted_idx] = j
                        elif sum_squared_distances + matrix[cluster-1][j-1] < matrix[cluster][sorted_idx]:
                            matrix[cluster][sorted_idx] = sum_squared_distances + matrix[cluster-1][j-1]
                            backtrack_matrix[cluster][sorted_idx] = j

    clusters = []
    cluster_right = len(backtrack_matrix[0]) - 1

    for cluster in range(len(backtrack_matrix)-1, -1, -1):
        cluster_left = backtrack_matrix[cluster][cluster_right]
        clusters.insert(0, sorted_data[cluster_left:cluster_right+1])

        if cluster > 0:
            cluster_right = cluster_left - 1

    return clusters

if __name__ == "__main__":
    try:
        ckmeans([], 10)
        1/0
    except ValueError:
        pass

    tests = [
        (([1], 1), [[1]]),
        (([1,1,1,1], 1), [[1,1,1,1]]),
        (([1,2,3], 3), [[1], [2], [3]]),
        (([1,2,2,3], 3), [[1,2], [2], [3]]),
        (([1,2,2,3,3], 3), [[1,2], [2], [3,3]]),
        (([1,2,3,2,3], 3), [[1,2], [2], [3,3]]),
        (([3,2,3,2,1], 3), [[1,2], [2], [3,3]]),
        (([3,2,3,5,2,1], 3), [[1,2,2], [3,3], [5]]),
        (([0,1,2,100,101,103], 2), [[0,1,2], [100,101,103]]),
        (([0,1,2,50,100,101,103], 3), [[0, 1, 2], [50], [100, 101, 103]]),
        (([-1,2,-1,2,4,5,6,-1,2,-1], 3), [[-1, -1, -1, -1], [2, 2, 2], [4, 5, 6]]),
    ]

    for test in tests:
        args, expected = test
        result = ckmeans(*args)
        assert np.array_equal(result, expected), "ckmeans({}) = {} != {}".format(args, result, expected)
