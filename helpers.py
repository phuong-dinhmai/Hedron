import numpy as np
from scipy.linalg import orth, null_space, norm
import time
import cvxpy as cp

ULTRA_LOW_TOLERANCE = 5e-3
LOW_TOLERANCE = 1e-6
DEFAULT_TOLERANCE = 1e-9
HIGH_TOLERANCE = 1e-13
MAX_TOLERANCE = 2.220446049250313e-16 


def check_orthogonal_vector(vector: np.ndarray, basis_space: np.ndarray):
    """
        Checks if a vector is orthogonal with vector space
    :param vector: a vector is to be checked
    :type vector: np.ndarray
    :param basis_space: The vector space need to check
    :type basis_space: np.ndarray
    :return: True if they are orthogonal, False otherwise
    :rtype: bool
    """
    for orth_vector in basis_space.T:
        if np.dot(vector, orth_vector) > LOW_TOLERANCE:
            return False
    return True


def check_spanned_vector(vector: np.ndarray, basis_space: np.ndarray):
    """
        Checks if a vector is orthogonal with vector space
    :param vector: a vector is to be checked
    :type vector: np.ndarray
    :param basis_space: The vector space need to check
    :type basis_space: np.ndarray
    :return: True if the vector belongs to the span of the basis vector space, False otherwise
    :rtype: bool
    """
    m_vector = vector.reshape((vector.shape[0], 1))
    return np.linalg.matrix_rank(np.concatenate((basis_space, m_vector), axis=1)) == np.linalg.matrix_rank(basis_space)


def is_ranking(ranking: np.ndarray, size: int = None):
    """
        Checks if `ranking` is a ranking, i.e. performs simple asserts to check if `ranking` is a valid ranking
    :param ranking: a ranking whose formatting is to be checked
    :type ranking: np.ndarray
    :param size: The size of the ranking, optional
    :type size: int
    """
    if size is not None:
        assert len(ranking) == size, "`ranking` must be of size " + str(size)
    else:
        size = len(ranking)
    assert np.all(np.sort(ranking) == np.arange(0, size)), "`ranking` is not a permutation of {0,...,n-1}"
    return True


def invert_permutation(permutation):
    """
    Inverts a permutation: If `permutation[i]==j`, then `invert_permutation(permutation)[j]==i`.

    :param permutation: A permutation represented as an array containing the integers 0 to n
    :type permutation: numpy.ndarray
    :return: The inverse permutation
    :rtype: numpy.ndarray
    """
    return np.argsort(permutation)


def majorized(majorized_vector: np.array, majorizing_vector: np.array, tolerance: float = LOW_TOLERANCE) -> bool:
    """
        Checks whether `a` is majorized by `b`: a<b

        :param majorized_vector: The left hand side of the comparison a<b
        :type majorized_vector: numpy.array
        :param majorizing_vector: The right hand side of the comparison a<b
        :type majorizing_vector: numpy.array
        :param tolerance: the tolerance that is allowed
        :type tolerance: float
        :return: `True` if a < b, false otherwise
        :rtype: bool
    """
    return np.all(np.cumsum(-np.sort(-majorized_vector)) <= np.cumsum(-np.sort(-majorizing_vector)) + tolerance) \
           and np.abs(np.sum(majorized_vector) - np.sum(majorizing_vector)) < tolerance


def projection_matrix_on_subspace(U: np.ndarray):
    """
        Projection matrices onto the two sub-spaces spanned by the columns of U
    """
    return U @ np.linalg.inv(U.T @ U) @ U.T


def project_vector_on_subspace(direction: np.ndarray, subspace_matrix: np.ndarray):
    """
        Compute vector projection in any subspace when the orthorgonal vector of the subspace is known

        :param direction: the 1D vector need to find the projection 
        :type direction: numpy.array
        :param subspace_matrix: the matrix of othorgonal vector represent the subspace (each column is an vector)
        :type subspace_matrix: numpy.ndarray
        :return: The projection of the direction vector in the subspace 
        :rtype: numpy.ndarray (1D)   
    """
    param = projection_matrix_on_subspace(subspace_matrix)
    res = direction @ param
    # subspace_matrix = orth(subspace_matrix)
    # param = direction @ subspace_matrix
    # res = (param * subspace_matrix).sum(axis=1)
    # res[np.abs(res) < HIGH_TOLERANCE] = 0
    return res


def project_on_vector_space(point_to_project: np.ndarray, normal_vectors: np.ndarray) -> np.array:
    """
        Given an (m x n)-matrix `A`, this function computes the projection of `point_to_project` onto the linear subspace S = {x in R^n | Ax = 0}.

        The rows of matrix `A` are vectors orthogonal to the subspace
    :param point_to_project: The point to project on the linear subspace
    :type point_to_project: numpy.ndarray
    :param normal_vectors: A matrix whose rows are normal vectors to the subspace
    :return:
    """
    n = normal_vectors.shape[1]
    assert n == len(point_to_project), "The normal vectors must have the same dimension as the `point_to_project`"
    # print(point_to_project)
    P = orth(normal_vectors.T)
    # print(P)
    # P = normal_vectors.T
    return point_to_project - P @ (P.T @ point_to_project)


def project_point_on_plane(point: np.ndarray, A: np.ndarray, b: np.ndarray):
    """
        Compute point projection in any subspace when the multiply matrix and constants is know

        :param point: the coordination of the point need to find the projection 
        :type point: numpy.array
        :param A: the matrix of orthogonal vector represent the subspace (each row is an vector)
        :type A: numpy.ndarray
        :param b: the matrix of orthogonal vector represent the subspace (each row is an vector)
        :type b: numpy.ndarray
        :return: The projection of the point in the subspace Ax=b
        :rtype: numpy.ndarray (1D)   
    """
    T = np.concatenate((A, [b]), axis=0)
    T = orth(T)
    _A = T[:-1, :]
    _b = T[-1, :]
    P = _A @ np.linalg.inv(_A.T @ _A)
    return (np.eye(_A.shape[0]) - P @ _A.T) @ point + P @ _b
    # P = A @ np.linalg.inv(A.T @ A)
    # return (np.eye(A.shape[0]) - P @ A.T) @ point + P @ b


def intersect_vector_space(orthogonal_space_1: np.ndarray, orthogonal_space_2: np.ndarray):
    """
        Return basis vectors of intersection of 2 vector space
        (solution find in https://math.stackexchange.com/questions/25371/how-to-find-a-basis-for-the-intersection-of-two-vector-spaces-in-mathbbrn)

        :param orthogonal_space_1: Basis vectors of the first vector space (column is the basis vector)
        :param orthogonal_space_2: Basis vectors of the second vector space (column is the basis vector)
    """
    A = np.concatenate((orthogonal_space_1, -orthogonal_space_2), axis=1)
    A_comple = null_space(A, HIGH_TOLERANCE)
    if A_comple.shape[1] == 0:
        return A_comple
    return orthogonal_space_1 @ A_comple[:orthogonal_space_1.shape[1], :]
    # return orth(orthogonal_space_2 @ A_comple[orthogonal_space_1.shape[1]:, :])


class Objective:
    def __init__(self, relevance_score, group_fairness, group_masking, gamma):
        self.relevance_score = relevance_score
        self.target_group_unfairness = group_fairness
        self.group_masking = group_masking
        self.pbm = gamma

    def utils(self, point):
        return self.relevance_score @ point

    def unfairness(self, point):
        return np.sum((self.group_masking.T @ point - self.target_group_unfairness) ** 2)

    def objectives(self, point):
        return self.utils(point), self.unfairness(point)

    def convex_constraints_prob(self, group_unfairness):
        n_doc, _ = self.group_masking.shape
        gamma_sum = np.cumsum(self.pbm)
        vars = cp.Variable(n_doc)
        constrs = [cp.sum_largest(vars, i) <= gamma_sum[i-1] for i in range(1, n_doc)]
        constrs.append(self.group_masking.T @ vars == group_unfairness)
        obj_func = cp.Maximize(cp.sum(self.relevance_score.T @ vars))
        prob = cp.Problem(obj_func, constrs)
        prob.solve(verbose=False)  # Returns the optimal value.
        # print("status:", prob.status)
        if prob.status == cp.OPTIMAL:
            return vars.value
        return None


if __name__ == "__main__":
    # U = np.asarray([(1,1,0,-1), (0,1,3,1)])
    # V = np.asarray([(0,-1,-2,1), (1,2,2,-2)])
    # # intersection_vector_space = intersect_vector_space(U.T, V.T)
    # # print(intersection_vector_space)
    # start = time.time()
    # project_vector_on_subspace(U[0], V.T)
    # end = time.time()
    # print(end - start)
    # _V = null_space(V)
    # start = time.time()
    # project_on_vector_space(U[0], _V.T)
    # end = time.time()
    # print(end - start)
    matrix = np.asarray([[1, 1, 1, 1], [1, 1, 1, 0]])
    point = np.asarray([0, -0.025, 0.025, 0])
    print(project_on_vector_space(point, matrix))

