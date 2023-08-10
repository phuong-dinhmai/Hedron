import numpy as np
from scipy.linalg import orth, null_space
import time

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
    # print(np.cumsum(-np.sort(-majorized_vector)) - np.cumsum(-np.sort(-majorizing_vector)))
    # print(np.abs(np.sum(majorized_vector) - np.sum(majorizing_vector)))
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


def find_face_intersection_bisection(gamma: np.ndarray, starting_point: np.ndarray,
                                     direction: np.ndarray, precision: float = DEFAULT_TOLERANCE) -> np.ndarray:
    """
        Executes a bisection search in the PBM-expohedron using the majorization criterion.

        It finds the intersection of a half-line starting at `starting_point` in the direction `direction` with the border of the expohedron defined by `gamma`.
    :param gamma: Any vertex of the PBM-expohedron
    :type gamma: numpy.ndarray
    :param starting_point: The starting point of the half-line
    :type starting_point: numpy.ndarray
    :param direction: The direction of the half-line
    :type direction: numpy.ndarray
    :param precision: The presicion required for termination of bisection
    :type precision: float, optional
    :return: The intersection of the expohedron's boundary with the half-line
    :rtype numpy.ndarray
    """
    # 0. Input checks
    n = len(gamma)
    assert n == len(starting_point), "`starting_point` does not have the same length as `gamma`."
    assert n == len(direction), "`direction` does not have the same length as `gamma`."
    assert majorized(starting_point, gamma), "`starting_point` needs to be majorized by `gamma`. Check your inputs or decrease majorization tolerance."

    # If point and direction are in the same zone
    if np.all(np.argsort(starting_point) == np.argsort(direction)):
        zone = np.argsort(starting_point)
        Gk = np.cumsum(np.sort(gamma))
        Sk = np.cumsum(starting_point[zone])
        Dk = np.cumsum(direction[zone])
        # eliminate the coordinates where Dk is zero; no information about Lambda can be obtained from them. todo: refer to a proof in paper

        # Lambda = min((Gk - Sk)[0:(n - 1)] / Dk[0:(n - 1)])
        indices = np.where(np.abs(Dk) > 1e-12)
        bounds = (Gk - Sk)[indices] / Dk[indices]
        Lambda = min(bounds[np.where(bounds >= 0)], default=0)

        return starting_point + Lambda * direction
    else:
        # Start binary search
        # 1. Find upper and lower bound
        k = 1

        while majorized((starting_point + k*direction) / np.sum((starting_point + k*direction)) * np.sum(gamma), gamma):
            # We make sure the tested point is in the hyperplane containing the expohedron
            # The division phase is for point projection to expohedron
            k *= 2
        # print(k)

        upper_bound = (starting_point + k*direction) / np.sum((starting_point + k*direction)) * np.sum(gamma)
        lower_bound = starting_point

        # 2. Do bisection
        nb_iterations = 0
        while True:
            nb_iterations += 1
            center = (upper_bound + lower_bound) / 2
            if majorized(center, gamma, tolerance=precision):  # project center on face's affine subspace
                lower_bound = center
            else:
                upper_bound = center
            if np.all(np.abs(upper_bound - lower_bound) < precision):
                # print(nb_iterations)
                return lower_bound 
            else:
                pass


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

