import numpy as np
from scipy.linalg import orth

LOW_TOLERANCE = 1e-12
HIGH_TOLERANCE = 1e-10


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
    # print(np.cumsum(-np.sort(-majorized_vector)) <= np.cumsum(-np.sort(-majorizing_vector)) + tolerance)
    # print(np.abs(np.sum(majorized_vector) - np.sum(majorizing_vector)))
    return np.all(np.cumsum(-np.sort(-majorized_vector)) <= np.cumsum(-np.sort(-majorizing_vector)) + tolerance) \
           and np.abs(np.sum(majorized_vector) - np.sum(majorizing_vector)) < tolerance


def projection_matrix_on_subspace(U: np.ndarray):
    """
        Projection matrices onto the two sub-spaces spanned by the columns of U
    """
    return U @ np.linalg.inv(U.T @ U) @ U.T


def orthogonal_complement(x: np.ndarray, threshold: float = LOW_TOLERANCE):
    """
        Compute orthogonal complement of a matrix

        This works along axis zero, i.e. rank == column rank, or number of rows > column rank otherwise orthogonal complement is empty
        :param x: the matrix need to find the orthogonal complement
        :type x: numpy.ndarray
        :param threshold: the tolerance that is allowed
        :type threshold: float
        :return: List of orthogonal vector of the complement if have, None otherwise
        :rtype: np.ndarray
    """
    x = np.asarray(x)
    r, c = x.shape
    if r < c:
        import warnings
        warnings.warn('fewer rows than columns', UserWarning)

    # we assume svd is ordered by decreasing singular value, o.w. need sort
    s, v, d = np.linalg.svd(x)
    rank = (v > threshold).sum()

    oc = s[:, rank:]
    return oc


def project_vector_on_subspace(direction: np.ndarray, subspace_matrix: np.ndarray):
    """
        Compute vector projection in any subspace when the orthorgonal vector of the subspace is known

        :param direction: the 1D vector need to find the projection 
        :type direction: numpy.array
        :param subspace_matrix: the matrix of othorgonal vector represent the subspace (each row is an vector)
        :type subspace_matrix: numpy.ndarray
        :return: The projection of the direction vector in the subspace 
        :rtype: numpy.ndarray (1D)   
    """
    res = np.zeros(direction.shape)
    for orth_vector in subspace_matrix.T:
        param = np.dot(direction, orth_vector) / np.dot(orth_vector, orth_vector)
        res += param * orth_vector
    return res


def project_point_onto_plane(point: np.ndarray, A: np.ndarray, b: np.ndarray):
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
    return point - (A.T @ point - b) @ A.T


def intersect_vector_space(orthogonal_space_1: np.ndarray, orthogonal_space_2: np.ndarray):
    """
        Return basis vectors of intersection of 2 vector space
        (solution find in https://math.stackexchange.com/questions/25371/how-to-find-a-basis-for-the-intersection-of-two-vector-spaces-in-mathbbrn)

        :param orthogonal_space_1: Basis vectors of the first vector space (column is the basis vector)
        :param orthogonal_space_2: Basis vectors of the second vector space (column is the basis vector)
    """
    P_u = projection_matrix_on_subspace(orthogonal_space_1)
    P_v = projection_matrix_on_subspace(orthogonal_space_2)
    return orthogonal_complement(orth(P_u @ P_v - np.identity(orthogonal_space_1.shape[0])))


def find_face_intersection_bisection(gamma: np.ndarray, starting_point: np.ndarray,
                                     direction: np.ndarray, precision: float = LOW_TOLERANCE) -> np.ndarray:
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

    # direction = direction / np.linalg.norm(direction)  # normalize direction
    # 1. Find upper and lower bound
    k = 1

    while majorized((starting_point + k*direction) / np.sum((starting_point + k*direction)) * np.sum(gamma), gamma):
        # We make sure the tested point is in the hyperplane containing the expohedron
        # The division phase is for point projection to expohedron
        k *= 2

    upper_bound = k
    lower_bound = 0

    # 2. Do bisection
    nb_iterations = 0
    while True:
        nb_iterations += 1
        center = (upper_bound + lower_bound) / 2.0
        point = (starting_point + center*direction) / np.sum((starting_point + center*direction)) * np.sum(gamma)
        if majorized(point, gamma, tolerance=precision):  # project center on face's affine subspace
            lower_bound = center
        else:
            upper_bound = center
        if (upper_bound - lower_bound) < precision:
            return (starting_point + lower_bound*direction) / np.sum((starting_point + lower_bound*direction)) * np.sum(gamma)
        else:
            pass

    # upper_bound = (starting_point + k*direction) / np.sum((starting_point + k*direction)) * np.sum(gamma)
    # lower_bound = starting_point

    # # 2. Do bisection
    # nb_iterations = 0
    # while True:
    #     nb_iterations += 1
    #     center = (upper_bound + lower_bound) / 2
    #     if majorized(center, gamma, tolerance=precision):  # project center on face's affine subspace
    #         lower_bound = center
    #     else:
    #         upper_bound = center
    #     if np.all(np.abs(upper_bound - lower_bound) < precision):
    #         return lower_bound
    #     else:
    #         pass


if __name__ == "__main__":
    U = np.asarray([(1,1,0,-1), (0,1,3,1)]).T
    V = np.asarray([(0,-1,-2,1), (1,2,2,-2)]).T
    print(U.T @ V)
    intersection_vector_space = intersect_vector_space(U, V)
    print(intersection_vector_space)
