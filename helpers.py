import numpy as np
from scipy.linalg import orth

LOW_TOLERANCE = 1e-12


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


def majorized(a: np.array, b: np.array, tolerance: float = LOW_TOLERANCE) -> bool:
    """
        Checks whether `a` is majorized by `b`: a<b

        :param a: The left hand side of the comparison a<b
        :type a: numpy.array
        :param b: The right hand side of the comparison a<b
        :type b: numpy.array
        :param tolerance: the tolerance that is allowed
        :type tolerance: float
        :return: `True` if a < b, false otherwise
        :rtype: bool
    """
    return np.all(np.cumsum(-np.sort(-a)) <= np.cumsum(-np.sort(-b)) + tolerance) and np.abs(np.sum(a) - np.sum(b)) < tolerance


def projection_matrix_on_subspace(U: np.ndarray):
    """
        Projection matrices onto the two sub-spaces spanned by the columns of U
    """
    matrix = U @ np.linalg.inv(U.T @ U) @ U.T
    return matrix


def orthogonal_complement(x: np.ndarray, normalize: bool = False, threshold: float = LOW_TOLERANCE):
    """
        Compute orthogonal complement of a matrix

        This works along axis zero, i.e. rank == column rank, or number of rows > column rank otherwise orthogonal complement is empty
        :param x: the matrix need to find the orthogonal complement
        :type x: numpy.ndarray
        :param normalize: equals True if the orthogonal vectors of complement need to standardize.
        :type normalize: bool
        :param tolerance: the tolerance that is allowed
        :type tolerance: float
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

    if normalize:
        k_oc = oc.shape[1]
        oc = oc.dot(np.linalg.inv(oc[:k_oc, :]))
    return oc


def direction_projecion_on_subspace(direction: np.ndarray, subspace_matrix: np.ndarray):
    """
        Compute vector projection in any subspace when the orthorgonal vector of the subspace is known

        :param direction: the 1D vector need to find the projection 
        :type a: numpy.array
        :param subspace_matrix: the matrix of othorgonal vector represent the subspace (each row is an vector)
        :type subspace_matrix: numpy.ndarray
        :return: The projection of the direction vector in the subspace 
        :rtype: numpy.ndarray (1D)   
    """
    orthogonal_direction = np.zeros(direction.shape)
    for orthogonal_vector in subspace_matrix:
        projection = np.dot(orthogonal_vector, direction) / np.dot(orthogonal_vector, orthogonal_vector) * orthogonal_vector
        orthogonal_direction += projection
    return orthogonal_direction


def intersect_vector_space(orthogonal_space_1: np.ndarray, orthogonal_space_2: np.ndarray):
    """
        Return basis vectors of intersection of 2 vector space
        (solution find in https://math.stackexchange.com/questions/767882/linear-algebra-vector-space-how-to-find-intersection-of-two-subspaces)

        :param orthogonal_space_1: Basis vectors of the first vector space (column is the basis vector)
        :param orthogonal_space_2: Basis vectors of the second vector space (column is the basis vector)
    """
    # TODO: why is it subtract but not substitution like in the solution?
    P_u = projection_matrix_on_subspace(orthogonal_space_1)
    P_v = projection_matrix_on_subspace(orthogonal_space_2)
    # print(P_u)
    # print(P_v)
    # print(orth(P_u - P_v))
    return orthogonal_complement(orth(P_u - P_v))
    # return orthogonal_complement(orth(P_u @ P_v - np.identity(orthogonal_space_1.shape[0])))


def find_face_intersection_bisection(gamma: np.ndarray, starting_point: np.ndarray, direction: np.ndarray, precision: float) -> np.ndarray:
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
            return lower_bound
        else:
            pass


if __name__ == "__main__":
    U = np.asarray([(1, 3, 4), (2, 5, 1)]).T
    V = np.asarray([(1, 1, 2), (2, 2, 1)]).T
    intersection_vector_space = intersect_vector_space(U, V)
