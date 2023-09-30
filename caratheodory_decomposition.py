import warnings

import numpy as np

from expohedron import Expohedron
from helpers import invert_permutation, project_on_vector_space


ULTRA_LOW_TOLERANCE = 5e-3
LOW_TOLERANCE = 1e-6
DEFAULT_TOLERANCE = 1e-9
HIGH_TOLERANCE = 1e-15
MAX_TOLERANCE = 2.220446049250313e-16  # 2.220446049250313e-16


def caratheodory_decomposition_pbm_gls(gamma: np.ndarray, point: np.ndarray, tol: float = HIGH_TOLERANCE):
    """
    Finds the Carathéodory decomposition of a point `x` in a PBM-expohedron with vertex `gamma` using the GLS method

    This is done using the GLS procedure
    A non-zero tolerance is necessary to account for numerical imprecision
    :param gamma: The initial vertex of the expohedron
    :type gamma: numpy.ndarray
    :param point: The point to decompose
    :type point: numpy.ndarray
    :param tol: The allowed tolerance
    :type tol: float
    :return: A tuple whose first element is a (matrix whose columns contain the vertices of the decomposition)
        and whose second element is a vector of convex coefficients. We have `point == vertices @ convex_coefficients`.
    :rtype:
    """
    hedron = Expohedron(gamma)
    assert hedron.contains(point), "`x` is not majorized by `gamma`. Only points inside the expohedron can be decomposed as a convex sum of its vertices."

    gamma = np.sort(gamma)  # We work with an increasing permutation of gamma
    n = len(point)
    vertices = np.zeros((n, n))  # Initializing the vertices (empty for now)
    convex_coefficients = np.zeros(n)  # Initializing the convex coefficients (empty for now)


    vertices[:, 0] = gamma[invert_permutation(np.argsort(point))]  # Initialize the initial vertex
    convex_coefficients[0] = 1
    x = point
    dim = hedron.identify_face(x).dim
    for i in np.arange(0, n-1):
        # pdb.set_trace()
        if np.all(np.abs(vertices @ convex_coefficients - point) < tol):
            return convex_coefficients, vertices
        v = vertices[:, i]
        approx_direction = x - v
        direction = project_on_vector_space(approx_direction, hedron.identify_face(x).find_face_subspace_without_parent().T)
        # direction = approx_direction
        intersection = hedron.find_face_intersection(v, direction)
        intersection = hedron.identify_face(intersection).post_correction(intersection)
        old_ci = convex_coefficients[i]
        convex_coefficients[i] = np.linalg.norm(intersection-x) / np.linalg.norm(intersection-v) * convex_coefficients[i]
        convex_coefficients[i+1] = old_ci - convex_coefficients[i]
        vertices[:, i+1] = gamma[invert_permutation(np.argsort(intersection))]  # Choose a vertex with the same ordering as `u`
        x = intersection  # `u`, the intersection, is the new point to decompose
        assert hedron.identify_face(x).dim < dim, "At each step, the dimensionality of the face must be reduced"
        dim = hedron.identify_face(x).dim
        if dim == 0:
            break  # if we finish on a vertex early, then we break

    # Remove vertices whose coefficients are below a certain threshold
    convex_coefficients[np.where(convex_coefficients < MAX_TOLERANCE)] = 0  # Affect 0 to negligible coefficients
    indices = np.squeeze(np.where(convex_coefficients != 0))
    convex_coefficients = convex_coefficients[indices]
    vertices = vertices[:, indices]

    # Final test
    assert np.abs(np.sum(convex_coefficients) - 1) < tol, "Convex coefficients must sum to 1"
    assert np.all(np.abs(vertices @ convex_coefficients - point) < ULTRA_LOW_TOLERANCE), "Carathéodory decomposition did not work"
    if np.any(np.abs(vertices @ convex_coefficients - point) > DEFAULT_TOLERANCE):
        warnings.warn("Beware, Carathéodory decomposition has reconstruction precision lower than " + str(DEFAULT_TOLERANCE), RuntimeWarning)

    return convex_coefficients, vertices
