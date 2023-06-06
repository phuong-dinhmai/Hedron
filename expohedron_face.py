import numpy as np
from scipy.linalg import orth

from helpers import majorized, invert_permutation, project_point_onto_plane

LOW_TOLERANCE = 1e-12


class Face:
    """
        Implements a face object of an expohedron.

        The face are all points in space such that
    """
    gamma: np.ndarray
    zone: np.ndarray
    splits: np.ndarray
    dim: int

    def __init__(self, gamma, zone, splits):
        n = len(gamma)
        assert np.all(np.sort(zone) == np.arange(0, n)), "zone must be a permutation"
        assert n == len(zone)
        assert issubclass(splits.dtype.type, np.integer), "splits must contain integer indices"
        assert np.all(0 <= splits) and np.all(splits < n), "The indices in split must be in the adequate range"
        assert len(splits) > 0, "All faces must have at least one split. The whole expohedron has exactly one split"

        self.gamma = np.sort(gamma)
        self.zone = zone
        self.splits = splits
        self.dim = n - len(splits)

    def contains(self, point: np.array, tolerance: float = 1e-12) -> bool:
        """
            Checks if a point is inside the face
        :param point:
        :return:
        :rtype: bool
        """
        # majorization condition
        maj = majorized(point, self.gamma)  
        # Check if the splits of `point` are a subset of `self.splits`
        face_condition: bool = len(np.setdiff1d(self.splits, np.where(np.abs(np.cumsum(point[self.zone]) - np.cumsum(self.gamma)) < tolerance)[0])) == 0  
        # print(maj, " ", face_condition)
        return maj and face_condition

    def equal(self, face: "Face") -> bool:
        """
            Checks if `self` is equal to `face`
        :param face:
        :return:
        :rtype: bool
        """
        return np.all(invert_permutation(self.zone)[self.splits] == invert_permutation(face.zone)[face.splits])


def find_face_subspace_without_parent(face: Face) -> np.ndarray:
    """
        Computes the smallest linear subspace in which `face` lies. Returns the orthonormal vectors to the subspace.
    :param face: The face whose subspace is to be computed
    :type face: Face
    :return: The normal vectors
    """
    n = len(face.gamma)  # The dimensionality of the space
    n_orth = n - face.dim  # The dimensionality of the orthogonal space
    A = np.zeros((n_orth, n))
    for j in np.arange(0, n_orth):
        i = face.splits[j]
        nu = np.ones(n)
        s1 = np.sum(face.gamma[0:i+1])
        s2 = np.sum(face.gamma[i+1:n])
        if s2 == 0:
            psi = 1
        else:
            psi = s1 / s2
        nu[i+1:n] = -psi
        A[j, :] = nu[invert_permutation(face.zone)]
    return orth(A.T)


def identify_face(gamma: np.ndarray, point_on_face: np.ndarray, tolerance: float = LOW_TOLERANCE) -> Face:
    """
        Computes the smallest face of the `gamma`-PBM-expohedron of which `point` is situated

    :param gamma: A vertex of the expohedron
    :type gamma: numpy.array
    :param point_on_face: The point to be examined
    :type point_on_face: numpy.array
    :param tolerance: The allowed tolerance
    :type: float, optional
    :return: The smallest face in which the intersection lies

    A tuple containing:
            (1) The permutation corresponding to an order-preserving zone
            (2) The indices of the splits and the dimensionality of the face
    :rtype: Face
    """
    n = len(gamma)
    assert n == len(point_on_face)

    splits = np.where(np.abs(np.cumsum(np.sort(gamma)) - np.cumsum(np.sort(point_on_face))) < tolerance)
    return Face(gamma, np.argsort(point_on_face), splits[0])


def post_correction(face, point_on_face):
    vertex_of_face = face.gamma[invert_permutation(face.zone)]
    face_subspace = find_face_subspace_without_parent(face)
    projected_point = project_point_onto_plane(point_on_face, face_subspace, face_subspace.T @ vertex_of_face)
    assert face.contains(projected_point), "There has been an error in the projection on a face's subspace"
    return projected_point

