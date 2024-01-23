"""
    Copyright © 2022 Naver Corporation. All rights reserved.

    This file regroups a number of function about the expohedron
"""

import numpy as np
from scipy.linalg import orth

from main.helpers import majorized, invert_permutation, project_on_vector_space

ULTRA_LOW_TOLERANCE = 5e-3
LOW_TOLERANCE = 1e-6
DEFAULT_TOLERANCE = 1e-9
HIGH_TOLERANCE = 1e-15
MAX_TOLERANCE = 2.220446049250313e-16  # 2.220446049250313e-16

    
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
        maj = majorized(point, self.gamma)  # majorization condition
        face_condition: bool = len(np.setdiff1d(self.splits, np.where(np.abs(np.cumsum(point[self.zone]) - np.cumsum(self.gamma)) < tolerance)[0])) == 0  # Check if the splits
        # of `point` are a subset of `self.splits`
        return maj and face_condition

    def equal(self, face: "Face") -> bool:
        """
            Checks if `self` is equal to `face`
        :param face:
        :return:
        :rtype: bool
        """
        return np.all(invert_permutation(self.zone)[self.splits] == invert_permutation(face.zone)[face.splits])

    def post_correction(self, point: np.ndarray, tolerance: float = DEFAULT_TOLERANCE) -> np.ndarray:
        """
            Projects a point `point` onto the smallest *affine* subspace that contains the face `face`.
        :param face: The face on whose subspace to project
        :type face: Face
        :param point: The point to project
        :type point: numpy.ndarray
        :param tolerance: The allowed tolerance
        :type tolerance: float, optional
        :return: The projected point, that must now lie on the affine subspace
        :rtype: numpy.ndarray
        """
        vertex_of_face = self.gamma[invert_permutation(self.zone)]
        face_subspace = self.find_face_subspace_without_parent()
        projected_point = project_on_vector_space(point - vertex_of_face, face_subspace.T) + vertex_of_face
        assert self.contains(projected_point), "There has been an error in the projection on a face's subspace"
        return projected_point

    def find_face_subspace_without_parent(self) -> np.ndarray:
        """
            Computes the smallest linear subspace in which `face` lies. Returns the orthonormal vectors to the subspace.
        :param face: The face whose subspace is to be computed
        :type face: Face
        :return: The normal vectors
        """
        n = len(self.gamma)  # The dimensionality of the space
        n_orth = n - self.dim  # The dimensionality of the orthogonal space
        A = np.zeros((n_orth, n))
        for j in np.arange(0, n_orth):
            i = self.splits[j]
            nu = np.ones(n)
            s1 = np.sum(self.gamma[0:i+1])
            s2 = np.sum(self.gamma[i+1:n])
            if s2 == 0:
                psi = 1
            else:
                psi = s1 / s2
            nu[i+1:n] = -psi
            A[j, :] = nu[invert_permutation(self.zone)]
        return orth(A.T)


class Expohedron:
    def __init__(self, gamma):
        self.gamma = gamma

    def contains(self, point):
        return majorized(point, self.gamma)

    def find_face_intersection(self, starting_point: np.ndarray, direction: np.ndarray, precision: float = 1e-12) -> np.ndarray:
        """
            Finds the intersection of a half-line with the border of the expohedron

            Given a starting point `starting_point` in an expohedron given by `gamma` and given a direction vector `direction`, this function find
            the intersection of the half line starting at `starting_point` with direction `direction` with the border of the polytope
        :param gamma: A vertex of the PBM-expohedron
        :type gamma: numpy.ndarray
        :param starting_point: the starting point of the half line
        :type starting_point: numpy.ndarray
        :param direction: the direction of the half line
        :type direction: numpy.ndarray
        :param precision: THe precision to be used for the bisection method
        :type precision: float, optional
        :return: The intersection of the half-line with the border of the polytope
        :rtype: numpy.ndarray
        """
        if np.linalg.norm(direction) < precision:
            Warning("`direction` has norm lower than " + str(precision) + ". Assumed to be 0. Returning `starting_point`.")
            return starting_point
        assert np.sum(direction) < LOW_TOLERANCE, "`direction`'s elements must sum to zero"

        if np.all(np.argsort(starting_point) == np.argsort(direction)):
            return self.find_face_intersection_same_ordering(self.gamma, starting_point, direction)
        else:
            return self.find_face_intersection_bisection(self.gamma, starting_point, direction, precision)

    def find_face_intersection_same_ordering(self, starting_point: np.ndarray, direction: np.ndarray, override_order_constraint: bool = False,
                                         zone: np.ndarray = None) -> np.ndarray:
        """
            Given a starting point `starting_point` in an expohedron given by `gamma` and given a direction vector `direction`, this function finds

            the intersection of the half line starting at `starting_point` with direction `direction` with the border of the polytope
            This is done under the assumption that all points in the half-line have the same ordering
            voir p. 71 et 88 de mon carnet
        :param gamma: A vertex of the PBM-expohedron
        :type gamma: numpy.ndarray
        :param starting_point: the starting point of the half line
        :type starting_point: numpy.ndarray
        :param direction: the direction of the half line
        :type direction: numpy.ndarray
        :param override_order_constraint: If `True` the order constraint is not checked and can be overridden
        :type override_order_constraint: bool, optional
        :param zone: This parameter serves to lift ambiguity whenever the starting point is in several zones. If this argument is given and `starting_point` is indeed in `zone`, then `zone is chosen`
        :type zone: np.ndarray, optional
        :return: The intersection of the half-line with the border of the polytope
        :rtype: numpy.ndarray
        """
        if zone is not None:
            assert np.all(starting_point[zone] == np.sort(starting_point)), "The starting point is not in given zone"
        else:
            zone = np.argsort(starting_point)
        if not override_order_constraint:
            assert np.all(np.sort(starting_point) == starting_point[np.argsort(direction)]), "Both starting point and direction need to have the same ordering"
        assert len(self.gamma) == len(starting_point), "gamma and starting_point must have same length"
        assert len(self.gamma) == len(direction), "gamma and direction must have same length"
        assert np.sum(direction) < LOW_TOLERANCE, "`direction`'s elements must sum to zero"

        Gk = np.cumsum(np.sort(self.gamma))
        Sk = np.cumsum(starting_point[zone])
        Dk = np.cumsum(direction[zone])
        # eliminate the coordinates where Dk is zero; no information about Lambda can be obtained from them. todo: refer to a proof in paper

        # Lambda = min((Gk - Sk)[0:(n - 1)] / Dk[0:(n - 1)])
        indices = np.where(np.abs(Dk) > 1e-12)
        bounds = (Gk - Sk)[indices] / Dk[indices]
        Lambda = min(bounds[np.where(bounds >= 0)], default=0)

        return starting_point + Lambda * direction

    def find_face_intersection_bisection(self, starting_point: np.ndarray, direction: np.ndarray, precision: float = DEFAULT_TOLERANCE) -> np.ndarray:
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
        n = len(self.gamma)
        assert n == len(starting_point), "`starting_point` does not have the same length as `gamma`."
        assert n == len(direction), "`direction` does not have the same length as `gamma`."
        assert majorized(starting_point, self.gamma), "`starting_point` needs to be majorized by `gamma`. Check your inputs or decrease majorization tolerance."

        # 1. Find upper and lower bound
        k = 1
        while majorized((starting_point + k*direction) / np.sum((starting_point + k*direction)) * np.sum(self.gamma), self.gamma):  # We make sure the tested point is in the
            # hyperplane containing the expohedron
            k *= 2
        upper_bound = (starting_point + k*direction) / np.sum((starting_point + k*direction)) * np.sum(self.gamma)
        lower_bound = starting_point

        # 2. Do bisection
        nb_iterations = 0
        while True:
            nb_iterations += 1
            center = (upper_bound + lower_bound) / 2
            if majorized(center, self.gamma, tolerance=precision):  # project center on face's affine subspace
                lower_bound = center
            else:
                upper_bound = center
            if np.all(np.abs(upper_bound - lower_bound) < precision):
                return lower_bound
                # return self.identify_face(lower_bound).post_correction(lower_bound)
            else:
                pass
    
    def identify_face(self, point_on_face: np.ndarray, tolerance: float = LOW_TOLERANCE) -> Face:
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
        n = len(self.gamma)
        assert n == len(point_on_face)

        splits = np.where(np.abs(np.cumsum(np.sort(self.gamma)) - np.cumsum(np.sort(point_on_face))) < tolerance)
        return Face(self.gamma, np.argsort(point_on_face), splits[0])


def find_face_subspace_without_parent_2(point, gamma, tolerance=LOW_TOLERANCE) -> np.ndarray:
    n = len(gamma)  # The dimensionality of the space
    splits = np.where(np.abs(np.cumsum(np.sort(-gamma)) - np.cumsum(np.sort(-point))) < tolerance)[0]
    n_orth = len(splits)  # The dimensionality of the orthogonal space
    A = np.zeros((n, n_orth))
    pos = invert_permutation(-point)
    for j in np.arange(0, n_orth):
        i = splits[j]
        A[pos[:i+1], j] = 1
    return A


def update_face_by_point(next_point, ori_face, gamma, tolerance=LOW_TOLERANCE) -> np.ndarray:
    n = len(gamma)  # The dimensionality of the space
    splits_ori = ori_face.sum(axis=0).astype(int) - 1
    splits_next = np.where(np.abs(np.cumsum(np.sort(-gamma)) - np.cumsum(np.sort(-next_point))) < tolerance)[0]
    splits = set(splits_next).difference(splits_ori)
    n_orth = len(splits)
    A = np.zeros((n, n_orth))
    pos = invert_permutation(-next_point)
    for j, i in enumerate(splits):
        A[pos[:i+1], j] = 1
    return np.concatenate([A, ori_face], axis=1)
