import numpy as np
import pandas as pd

from helpers import majorized, invert_permutation, is_ranking

class PBMexpohedron:
    relevance_vector: np.ndarray
    pbm: np.ndarray
    n: int
    prp_vertex: np.ndarray
    prp_utility: float
    prp_unfairness: float

    def __init__(self, pbm: np.ndarray, relevance_vector: np.ndarray):
        assert np.all(0 <= relevance_vector) and np.all(relevance_vector <= 1), "The values of `relevance_vector` must all be between 0 and 1"
        self.relevance_vector = relevance_vector
        self.pbm = pbm
        self.n = len(relevance_vector)
        self.prp_vertex = pbm[invert_permutation(np.argsort(-relevance_vector))]
        self.prp_utility = self.prp_vertex @ relevance_vector

    def __repr__(self):
        string1 = "PBM expohedron:\n\trelevance_vector of length " + str(self.n) + " :\n\t" + str(self.relevance_vector)
        string2 = "\n\tPBM = " + str(self.pbm)
        return string1 + string2

    def __str__(self):
        return self.__repr__()

    def is_inside(self, point: np.ndarray) -> bool:
        return majorized(point, self.pbm)

    def get_vertex(self, ranking: np.ndarray) -> np.ndarray:
        """
            Given a ranking, a relevance vector and an abandon probability, computes the exposure vector (in the document space)

            The ranking is such that when applied to the document indices
        :param ranking: A matrix of size D x D
        :type ranking: numpy.array
        :return: A column vector of size D
        :rtype: numpy.array
        """
        assert is_ranking(ranking)
        return self.pbm[invert_permutation(ranking)]

    def utility(self, exposure_vector: np.ndarray) -> float:
        return exposure_vector @ self.relevance_vector

    def nutility(self, exposure_vector: np.ndarray) -> float:
        return self.utility(exposure_vector) / self.prp_utility

    def unfairness(self, exposure_vector: np.ndarray, fairness: str = "meritocratic",
                   p_norm: float = 2, meritocratic_endpoint: str = "intersection") -> float:
        """

        :param fairness:
        :param p_norm:
        :param meritocratic_endpoint:
        :return:
        """
        if fairness == "demographic":
            target = self.demographic_fairness_target()
        elif fairness == "meritocratic":
            target = self.meritocratic_target_exposure(type=meritocratic_endpoint)
        else:
            raise ValueError("Invalid value for `fairness`")
        return compute_unfairness(exposure_vector, target, p_norm)

    def nunfairness(self, exposure_vector: np.ndarray, fairness: str = "meritocratic",
                    p_norm: float = 2, meritocratic_endpoint: str = "intersection") -> float:
        """
            Normalized unfairness
        :param fairness:
        :param p_norm:
        :param meritocratic_endpoint:
        :return:
        """
        return self.unfairness(exposure_vector, fairness, p_norm, meritocratic_endpoint) / \
               self.unfairness(self.prp_vertex, fairness, p_norm, meritocratic_endpoint)

    def demographic_fairness_target(self) -> np.ndarray:
        """
            Computes the feasible demographic target exposure vector

        :param expohedron: The expohedron to consider
        :type expohedron: DBNexpohedron
        :return: The demographic target exposure
        :rtype: numpy.ndarray
        """
        return np.ones(self.n) * (self.prp_vertex @ np.ones(self.n)) / (np.ones(self.n) @ np.ones(self.n))

    def meritocratic_target_exposure(self, type: str="intersection") -> np.ndarray:
        """
                Computes the feasible meritocratic target exposure vector

            :param type: How to find a feasible target if the "true" one is infeasible
            :type type: str, optional
            :return: The meritocratic target exposure
            :rtype: numpy.ndarray
        """
        assert type == "intersection", "Only intersection method is currently supported"
        true_fairness_endpoint = self.relevance_vector * (self.prp_vertex @ np.ones(self.n)) / (self.relevance_vector @ np.ones(self.n))
        if self.is_inside(true_fairness_endpoint):
            return true_fairness_endpoint
        else:
            demographic_fairness_point = self.demographic_fairness_target()
            direction = true_fairness_endpoint - demographic_fairness_point
            return find_face_intersection(self.pbm, demographic_fairness_point, direction)

    def target_exposure(self, fairness: str, meritocratic_endpoint: str = "intersection"):
        if fairness == "demographic":
            return self.demographic_fairness_target()
        elif fairness == "meritocratic":
            return self.meritocratic_target_exposure(type=meritocratic_endpoint)
        else:
            raise ValueError("Invalid value for `fairness`")