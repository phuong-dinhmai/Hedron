import numpy as np
import pandas as pd

from helpers import majorized, invert_permutation, is_ranking


class PBMexpohedron:
    relevance_vector: np.ndarray
    item_group_mask: np.ndarray
    pbm: np.ndarray
    fairness: np.ndarray
    n: int
    prp_vertex: np.ndarray
    prp_utility: float
    prp_unfairness: float

    def __init__(self, pbm: np.ndarray, relevance_vector: np.ndarray, item_group_mask: np.ndarray):
        assert np.all(0 <= relevance_vector) and np.all(relevance_vector <= 1), \
            "The values of `relevance_vector` must all be between 0 and 1"
        assert np.all(np.sum(item_group_mask, axis=1) == 1), "Each item MUST belongs to EXACT ONE GROUP"
        assert len(item_group_mask.shape) == 2 and item_group_mask.shape[0] == relevance_vector.shape[0], \
            "Masking group error. It must be a matrix with number of rows equals number of items in relevance vector"

        self.relevance_vector = relevance_vector
        self.pbm = pbm
        self.item_group_mask = item_group_mask
        self.n = len(relevance_vector)
        self.prp_vertex = pbm[invert_permutation(np.argsort(-relevance_vector))]
        self.prp_utility = self.prp_vertex @ relevance_vector

    def __repr__(self):
        string1 = "PBM expohedron:\n\trelevance_vector of length " + str(self.n) + " :\n\t" + str(self.relevance_vector)
        string2 = "\n\tPBM = " + str(self.pbm)
        return string1 + string2

    def __str__(self):
        return self.__repr__()

    def set_fairness(self, fairness_vector):
        self.fairness = fairness_vector

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

    def unfairness(self, exposure_vector, p_norm):
        group_exposure = exposure_vector.reshape((self.n, 1)) @ self.item_group_mask
        return np.linalg.norm(group_exposure - self.fairness, ord=p_norm)

    def nunfairness(self, exposure_vector: np.ndarray, p_norm: float = 2) -> float:
        """
            Normalized unfairness
        :param exposure_vector
        :param p_norm:
        :return:
        """
        return self.unfairness(exposure_vector, p_norm) / self.unfairness(self.prp_vertex, p_norm)