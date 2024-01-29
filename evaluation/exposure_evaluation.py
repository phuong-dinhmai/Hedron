import numpy as np


def utils(exposure: np.ndarray, relevance_score: np.ndarray):
    return np.sum(relevance_score.T @ exposure)


def unfairness(exposure: np.ndarray, item_list: np.array, fair_exposure):
    return np.sum((item_list.T @ exposure - fair_exposure) ** 2)


def evaluation(exposure: np.array, relevance_score: np.array,
               item_list: np.array, fair_exposure, args):
    user_utilities = utils(exposure, relevance_score)
    unfairness_val = unfairness(exposure, item_list, fair_exposure)
    return user_utilities, unfairness_val


def normalize_evaluation(exposure: np.array, relevance_score: np.array,
                         item_list: np.array, fair_exposure, optimal_util_point):
    utils, unfair = evaluation(exposure, relevance_score, item_list, fair_exposure, None)
    utils_norm_term, unfair_norm_term = evaluation(optimal_util_point, relevance_score, item_list, fair_exposure, None)
    return utils / utils_norm_term, unfair / unfair_norm_term
