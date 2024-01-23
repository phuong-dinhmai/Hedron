import numpy as np


def utils(ranking_probability: np.ndarray, relevance_score: np.ndarray, gamma: np.ndarray):
    return np.sum(relevance_score.T @ ranking_probability @ gamma)


def unfairness(ranking_probability: np.array, item_list: np.array, gamma, fair_exposure):
    return np.sum((item_list.T @ (ranking_probability @ gamma) - fair_exposure) ** 2)


def evaluation(ranking_probability: np.array, relevance_score: np.array,
               item_list: np.array, gamma, fair_exposure):
    gamma = -np.sort(-gamma)
    user_utilities = utils(ranking_probability, relevance_score, gamma)
    unfairness_val = unfairness(ranking_probability, item_list, gamma, fair_exposure)
    return user_utilities, unfairness_val


def normalize_evaluation(ranking_probability: np.array, relevance_score: np.array,
                         item_list: np.array, gamma, fair_exposure):
    n_doc = item_list.shape[0]
    gamma = -np.sort(-gamma)
    utils, unfair = evaluation(ranking_probability, relevance_score, item_list, gamma, fair_exposure)
    end_matrix = np.zeros((n_doc, n_doc))
    order = np.argsort(-relevance_score)
    for i in range(n_doc):
        end_matrix[i, order[i]] = 1
    utils_norm_term, unfair_norm_term = evaluation(relevance_score, item_list, gamma, fair_exposure)
    return utils / utils_norm_term, unfair / unfair_norm_term
