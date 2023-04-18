import numpy as np
from billiard import billiard_word
from utils import compute_unfairness


def evaluate_probabilty(ranking_probability: np.array, relevance_score: np.array, item_list: np.array):
    n_doc = relevance_score.shape[0]
    gamma = 1 / np.log(np.arange(0, n_doc) + 2) #the DCG exposure

    group_size = relevance_score @ item_list
    fair_exposure = group_size / np.sum(group_size) * np.sum(gamma)
    
    user_utilities = np.sum((relevance_score.T @ ranking_probability) * gamma)
    unfairness = np.sum((ranking_probability @ gamma @ item_list - fair_exposure) ** 2)
    return user_utilities, unfairness


def time_horizon_evaluate(ranking_probability, vertices, relevance_score, fair_exposure, pbm):
    # Delivery
    time_horizon = 2  # how many rankings should be delivered
    generator = billiard_word(ranking_probability)
    exposure = 0
    utility_matrix = np.zeros(time_horizon) * np.nan
    unfairness_matrix = np.zeros(time_horizon) * np.nan
    idcg = relevance_score @ pbm
    for k in np.arange(0, 2*len(pbm)):
        index = next(generator)  # Faire chauffer l'appareil
    for t in np.arange(1, time_horizon):
        index = next(generator)
        exposure += vertices[:, index]
        utility_matrix[t] = exposure/t @ relevance_score / idcg
        unfairness_matrix[t] = compute_unfairness(exposure/t, fair_exposure) / np.sum(pbm)