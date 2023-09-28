import numpy as np


def billiard_word(frequency):
    """
    We break ties by order in the input.

    Example
    ------
    To make the first 20 letters of the balanced sequence
        0, 1, 1, 0, 1, 0, 1, 1, 0, 1, ...,
    which repeats every 5 letters, do
        from billiard_word import billiard_word
        gen = billiard_word([2/5, 3/5])
        sequence = [next(gen) for _ in range(20)]
    """
    import heapq

    assert all(_ > 0 for _ in frequency)
    assert abs(sum(frequency) - 1.) < 1e-9

    tiny = 1e-9  # control roundoff issues for finite words
    heap = [(tiny*_, _) for _ in range(len(frequency))]
    while True:
        phase, letter = heapq.heappop(heap)
        heapq.heappush(heap, (phase + 1./frequency[letter], letter))
        yield letter



def evaluate_probabilty(ranking_probability: np.array, relevance_score: np.array, item_list: np.array):
    n_doc, n_group = item_list.shape
    gamma = 1 / np.log(np.arange(0, n_doc) + 2) #the DCG exposure

    group_size = item_list.sum(axis=0)
    fair_exposure = group_size / np.sum(group_size) * np.sum(gamma)
    gamma = gamma.reshape([n_doc, 1])
    fair_exposure = fair_exposure.reshape([n_group, 1])
    
    user_utilities = np.sum(relevance_score.T @ ranking_probability @ gamma)
    unfairness = np.sum((item_list.T @ (ranking_probability @ gamma) - fair_exposure) ** 2)
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
        unfairness_matrix[t] = np.linalg.norm(exposure/t - fair_exposure) / np.sum(pbm)