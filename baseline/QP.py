import time

import cvxpy as cp
import numpy as np
import argparse

from evaluation.bistochastic_evaluation import evaluation

parser = argparse.ArgumentParser()
parser.add_argument("-alpha", type=float, help="Trade off hyperparameter between utilities and fairness", default=0.5)


class QP:
    relevance_score: np.array 
    item_list: np.array
    alpha: float
    gamma: np.array
    fair_exposure: np.array

    def __init__(self, relevance_score, item_list, gamma, target_fairness, alpha) -> None:
        n_doc, n_group = item_list.shape

        self.relevance_score = relevance_score
        self.item_list = item_list
        self.alpha = alpha
        self.fair_exposure = target_fairness
        self.gamma = gamma

        # normalize 2 objectives for better alpha search
        self.optimal_utils_matrix = np.zeros((n_doc, n_doc))
        order = np.argsort(-relevance_score)
        for i in range(n_doc):
            self.optimal_utils_matrix[i, order[i]] = 1

        self.optimal_utils = self.utils(self.optimal_utils_matrix).value
        self.unfairness_at_optimal_utils = self.unfairness(self.optimal_utils_matrix).value

        # Birkhoff polotype
        self.ranking_probability = cp.Variable((n_doc, n_doc))

        self.constraints = [0 <= self.ranking_probability, self.ranking_probability <= 1]
        self.constraints.append(cp.sum(self.ranking_probability, axis=0) == np.ones(n_doc))
        self.constraints.append(cp.sum(self.ranking_probability, axis=1) == np.ones(n_doc))

        obj_func = self.alpha * self.utils(self.ranking_probability) - \
                   (1-self.alpha) * self.unfairness(self.ranking_probability)

        self.obj_func = cp.Maximize(obj_func)

    def utils(self, matrix):
        return cp.sum(self.relevance_score.T @ matrix @ self.gamma)

    def normalize_utils(self, matrix):
        return self.utils(matrix) / self.optimal_utils

    def unfairness(self, matrix):
        return cp.sum_squares(self.item_list.T @ (matrix @ self.gamma) - self.fair_exposure)

    def normalize_unfairness(self, matrix):
        return self.unfairness(matrix) / self.unfairness_at_optimal_utils

    def optimize(self):
        prob = cp.Problem(self.obj_func, self.constraints)
        prob.solve(verbose=False)  # Returns the optimal value.
        # print("status:", prob.status)
        # print("optimal value", prob.value)
        # print("optimal var", self.ranking_probability.value)
        return prob.status, self.ranking_probability.value


def experiment(relevance_score: np.ndarray, item_list: np.ndarray, gamma: np.ndarray,
               target_fairness: np.ndarray, alpha_arr: list):
    n_doc, n_group = item_list.shape
    target_fairness = target_fairness

    pareto_set = []
    pareto_front = []
    for alpha in alpha_arr:
        try:
            solver = QP(relevance_score=relevance_score.reshape([n_doc, 1]),
                        item_list=item_list, gamma=gamma, target_fairness=target_fairness, alpha=alpha)
            status, result = solver.optimize()

            # print(status)
            if status == cp.OPTIMAL:
                pareto_set.append(result)
                objectives = evaluation(result, relevance_score, item_list, gamma, target_fairness)
                pareto_front.append(objectives)
        except Exception as e:
            print("Error at alpha: ", alpha)
            print(e)
            pareto_set.append(None)
            pareto_front.append((None, None))
        
    # print(pareto_front)

    return pareto_front, pareto_set


if __name__ == "__main__":
    from data.synthetic.load_data import load_data

    _rel, _item_list, _group_fairness, _gamma = load_data(20, 10)
    n_sample = 10
    start = time.time()
    experiment(_rel, _item_list, _gamma, _group_fairness, np.arange(0, n_sample + 1) / n_sample)
    end = time.time()
    print(end - start)
    


