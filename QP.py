import cvxpy as cp
import numpy as np
import argparse

from helpers import evaluate_probabilty

parser = argparse.ArgumentParser()
parser.add_argument("-alpha", type=float, help="Trade off hyperparameter between utilities and fairness", default=0.5)

class QP:
    relevance_score: np.array 
    item_list: np.array
    alpha: float
    gamma: np.array

    def __init__(self, relevance_score, item_list, alpha) -> None:
        self.relevance_score = relevance_score
        self.item_list = item_list
        self.alpha = alpha
        
        n_doc = relevance_score.shape[0]
        self.gamma = 1 / np.log(np.arange(0, n_doc) + 2) #the DCG exposure
        self.gamma.reshape((n_doc, 1))

        group_size = item_list.sum(axis=0)
        self.fair_exposure = group_size / np.sum(group_size) * np.sum(self.gamma)
        # print(self.fair_exposure)

        # Birkhoff polotype
        self.ranking_probability = cp.Variable((n_doc, n_doc))

        self.constraints = [0 <= self.ranking_probability, self.ranking_probability <= 1]
        self.constraints.append(cp.sum(self.ranking_probability, axis=0) == np.ones((n_doc)))
        self.constraints.append(cp.sum(self.ranking_probability, axis=1) == np.ones((n_doc)))
        print(self.constraints)

        self.obj_func = cp.Maximize(self.alpha * (relevance_score.T @ self.ranking_probability) @ self.gamma  \
                        - (1-self.alpha) * cp.sum((self.ranking_probability @ self.gamma @ self.item_list - self.fair_exposure) ** 2))

    def optimize(self):
        prob = cp.Problem(self.obj_func, self.constraints)
        prob.solve(verbose=False)  # Returns the optimal value.
        # print("status:", prob.status)
        # print("optimal value", prob.value)
        # print("optimal var", self.ranking_probability.value)
        return prob.status, self.ranking_probability.value

if __name__ == "__main__":
    relevance_score = np.loadtxt("data/relevance_score.csv", delimiter=",").astype(np.double)
    item_list = np.loadtxt("data/item_group.csv", delimiter=",").astype(np.int32)

    n_doc = relevance_score.shape[0]

    pareto_set = []
    pareto_front = []
    alpha_arr = np.arange(1, 11) / 10
    for alpha in alpha_arr:
        solver = QP(relevance_score=relevance_score.reshape((100, 1)), item_list=item_list, alpha=alpha)
        status, result = solver.optimize()

        if status == cp.OPTIMAL:
            pareto_set.append(result)

    for point in pareto_set:
        objecties = evaluate_probabilty(point, relevance_score, item_list)
        pareto_front.append(objecties)
    print (pareto_front)


