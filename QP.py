import time

import cvxpy as cp
import numpy as np
import argparse
import random

from evaluation import evaluate_probabilty

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
        
        n_doc, n_group = item_list.shape
        self.gamma = 1 / np.log(np.arange(0, n_doc) + 2) #the DCG exposure

        group_size = item_list.sum(axis=0)
        self.fair_exposure = group_size / np.sum(group_size) * np.sum(self.gamma)
        self.gamma = self.gamma.reshape([n_doc, 1])
        self.fair_exposure = self.fair_exposure.reshape([n_group, 1])
        # print(self.fair_exposure)

        # Birkhoff polotype
        self.ranking_probability = cp.Variable((n_doc, n_doc))

        self.constraints = [0 <= self.ranking_probability, self.ranking_probability <= 1]
        self.constraints.append(cp.sum(self.ranking_probability, axis=0) == np.ones(n_doc))
        self.constraints.append(cp.sum(self.ranking_probability, axis=1) == np.ones(n_doc))

        utilities = cp.sum(relevance_score.T @ self.ranking_probability @ self.gamma)
        group_exposure = self.item_list.T @ (self.ranking_probability @ self.gamma)

        self.obj_func = cp.Maximize(self.alpha * utilities - (1-self.alpha) * cp.sum((group_exposure - self.fair_exposure) ** 2))

    def optimize(self):
        prob = cp.Problem(self.obj_func, self.constraints)
        prob.solve(verbose=False)  # Returns the optimal value.
        # print("status:", prob.status)
        # print("optimal value", prob.value)
        # print("optimal var", self.ranking_probability.value)
        return prob.status, self.ranking_probability.value


def experiment(relevance_score, item_list):
    n_doc = relevance_score.shape[0]

    pareto_set = []
    pareto_front = []
    alpha_arr = np.arange(0, 81) / 80
    for alpha in alpha_arr:
        solver = QP(relevance_score=relevance_score.reshape([n_doc, 1]), item_list=item_list, alpha=alpha)
        status, result = solver.optimize()

        if status == cp.OPTIMAL:
            pareto_set.append(result)

    for point in pareto_set:
        objecties = evaluate_probabilty(point, relevance_score, item_list)
        pareto_front.append(objecties)
    print(pareto_front)

    pareto_set = np.reshape(pareto_set, [len(pareto_set), n_doc*n_doc])
    np.savetxt("base_result.txt", pareto_set, delimiter=",")

    return pareto_front


if __name__ == "__main__":
    n_doc = 100
    n_group = 25
    np.random.seed(n_doc)
    relevance_score = np.random.rand(n_doc)
    # np.savetxt("data/relevance_score.csv", relevance_score, delimiter=",")

    item_list = np.zeros((n_doc, n_group))
    for i in range(n_doc):
        j = random.randint(0, n_group-1)
        item_list[i][j] = 1

    # np.savetxt("data/item_group.csv", item_list, delimiter=",")

    # relevance_score = np.loadtxt("data/relevance_score.csv", delimiter=",").astype(np.double)
    # item_list = np.loadtxt("data/item_group.csv", delimiter=",").astype(np.int32)
    start = time.time()
    experiment(relevance_score, item_list)
    end = time.time()
    print(end - start)

    


