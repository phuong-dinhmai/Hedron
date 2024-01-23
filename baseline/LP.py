import time

import cvxpy as cp
import numpy as np
import argparse

from evaluation.bistochastic_evaluation import evaluation

parser = argparse.ArgumentParser()
parser.add_argument("-alpha", type=float, help="Trade off hyperparameter between utilities and fairness", default=0.5)


class LP:
    relevance_score: np.array
    item_list: np.array
    gamma: np.array
    fair_exposure: np.array

    def __init__(self, relevance_score, item_list, gamma, fair_exposure) -> None:
        self.relevance_score = relevance_score
        self.item_list = item_list

        n_doc, n_group = item_list.shape

        self.gamma = gamma
        self.fair_exposure = fair_exposure
        # print(self.fair_exposure)

        # Birkhoff polotype
        self.ranking_probability = cp.Variable((n_doc, n_doc))

        self.constraints = [0 <= self.ranking_probability, self.ranking_probability <= 1]
        self.constraints.append(cp.matmul(np.ones((1, n_doc)), self.ranking_probability) == np.ones((1, n_doc)))
        self.constraints.append(cp.matmul(self.ranking_probability, np.ones((n_doc,))) == np.ones((n_doc,)))

        utilities = cp.sum(relevance_score.T @ self.ranking_probability @ self.gamma)
        group_exposure = self.item_list.T @ (self.ranking_probability @ self.gamma)
        self.constraints.append(group_exposure[:, 0] == self.fair_exposure)

        self.obj_func = cp.Maximize(utilities)

    def optimize(self):
        prob = cp.Problem(self.obj_func, self.constraints)
        prob.solve(verbose=True)  # Returns the optimal value.
        # print("status:", prob.status)
        # print("optimal value", prob.value)
        # print("optimal var", self.ranking_probability.value)
        return prob.status, self.ranking_probability.value


def experiment(relevance_score, item_list, gamma, target_fairness):
    n_doc, n_group = item_list.shape
    target_fairness = target_fairness.reshape([n_group, 1])

    try:
        solver = LP(relevance_score=relevance_score.reshape([n_doc, 1]),
                    item_list=item_list, gamma=gamma, fair_exposure=target_fairness)
        status, result = solver.optimize()

        if status == cp.OPTIMAL:
            objectives = evaluation(result, relevance_score, item_list, gamma, target_fairness)
    except:
        raise Exception("Error")

    # print(np.sort(relevance_score) @ np.sort(gamma))
    print(objectives)

    # pareto_set = np.reshape(pareto_set, [len(pareto_set), n_doc*n_doc])
    # np.savetxt("base_result.txt", pareto_set, delimiter=",")

    return result, objectives


if __name__ == "__main__":
    from data.synthetic.load_data import load_data
    rel, item_group_masking, group_fairness, _gamma = load_data(20, 10)

    start = time.time()
    results, objs = experiment(rel, item_group_masking, _gamma, group_fairness)
    end = time.time()
    print(end-start)
    # with open("results/running_LP.json", "w") as f_out:
    #     json.dump(running_time, f_out)





