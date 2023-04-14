import cvxpy as cp
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-alpha", type=float, help="Trade off hyperparameter between utilities and fairness", default=0.5)

class QP:
    relevance_score: np.array 
    item_list: np.array
    alpha: float

    def __init__(self, relevance_score, item_list, alpha) -> None:
        self.relevance_score = relevance_score
        self.item_list = item_list
        self.alpha = alpha
        
        n_doc = relevance_score.shape[0]
        # Birkhoff polotype
        self.ranking_probability = cp.Variable((n_doc, n_doc))

        self.constraints = [0 <= self.ranking_probability, self.ranking_probability <= 1]
        self.constraints.append(cp.sum(self.ranking_probability, axis=0) == 1)
        self.constraints.append(cp.sum(self.ranking_probability, axis=1) == 1)

        self.obj_func = self.alpha * relevance_score

if __name__ == "__main__":
    relevance_score = np.loadtxt("data/relevance_score.csv", delimiter=",", dtype=np.double)
    item_list = np.loadtxt("data/item_group.csv", delimiter=",", dtype=np.int)

    n_doc = relevance_score.shape[0]

