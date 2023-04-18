import random
import time

from pymoo.problems.functional import FunctionalProblem
from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.optimize import minimize
from pymoo.util.ref_dirs import get_reference_directions
from pymoo.visualization.scatter import Scatter

import numpy as np
import pandas as pd
from utils import *

from expohedron import caratheodory_decomposition_pbm_gls

tolerance = 1e-12


class Expohedron_test:
    relevance_vector: np.array #relevance score of items to query
    item_list: np.array #group mask of items
    n: int #number of items needed ordered
    prp_vertex: np.array #order of items has highest user utilities (decreased order of relevant score) 
    prp_utility: float #highest user utilities archieve without caring about fairness
    fair_exposure: np.array #exposure fairness of group 

    def __init__(self, relevance_vector, item_list) -> None:
        assert np.all(0 <= relevance_vector) and np.all(relevance_vector <= 1), "The values of `relevance_vector` must all be between 0 and 1"
        self.relevance_vector = relevance_vector
        n_doc = relevance_score.shape[0]
        self.item_list = item_list
        self.pbm = 1 / np.log(np.arange(0, n_doc) + 2) #the DCG exposure
        self.n = n_doc
        self.prp_vertex = self.pbm[invert_permutation(np.argsort(-relevance_vector))]
        self.prp_utility = np.sum(self.prp_vertex * relevance_vector)
        
        group_size = self.relevance_vector @ item_list
        self.fair_exposure = group_size / np.sum(group_size) * np.sum(self.pbm)
        # print(self.fair_exposure)

        objs = [
            lambda x: self.prp_utility - x @ self.relevance_vector.T,
            lambda x: np.sum((x @ self.item_list - self.fair_exposure) ** 2)
        ]

        ieq_constrs = []
        for i in range(self.n-1):
            ieq_constrs.append(
                lambda x: np.sum((-np.sort(-x))[:i]) - np.sum((-np.sort(-self.pbm))[:i]) - tolerance,
            )
        eq_constrs = [lambda x: np.abs(np.sum(x) - np.sum(self.pbm))]

        self.problem = FunctionalProblem(n_var=self.n, objs=objs, constr_ieq=ieq_constrs, constr_eq=eq_constrs,
                                         xl=1/np.log(n_doc+1), xu=1/np.log(2))

    def optimize(self):
        ref_dirs = get_reference_directions("das-dennis", 2, n_partitions=16)

        # create the algorithm object
        algorithm = NSGA3(
            pop_size=20,
            ref_dirs=ref_dirs
        )

        res = minimize(self.problem,
                       algorithm,
                       ('n_gen', 600),
                       seed=4,
                       verbose=True)
        print(res.F)
        return res.X

    def evaluate(self, x):
        user_utilities = np.sum(x * self.relevance_vector)
        unfairness = x @ self.item_list
        # for i in range(self.n):
        #     print(np.sum(self.pbm[:i]) - np.sum((-np.sort(-x))[:i]) - tolerance)
        return np.column_stack([self.prp_utility - user_utilities, np.sum((unfairness - self.fair_exposure)) ** 2])


if __name__ == "__main__":
    # n_doc = 100
    # n_group = 5
    # np.random.seed(n_doc)
    # relevance_score = np.random.rand(n_doc)
    # np.savetxt("data/relevance_score.csv", relevance_score, delimiter=",")

    # item_list = np.zeros((n_doc, n_group))
    # for i in range(n_doc):
    #     j = random.randint(0, n_group-1)
    #     item_list[i][j] = 1

    # np.savetxt("data/item_group.csv", item_list, delimiter=",")
    
    relevance_score = np.loadtxt("data/relevance_score.csv", delimiter=",").astype(np.double)
    item_list = np.loadtxt("data/item_group.csv", delimiter=",").astype(np.int32)

    hedron = Expohedron_test(relevance_vector=relevance_score, item_list=item_list)
    exposure_results = hedron.optimize()

    for exposure in exposure_results:
        # Decompose the fairness endpoint
        for i in range(hedron.n-1):
            print(np.sum((-np.sort(-exposure))[:i]) - np.sum((-np.sort(-hedron.pbm))[:i]) - tolerance)
        print(np.cumsum(-np.sort(-exposure)) - np.cumsum(-np.sort(-hedron.pbm)) - tolerance)
        print(np.abs(np.sum(exposure) - np.sum(hedron.pbm)))

        tic = time.time()
        convex_coefficients, vertices = caratheodory_decomposition_pbm_gls(hedron.pbm, exposure)
        toc = time.time()
        print("The Carathéodory decomposition took " + str(toc - tic) + " seconds.")

        print(convex_coefficients)
        print(vertices)


