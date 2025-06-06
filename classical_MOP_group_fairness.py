import random
import time
import numpy as np
import pandas as pd

from pymoo.problems.functional import FunctionalProblem
from pymoo.core.problem import Problem
from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.algorithms.moo.sms import SMSEMOA
from pymoo.algorithms.moo.rvea import RVEA
from pymoo.constraints.eps import AdaptiveEpsilonConstraintHandling
from pymoo.constraints.as_obj import ConstraintsAsObjective
from pymoo.optimize import minimize
from pymoo.util.ref_dirs import get_reference_directions

from utils import *
from expohedron import caratheodory_decomposition_pbm_gls, majorized

from evaluation import evaluate_probabilty

tolerance = 1e-6


class ConstraintsProblem(Problem):
    relevance_vector: np.array #relevance score of items to query
    item_list: np.array #group mask of items
    n: int #number of items needed ordered
    prp_vertex: np.array #order of items has highest user utilities (decreased order of relevant score) 
    prp_utility: float #highest user utilities archieve without caring about fairness
    fair_exposure: np.array #exposure fairness of group 

    def __init__(self,relevance_vector, item_list) -> None:
        assert np.all(0 <= relevance_vector) and np.all(relevance_vector <= 1), "The values of `relevance_vector` must all be between 0 and 1"
        self.relevance_vector = relevance_vector
        n_doc = relevance_score.shape[0]
        self.item_list = item_list
        self.pbm = 1 / np.log(np.arange(0, n_doc) + 2) #the DCG exposure
        self.n = n_doc

        prp_vertex = self.pbm[invert_permutation(np.argsort(-relevance_vector))]
        self.prp_utility = np.sum(prp_vertex * relevance_vector)
        
        group_size = self.relevance_vector @ item_list
        self.fair_exposure = group_size / np.sum(group_size) * np.sum(self.pbm)
        super().__init__(n_var=n_doc, n_obj=2, n_ieq_constr=n_doc, n_eq_constr=1, xl=1/np.log(n_doc+1), xu=1/np.log(2))

    def _evaluate(self, x, out, *args, **kwargs):
        out["H"] = np.abs(np.sum(x, axis=1) - np.sum(self.pbm)) / np.sum(self.pbm) # Pymoo is not good at handling equal constrainsts, just approximate them
        out["G"] = np.stack(np.cumsum((-np.sort(-x)), axis=1) - np.cumsum((-np.sort(-self.pbm))) - tolerance) / np.cumsum(-np.sort(-self.pbm))
        # eq_constr = np.abs(np.sum(x, axis=1) - np.sum(self.pbm)) - tolerance
        # out["G"] = np.concatenate([out["G"], eq_constr.reshape(eq_constr.shape[0], 1)], axis=1)

        out["F"] = np.stack([
            self.prp_utility - x @ self.relevance_vector.T,
            np.sum((x @ self.item_list - self.fair_exposure) ** 2, axis=1)
        ])
        

class Expohedron_constraint_test:
    def __init__(self, relevance_vector, item_list) -> None:
        self.n = relevance_vector.shape[0]
        self.pbm = 1 / np.log(np.arange(0, self.n) + 2)
        group_size = relevance_vector @ item_list
        self.fair_exposure = group_size / np.sum(group_size) * np.sum(self.pbm)
        self.problem = ConstraintsProblem(relevance_vector=relevance_vector, item_list=item_list)

    def optimize(self):
        ref_dirs = get_reference_directions("das-dennis", 2, n_partitions=10)

        # create the algorithm object
        # algorithm = NSGA3(
        #     pop_size=50,
        #     ref_dirs=ref_dirs
        # )
        algorithm = SMSEMOA()
        # algorithm = RVEA(ref_dirs=ref_dirs)
        # algorithm = AdaptiveEpsilonConstraintHandling(algorithm, perc_eps_until=0.5)

        res = minimize(self.problem, 
                       algorithm,
                       ('n_gen', 1000),
                       seed=4,
                       verbose=True)
        # print(res.F)
        return res.X

    def evaluate(self, x):
        user_utilities = np.sum(x * self.relevance_vector)
        unfairness = x @ self.item_list
        return self.prp_utility - user_utilities, np.sum((unfairness - self.fair_exposure)) ** 2


class Expohedron_fucntion_test:
    relevance_vector: np.array #relevance score of items to query
    item_list: np.array #group mask of items
    n: int #number of items needed ordered
    prp_vertex: np.array #order of items has highest user utilities (decreased order of relevant score) 
    prp_utility: float #highest user utilities archieve without caring about fairness
    fair_exposure: np.array #exposure fairness of group 

    def __init__(self, relevance_vector, item_list) -> None:
        assert np.all(0 <= relevance_vector) and np.all(relevance_vector <= 1), "The values of `relevance_vector` must all be between 0 and 1"
        self.relevance_vector = relevance_vector
        n_doc = relevance_vector.shape[0]
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
        for i in range(1, self.n):
            ieq_constrs.append(
                lambda x: (np.sum((-np.sort(-x))[:i]) - np.sum((-np.sort(-self.pbm))[:i]) - tolerance) / np.sum(self.pbm),
            )
        eq_constrs = [lambda x: np.abs(np.sum(x) - np.sum(self.pbm)) / np.sum(self.pbm)]

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

    # hedron = Expohedron_test(relevance_vector=relevance_score, item_list=item_list)
    hedron = Expohedron_constraint_test(relevance_vector=relevance_score, item_list=item_list)
    exposure_results = hedron.optimize()

    p = np.argmax(relevance_score)

    objectives = []
    for exposure in exposure_results:
        # if not majorized(exposure, hedron.pbm):
        #     continue
        delta = np.sum(hedron.pbm) - np.sum(exposure)
        print(delta)
        exposure[p] += delta

        # Decompose the fairness endpoint
        # print(np.cumsum((-np.sort(-exposure))) - np.cumsum((-np.sort(-hedron.pbm))))

        tic = time.time()
        convex_coefficients, vertices = caratheodory_decomposition_pbm_gls(hedron.pbm, exposure)
        toc = time.time()
        print("The Carathéodory decomposition took " + str(toc - tic) + " seconds.")

        # exposure = vertices @ convex_coefficients.reshape(convex_coefficients.shape[0], 1)
        user_utilities = np.sum(relevance_score.T @ exposure)
        unfairness = np.sum((exposure.T @ item_list - hedron.fair_exposure) ** 2)

        # TODO: change from Caratheodory decomposion to probability matrix for comparison

        objectives.append([user_utilities, unfairness])
    print(objectives)

