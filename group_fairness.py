from pymoo.problems.functional import FunctionalProblem
from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.optimize import minimize
from pymoo.util.ref_dirs import get_reference_directions
from pymoo.visualization.scatter import Scatter

import numpy as np
import pandas as pd
from utils import *

tolerance = 1e6


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
        self.item_list = item_list
        self.pbm = 1 / np.log(np.arange(0, n_doc) + 2) # the DCG exposure  
        self.n = len(relevance_vector)
        self.prp_vertex = self.pbm[invert_permutation(np.argsort(-relevance_vector))]
        self.prp_utility = np.sum(self.prp_vertex * relevance_vector)
        
        group_size = item_list.sum(axis=0)
        self.fair_exposure = self.pbm.sum() / group_size.sum() * group_size

        objs = [
            lambda x: np.sum(x * self.relevance_vector),
            lambda x: np.sum(np.matmul(x * self.relevance_vector, self.item_list) - self.fair_exposure) ** 2
        ]

        ieq_constrs = []
        for i in range(self.n):
            ieq_constrs.append(
                lambda x: np.sum((-np.sort(-x))[:i]) - np.sum(self.pbm[::-1][:i]) + tolerance,
            )
        ieq_constrs.append( 
            lambda x: np.abs(np.sum(x) - np.sum(self.pbm)) < tolerance
        )

        self.problem = FunctionalProblem(n_var=self.n, objs=objs, constr_ieq=ieq_constrs, xl=1/np.log(n_doc+1), xu=1/np.log(2))

    def optimize(self):
        ref_dirs = get_reference_directions("das-dennis", 2, n_partitions=12)

        # create the algorithm object
        algorithm = NSGA3(
            pop_size=50,
            ref_dirs=ref_dirs
        )

        res = minimize(self.problem,
                    algorithm,
                    ('n_gen', 200),
                    seed=1,
                    verbose=False)
        Scatter().add(res.F).show()


    def evaluate(self, x):
        user_utilities = np.sum(x * self.relevance_vector)
        unfairness = np.matmul(x * self.relevance_vector, self.item_list) - self.fair_exposure
        return np.column_stack([self.prp_utility - user_utilities, np.sum(unfairness ** 2)])


if __name__ == "__main__":
    n_doc = 100
    np.random.seed(n_doc)
    relevance_score = np.random.rand(n_doc)
    item_list = pd.read_csv("/home/phuong/Documents/RecSysReranking/data/ml-100k/u.item",
                        sep='|',
                        engine="python",
                        encoding="latin-1",
                        names=["item", "movie_title", "release_date", "video_release_date",
                                "IMDb_URL", "unknown", "Action", "Adventure", "Animation",
                                "Children", "Comedy", "Crime", "Documentary", "Drama", "Fantasy",
                                "Film_Noir", "Horror", "Musical", "Mystery", "Romance", "Sci_Fi",
                                "Thriller", "War", "Western"],
                        usecols=["item", "unknown", "Action", "Adventure", "Animation",
                                "Children", "Comedy", "Crime", "Documentary", "Drama", "Fantasy",
                                "Film_Noir", "Horror", "Musical", "Mystery", "Romance", "Sci_Fi",
                                "Thriller", "War", "Western"],
                        nrows=100
                        )

    hedron = Expohedron_test(relevance_vector=relevance_score, item_list=item_list.to_numpy())
    hedron.optimize()