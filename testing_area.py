import matplotlib.pyplot as plt
import numpy as np
import random
from scipy.linalg import orth

from expohederon_group_fairness import PBMexpohedron
from helpers import orthogonal_complement, direction_projection_on_subspace, intersect_vector_space


def draw():
    a = ((14.269311137873268, 1.187725136792305), (17.140071286605405, 1.1934830225907016),
         (17.140856954827676, 1.2004411126975776),
         (17.140634281456983, 1.2095967259609504), (17.141263990381397, 1.2222150136722743),
         (17.142658024163698, 1.2402153675248115),
         (17.143892656991724, 1.2672272378109188), (17.145942914339493, 1.3109890544499345),
         (17.149803488278486, 1.4028168916716548),
         (17.15684902996225, 1.6314599377584822), (17.173555531653545, 3.9043156468687505))
    b = [[14.385308417564582, 1.2092578403993637], [14.385421466188795, 1.2095057377990637],
         [14.33582183429067, 1.7839697210619498],
         [14.385299104916934, 1.2095057377990603], [14.386379195848214, 1.2095952191041774],
         [14.385905566552244, 1.2130189891963277],
         [14.386134089154195, 1.2177653081872186], [14.385636856853408, 1.2094263595077372],
         [14.385441955863373, 1.2095207021174028],
         [14.385276899449897, 1.2092477143885902], [14.38629857351739, 1.20926506836034],
         [14.385507477130469, 1.2092559874410718]]
    a = np.asarray(a)
    b = np.asarray(b)

    plt.plot(a[:, 1], a[:, 0], label="QP")
    plt.plot(b[:, 1], b[:, 0], label="NGSA-3")
    plt.ylabel("User utility")
    plt.xlabel("Unfairness")
    plt.legend(loc='upper left')
    plt.show()


def toy_example():
    # n_doc = 3
    # n_group = 2
    # gamma = np.asarray([4, 2, 1])
    # group_fairness = np.asarray([3.5, 3.5])
    # relevance_score = np.asarray([0.7, 0.5, 1])
    # item_group_masking = np.asarray([[1, 0], [1, 0], [0, 1]])

    n_doc = 10
    n_group = 3
    relevance_score = np.loadtxt("data/relevance_score.csv", delimiter=",").astype(np.double)
    item_group_masking = np.loadtxt("data/item_group.csv", delimiter=",").astype(np.int32)
    # np.random.seed(n_doc)
    # relevance_score = np.random.rand(n_doc)
    # np.savetxt("data/relevance_score.csv", relevance_score, delimiter=",")
    
    # item_group_masking = np.zeros((n_doc, n_group))
    # for i in range(n_doc):
    #     j = random.randint(0, n_group-1)
    #     item_group_masking[i][j] = 1
    # np.savetxt("data/item_group.csv", item_group_masking, delimiter=",")
    gamma = np.asarray([1/np.log2(2+1) for i in range(0, n_doc)])
    group_fairness = np.sum(gamma) / n_group * np.asarray([1] * n_group)
 
    expohedron = PBMexpohedron(pbm=gamma, relevance_vector=relevance_score, item_group_mask=item_group_masking)
    expohedron.set_fairness(group_fairness)
    expohedron_basis = np.asarray([[1] * n_doc]).T
    fairness_level_direction_space = orthogonal_complement(item_group_masking, False)
    # print(fairness_level_direction_space)
    # Since the vector space is orthogonal with a subspace in the expohedron space
    # The intersection space will also be the projection space
    fairness_level_projection_space = intersect_vector_space(fairness_level_direction_space, expohedron_basis)
    # print(fairness_level_projection_space)
    optimal_fairness_direction = direction_projection_on_subspace(relevance_score, fairness_level_projection_space)
    print(optimal_fairness_direction)


if __name__ == "__main__":
    toy_example()
