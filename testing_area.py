import matplotlib.pyplot as plt
import numpy as np
import random
from scipy.linalg import orth

from expohederon_group_fairness import PBMexpohedron
from helpers import orthogonal_complement, project_vector_on_subspace, intersect_vector_space
from helpers import majorized, find_face_intersection_bisection, project_point_onto_plane
from expohedron_face import identify_face, find_face_subspace_without_parent


def draw():
    # a = ((14.269311137873268, 1.187725136792305), (17.140071286605405, 1.1934830225907016),
    #      (17.140856954827676, 1.2004411126975776),
    #      (17.140634281456983, 1.2095967259609504), (17.141263990381397, 1.2222150136722743),
    #      (17.142658024163698, 1.2402153675248115),
    #      (17.143892656991724, 1.2672272378109188), (17.145942914339493, 1.3109890544499345),
    #      (17.149803488278486, 1.4028168916716548),
    #      (17.15684902996225, 1.6314599377584822), (17.173555531653545, 3.9043156468687505))
    # b = [[14.385308417564582, 1.2092578403993637], [14.385421466188795, 1.2095057377990637],
    #      [14.33582183429067, 1.7839697210619498],
    #      [14.385299104916934, 1.2095057377990603], [14.386379195848214, 1.2095952191041774],
    #      [14.385905566552244, 1.2130189891963277],
    #      [14.386134089154195, 1.2177653081872186], [14.385636856853408, 1.2094263595077372],
    #      [14.385441955863373, 1.2095207021174028],
    #      [14.385276899449897, 1.2092477143885902], [14.38629857351739, 1.20926506836034],
    #      [14.385507477130469, 1.2092559874410718]]
    b = []
    a = [[0.4104583 , 0.45802261, 0.56695921, 0.40645616, 0.54057212,
       0.49192265, 0.48953797, 0.40854049, 0.48202502, 0.28906483], [0.4629194 , 0.41056099, 0.56657536, 0.4571877 , 0.5316349 ,
       0.4619612 , 0.45569595, 0.46017279, 0.44778623, 0.28906483], [0.51538049, 0.36309938, 0.56619151, 0.50791923, 0.52269768,
       0.43199975, 0.42185394, 0.51180509, 0.41354745, 0.28906483], [0.56784158, 0.31563777, 0.56580766, 0.55865077, 0.51376046,
       0.4020383 , 0.38801193, 0.56343739, 0.37930866, 0.28906483], [0.60610169, 0.28906483, 0.5377078 , 0.59696698, 0.48888046,
       0.37784021, 0.36099725, 0.60172438, 0.3552493 , 0.32902645], [0.62629641, 0.28906483, 0.4743498 , 0.61948936, 0.44371931,
       0.3609738 , 0.34266777, 0.6230345 , 0.3441394 , 0.41982415], [0.64649113, 0.28906483, 0.41099179, 0.64201175, 0.39855817,
       0.34410739, 0.3243383 , 0.64434463, 0.3330295 , 0.51062185]]
    a = np.asarray(a)
    b = np.asarray(b)

    plt.plot(a[:, 1], a[:, 0], label="QP")
    plt.plot(b[:, 1], b[:, 0], label="NGSA-3")
    plt.ylabel("User utility")
    plt.xlabel("Unfairness")
    plt.legend(loc='upper left')
    plt.show()


def optimal_utility_point_in_fair_level(start_point: np.ndarray, basis_vectors: np.ndarray,
                                        direction: np.ndarray, gamma: np.ndarray):
    current_direction = project_vector_on_subspace(direction, basis_vectors)
    current_point = start_point
    previous_dim = basis_vectors.shape[1]

    while True:
        current_point = find_face_intersection_bisection(gamma, current_point, current_direction)
        current_face = identify_face(gamma, current_point)
        pareto_face_basis_vectors = find_face_subspace_without_parent(current_face)
        current_search_faces = np.concatenate((basis_vectors, pareto_face_basis_vectors), axis=1)
        current_search_faces = orth(current_search_faces)

        if current_search_faces.shape[1] == previous_dim:
            break
        previous_dim = current_search_faces.shape[1]

        current_direction = project_vector_on_subspace(current_direction, current_search_faces)
    return current_point


def toy_example():
    # n_doc = 3
    # n_group = 2
    # gamma = np.asarray([4, 2, 1])
    # group_fairness = np.asarray([3.5, 3.5])
    # relevance_score = np.asarray([0.7, 0.8, 1])
    # item_group_masking = np.asarray([[1, 0], [1, 0], [0, 1]])

    n_doc = 10
    n_group = 3
    relevance_score = np.loadtxt("data/relevance_score.csv", delimiter=",").astype(np.double)
    item_group_masking = np.loadtxt("data/item_group.csv", delimiter=",").astype(np.int32)
    gamma = np.asarray([1/np.log2(2+i) for i in range(0, n_doc)])
    group_fairness = np.sum(gamma) / n_group * np.asarray([1] * n_group)
    # np.random.seed(n_doc)
    # relevance_score = np.random.rand(n_doc)
    # np.savetxt("data/relevance_score.csv", relevance_score, delimiter=",")
    
    # item_group_masking = np.zeros((n_doc, n_group))
    # for i in range(n_doc):
    #     j = random.randint(0, n_group-1)
    #     item_group_masking[i][j] = 1
    # np.savetxt("data/item_group.csv", item_group_masking, delimiter=",")
 
    # expohedron = PBMexpohedron(pbm=gamma, relevance_vector=relevance_score, item_group_mask=item_group_masking)
    # expohedron.set_fairness(group_fairness)
    expohedron_basis = orthogonal_complement(np.asarray([[1] * n_doc]).T)
    fairness_level_basis_vector = orthogonal_complement(item_group_masking, False)
    # print(fairness_level_direction_space)

    # Since the vector space is orthogonal with a subspace in the expohedron space
    # The intersection space will also be the projection space
    fairness_level_projection_space = intersect_vector_space(item_group_masking, expohedron_basis)
    # print(fairness_level_projection_space)

    optimal_fairness_direction = project_vector_on_subspace(relevance_score, fairness_level_projection_space)
    # print(optimal_fairness_direction)

    # Random point in fairness surface
    initiate_fair_point = np.asarray([gamma.sum() / n_doc] * n_doc)
    initiate_fair_point = project_point_onto_plane(initiate_fair_point, item_group_masking, group_fairness)
    assert majorized(initiate_fair_point, gamma), "Initiate point is not in the expohedron"

    pareto_set = []
    pareto_point = optimal_utility_point_in_fair_level(initiate_fair_point, fairness_level_basis_vector,
                                                       relevance_score, gamma)
    pareto_set.append(pareto_point)

    step = 0.1
    while majorized(initiate_fair_point + step * optimal_fairness_direction, gamma):
        initiate_fair_point = initiate_fair_point + step * optimal_fairness_direction
        pareto_point = optimal_utility_point_in_fair_level(initiate_fair_point, fairness_level_basis_vector,
                                                           relevance_score, gamma)
        pareto_set.append(pareto_point)

    print(pareto_set)


if __name__ == "__main__":
    toy_example()
