import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import orth
import time

from expohederon_group_fairness import PBMexpohedron
from helpers import orthogonal_complement, project_vector_on_subspace, intersect_vector_space
from helpers import majorized, find_face_intersection_bisection, project_point_onto_plane, invert_permutation
from expohedron_face import identify_face, find_face_subspace_without_parent

from pareto import caratheodory_decomposition_pbm_gls

from evaluation import evaluate_probabilty


def draw(a):
    b = [(2.696631360807602, 8.38164711797325e-31), (2.9982496632271998, 0.00013244303963676644), (3.003844880196346, 0.0005902582028100382), 
    (3.0100993308470283, 0.0014888919836944163), (3.0171266843390243, 0.002988071026721005), (3.025086123918304, 0.005311710621510635), 
    (3.034195204880937, 0.008778863796807335), (3.0446932236007336, 0.013857740478585075), (3.0569748084539037, 0.02124722789016048), 
    (3.071485498368487, 0.03201027158876389), (3.08886268589049, 0.047812919881222736), (3.1101334292834264, 0.07142969890449904), 
    (3.1366609382818487, 0.10758952581778192), (3.1708492136243134, 0.16490729184319128), (3.2115036314778904, 0.24998261010097175), 
    (3.2460615055863933, 0.3371326852938169), (3.285034199992179, 0.4766726527435362), (3.318023563914365, 0.6293974870927266), 
    (3.3460413309433554, 0.8216059374545983), (3.347308401590674, 0.8393504613130608), (3.35187913828159, 1.1371531328662687)]
    # a = [[2.888080709604799, 4.437342591868191e-31], [2.9442077897492993, 0.01737058163979818], [3.000334869894391, 0.069482326559193], 
    # [3.0564619500394827, 0.15633523475818448], [3.1125890301845742, 0.2779293062367726], [3.1687161103296657, 0.43426454099495815], 
    # [3.224843190474757, 0.6253409390327388], [3.266340159865052, 0.851158500350116], [3.3518791390065936, 1.1371531281828773]]
    a = np.asarray(a)
    b = np.asarray(b)
    b = b[b[:, 1].argsort()]

    plt.plot(a[:, 1], a[:, 0], label="Hedron")
    plt.plot(b[:, 1], b[:, 0], label="QP")
    plt.ylabel("User utility")
    plt.xlabel("Unfairness")
    plt.legend(loc='upper left')
    plt.show()

# TODO: check this function - the while loop never pass the second iteration
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
    gamma = 1 / np.log(np.arange(0, n_doc) + 2)
    group_size = item_group_masking.sum(axis=0)
    group_fairness = group_size / np.sum(group_size) * np.sum(gamma)
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

    step = 0.01
    while majorized(initiate_fair_point + step * optimal_fairness_direction, gamma):
        initiate_fair_point = initiate_fair_point + step * optimal_fairness_direction
        pareto_point = optimal_utility_point_in_fair_level(initiate_fair_point, fairness_level_basis_vector,
                                                           relevance_score, gamma)
        assert majorized(pareto_point, gamma), "Something went wrong with the projection, new point is out of the hedron."
        pareto_set.append(pareto_point)

    pareto_set.append(gamma[invert_permutation(np.argsort(-relevance_score))])
    # print(pareto_set)

    objectives = []
    for exposure in pareto_set:
        user_utilities = np.sum(relevance_score.T @ exposure)
        unfairness = np.sum((exposure.T @ item_group_masking - group_fairness) ** 2)

        objectives.append([user_utilities, unfairness])
    print(objectives)
    return objectives


if __name__ == "__main__":
    objectives = toy_example()
    draw(objectives)