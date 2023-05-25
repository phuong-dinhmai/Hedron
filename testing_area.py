import matplotlib.pyplot as plt
import numpy as np
import time
from scipy.linalg import orth
import random

from helpers import orthogonal_complement, project_vector_on_subspace, intersect_vector_space, check_orthogonal_and_belonging_vector
from helpers import majorized, find_face_intersection_bisection, project_point_onto_plane, invert_permutation
from expohedron_face import identify_face, find_face_subspace_without_parent

from pareto import caratheodory_decomposition_pbm_gls
from evaluation import evaluate_probabilty

import QP


def draw(a, b):
    a = np.asarray(a)
    b = np.asarray(b)
    b = b[b[:, 1].argsort()]
    a = a[a[:, 1].argsort()]

    plt.plot(a[:, 1], a[:, 0], label="Hedron")
    plt.plot(b[:, 1], b[:, 0], label="QP")
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
        pareto_face_basis_matrix = find_face_subspace_without_parent(current_face)
        face_basis_vectors = orthogonal_complement(pareto_face_basis_matrix)
        current_search_faces = intersect_vector_space(face_basis_vectors, basis_vectors)
        current_search_faces[np.abs(current_search_faces) < 1e-13] = 0

        # TODO: why the intersection result empty vector space?
        if (current_search_faces.shape[1] == 0):
            break
        current_search_faces = orth(current_search_faces)
        if current_search_faces.shape[1] == previous_dim:
            break

        previous_dim = current_search_faces.shape[1]
        current_direction = project_vector_on_subspace(current_direction, current_search_faces)
        current_direction[np.abs(current_direction) < 1e-13] = 0

    return current_point


def example(relevance_score: np.ndarray, item_group_masking: np.ndarray, group_fairness: np.ndarray, gamma: np.ndarray):
    n_doc = item_group_masking.shape[0]

    expohedron_basis = orthogonal_complement(np.asarray([[1] * n_doc]).T)
    fairness_level_basis_vector = orthogonal_complement(item_group_masking, False)
    print("Fairness_level_direction_space")

    # Since the vector space is orthogonal with a subspace in the expohedron space
    # The intersection space will also be the projection space
    fairness_level_projection_space = intersect_vector_space(item_group_masking, expohedron_basis)
    # print(fairness_level_projection_space)

    optimal_fairness_direction = project_vector_on_subspace(relevance_score, fairness_level_projection_space)
    print("Optimal_fairness_direction")

    # Random point in fairness surface
    print('Initiate point')
    initiate_fair_point = np.asarray([gamma.sum() / n_doc] * n_doc)
    initiate_fair_point = project_point_onto_plane(initiate_fair_point, item_group_masking, group_fairness)
    assert majorized(initiate_fair_point, gamma), "Initiate point is not in the expohedron"

    pareto_set = []
    pareto_point = optimal_utility_point_in_fair_level(initiate_fair_point, fairness_level_basis_vector,
                                                       relevance_score, gamma)
    pareto_set.append(pareto_point)

    # raise Exception("test")

    print("Start search for pareto front")
    step = 0.1
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


def load_data():
    # n_doc = 3
    # n_group = 2
    # gamma = np.asarray([4, 2, 1])
    # group_fairness = np.asarray([3.5, 3.5])
    # relevance_score = np.asarray([0.7, 0.8, 1])
    # item_group_masking = np.asarray([[1, 0], [1, 0], [0, 1]])

    # relevance_score = np.loadtxt("data/relevance_score.csv", delimiter=",").astype(np.double)
    # item_group_masking = np.loadtxt("data/item_group.csv", delimiter=",").astype(np.int32)

    n_doc = 20
    n_group = 2

    np.random.seed(n_doc)
    relevance_score = np.random.rand(n_doc)
    np.savetxt("data_error/relevance_score.csv", relevance_score, delimiter=",")
    
    item_group_masking = np.zeros((n_doc, n_group))
    for i in range(n_doc):
        j = random.randint(0, n_group-1)
        item_group_masking[i][j] = 1
    np.savetxt("data_error/item_group.csv", item_group_masking, delimiter=",")

    gamma = 1 / np.log(np.arange(0, n_doc) + 2)
    group_size = item_group_masking.sum(axis=0)
    group_fairness = group_size / np.sum(group_size) * np.sum(gamma)

    return relevance_score, item_group_masking, group_fairness, gamma


if __name__ == "__main__":
    print("Load data")
    relevance_score, item_group, group_fairness, gamma = load_data()
    print("Start hedron experiment:")
    hedron_start = time.time()
    objectives = example(relevance_score, item_group, group_fairness, gamma)
    hedron_end = time.time()
    print("Start QP experiment:")
    qp_start = time.time()
    base_qp = QP.experiment(relevance_score, item_group)
    qp_end = time.time()
    print("Done")
    print(hedron_end - hedron_start)
    print(qp_end - qp_start)
    draw(objectives, base_qp)