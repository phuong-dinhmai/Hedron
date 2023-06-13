import matplotlib.pyplot as plt
import numpy as np
import time
from scipy.linalg import orth
import random

from helpers import orthogonal_complement, project_vector_on_subspace, intersect_vector_space, check_spanned_vector
from helpers import majorized, find_face_intersection_bisection, project_point_onto_plane, invert_permutation
from expohedron_face import identify_face, find_face_subspace_without_parent, post_correction

from expohedron import caratheodory_decomposition_pbm_gls
from evaluation import evaluate_probabilty

import QP


def draw(a, b):
    a = np.asarray(a)
    b = np.asarray(b)
    # b = b[b[:, 1].argsort()]
    # a = a[a[:, 1].argsort()]

    plt.plot(b[:, 1], b[:, 0], label="QP")
    plt.plot(a[:, 1], a[:, 0], label="Hedron")
    plt.ylabel("User utility")
    plt.xlabel("Unfairness")
    plt.legend(loc='lower right')
    plt.show()


def optimal_utility_point_in_fair_level(start_point: np.ndarray, complement_basis: np.ndarray, orthogonal_vectors: np.ndarray,
                                        direction: np.ndarray, gamma: np.ndarray):
    current_point = start_point
    # current_direction = project_vector_on_subspace(direction, orthogonal_vectors)
    current_direction = direction
    previous_face = identify_face(gamma, current_point)
    previous_utils = np.sum(np.dot(start_point, direction))
    previous_point = start_point

    k = 0

    while True:
        k += 1
        current_point = find_face_intersection_bisection(gamma, current_point, current_direction)
        current_point = current_point / np.sum(current_point) * np.sum(gamma)
        current_utils = np.sum(np.dot(current_point, direction))
        current_face = identify_face(gamma, current_point)

        if not previous_face.dim >= current_face.dim:
            return previous_point
            # raise Exception("if not face.dim < current_dim", "A precision error is likely to have occurred")

        # Post-correction
        face_complement = find_face_subspace_without_parent(current_face)
        face_orth = orthogonal_complement(face_complement)
        vertex_of_face = current_face.gamma[invert_permutation(current_face.zone)]

        current_search_faces = intersect_vector_space(face_orth, orthogonal_vectors)

        if current_search_faces.shape[1] == 0:
            break

        if  previous_utils >= current_utils : # and np.abs(previous_utils - current_utils) > 1e-12:
            print(previous_face.dim)
            print("--------------")
            break
        
        # print(current_point)
        distance_to_plane = np.concatenate((face_complement.T @ vertex_of_face - face_complement.T @ current_point,
                                            complement_basis.T @ start_point - complement_basis.T @ current_point), axis=0)
        if np.any(np.abs(distance_to_plane) >= 1e-13):
            A = np.concatenate((face_complement, complement_basis), axis=1)
            b = np.concatenate((face_complement.T @ vertex_of_face, complement_basis.T @ start_point), axis=0)
            current_point = project_point_onto_plane(current_point, A, b)
        assert current_face.contains(current_point), "Float point error"

        previous_face = current_face
        current_direction = project_vector_on_subspace(direction, current_search_faces)
        current_direction[np.abs(current_direction) < 1e-13] = 0
        previous_utils = current_utils
        previous_point = current_point

    return current_point


def example(relevance_score: np.ndarray, item_group_masking: np.ndarray, group_fairness: np.ndarray, gamma: np.ndarray):
    n_doc = item_group_masking.shape[0]

    expohedron_basis = orthogonal_complement(np.asarray([[1.0] * n_doc]).T)
    fairness_level_basis_vector = orthogonal_complement(item_group_masking)
    print("Fairness_level_direction_space")

    # Since the vector space is orthogonal with a subspace in the expohedron space
    # The intersection space will also be the projection space
    fairness_level_projection_space = intersect_vector_space(item_group_masking, expohedron_basis)

    # Random point in fairness surface
    print('Initiate point')
    initiate_fair_point = np.asarray([gamma.sum() / n_doc] * n_doc)
    initiate_fair_point = project_point_onto_plane(initiate_fair_point, item_group_masking, group_fairness)
    assert majorized(initiate_fair_point, gamma), "Initiate point is not in the expohedron"

    end_point = gamma[invert_permutation(np.argsort(-relevance_score))]
    optimal_fairness_direction = project_vector_on_subspace(end_point - initiate_fair_point,
                                                            fairness_level_projection_space)
    print("Optimal_fairness_direction")
    
    direction = project_vector_on_subspace(relevance_score, fairness_level_basis_vector)
    # direction = relevance_score

    pareto_set = []
    pareto_point = optimal_utility_point_in_fair_level(initiate_fair_point, item_group_masking, fairness_level_basis_vector,
                                                       direction, gamma)
    pareto_set.append(pareto_point)

    print("Start search for pareto front")
    step = 0.05
    nb_iteration = 0

    while True:
        nb_iteration += 1
        # print(nb_iteration)
        start_point = initiate_fair_point + (nb_iteration * step) * optimal_fairness_direction
        start_point = start_point / np.sum(start_point) * np.sum(gamma)
        if not majorized(start_point, gamma):
            break
        pareto_point = optimal_utility_point_in_fair_level(start_point, item_group_masking, fairness_level_basis_vector,
                                                           direction, gamma)
        assert majorized(pareto_point, gamma), "Projection went wrong, new point is out of the hedron."
        pareto_set.append(pareto_point)
        # break

    # pareto_set.append(end_point)

    objectives = []
    for exposure in pareto_set:
        user_utilities = np.sum(relevance_score.T @ exposure)
        unfairness = np.sum((exposure.T @ item_group_masking - group_fairness) ** 2)

        objectives.append([user_utilities, unfairness])
    print(objectives)
    return objectives


def load_data():
    # n_doc = 4
    # n_group = 2
    # relevance_score = np.asarray([0.7, 0.8, 1, 0.4])
    # item_group_masking = np.asarray([[0, 1], [0, 1], [1, 0], [1, 0]])

    # relevance_score = np.loadtxt("data_error/relevance_score.csv", delimiter=",").astype(np.double)
    # item_group_masking = np.loadtxt("data_error/item_group.csv", delimiter=",").astype(np.double)
    # n_doc = item_group_masking.shape[0]

    n_doc = 40
    n_group = 5
    
    np.random.seed(n_doc)
    relevance_score = np.random.rand(n_doc)
    # np.savetxt("data_error/relevance_score.csv", relevance_score, delimiter=",")
    
    item_group_masking = np.zeros((n_doc, n_group))
    for i in range(n_doc):
        j = np.random.randint(n_group, size=1)
        item_group_masking[i][j[0]] = 1
    # np.savetxt("data_error/item_group.csv", item_group_masking, delimiter=",")

    gamma = 1 / np.log(np.arange(0, n_doc) + 2)
    group_size = item_group_masking.sum(axis=0)
    group_fairness = group_size / np.sum(group_size) * np.sum(gamma)

    return relevance_score, item_group_masking, group_fairness, gamma


if __name__ == "__main__":
    print("Load data")
    _relevance_score, item_group, _group_fairness, _gamma = load_data()
    print("Start hedron experiment:")
    hedron_start = time.time()
    objs = example(_relevance_score, item_group, _group_fairness, _gamma)
    hedron_end = time.time()
    print("Start QP experiment:")
    qp_start = time.time()
    base_qp = QP.experiment(_relevance_score, item_group)
    qp_end = time.time()
    print("Done")
    print((hedron_end - hedron_start) / len(objs))
    print((qp_end - qp_start) / len(base_qp))
    draw(objs, base_qp)
