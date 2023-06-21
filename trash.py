import matplotlib.pyplot as plt
import numpy as np
import time
from scipy.linalg import null_space, orth
import random

from helpers import project_vector_on_subspace, project_on_vector_space, project_point_on_plane
from helpers import majorized, find_face_intersection_bisection, invert_permutation, intersect_vector_space
from expohedron_face import identify_face, find_face_subspace_without_parent

from expohedron import caratheodory_decomposition_pbm_gls
from evaluation import evaluate_probabilty

import QP


HIGH_TOLERANCE = 1e-12


def draw(a, b):
    a = np.asarray(a)
    b = np.asarray(b)
    # b = b[b[:, 1].argsort()]
    a = a[a[:, 1].argsort()]

    plt.plot(b[:, 1], b[:, 0], label="QP")
    plt.plot(a[:, 1], a[:, 0], label="Hedron")
    plt.ylabel("User utility")
    plt.xlabel("Unfairness")
    plt.legend(loc='lower right')
    plt.show()


def optimal_utility_point_in_fair_level(start_point: np.ndarray, complement_basis: np.ndarray,
                                        direction: np.ndarray, gamma: np.ndarray, checkpoint=False):
    current_point = start_point
    current_direction = project_on_vector_space(direction, complement_basis.T)
    # current_direction = direction
    previous_face = identify_face(gamma, current_point)
    previous_utils = direction @ start_point
    # basis_orth = null_space(complement_basis.T, HIGH_TOLERANCE)
    # complement_basis = null_space(basis_orth.T, HIGH_TOLERANCE)
    # _b_base = complement_basis.T @ start_point
    # k = 0
    n_doc = start_point.shape[0]
    arr = []

    while True:
        current_point = find_face_intersection_bisection(gamma, current_point, current_direction)
        current_utils = direction @ current_point
        current_face = identify_face(gamma, current_point)
        arr.append(current_point)

        if not previous_face.dim >= current_face.dim:
            # return current_point
            raise Exception("if not face.dim < current_dim", "A precision error is likely to have occurred")

        # Post-correction
        face_complement = find_face_subspace_without_parent(current_face)
        # face_orth = null_space(face_complement.T)
        vertex_of_face = current_face.gamma[invert_permutation(current_face.zone)]

        # current_search_faces = intersect_vector_space(face_orth, basis_orth)
        current_search_faces = orth(np.concatenate([face_complement, complement_basis], axis=1))

        if current_search_faces.shape[1] == n_doc:
            break

        if  previous_utils >= current_utils : # and np.abs(previous_utils - current_utils) > 1e-12:
            print(previous_face.dim)
            print("--------------")
            break

        # A = np.concatenate((face_complement, complement_basis), axis=1)
        # b = np.concatenate((face_complement.T @ vertex_of_face, _b_base), axis=0)
        # current_point = project_point_on_plane(current_point, A, b)
        # current_point = project_point_on_plane(current_point, face_complement, face_complement.T @ vertex_of_face)
        current_point = project_on_vector_space(current_point - vertex_of_face, face_complement.T) + vertex_of_face
        assert current_face.contains(current_point), "Float point error"

        previous_face = current_face
        # current_direction = project_vector_on_subspace(direction, current_search_faces)
        current_direction = project_on_vector_space(direction, current_search_faces.T)
        previous_utils = current_utils

    return arr if checkpoint else current_point


def example(relevance_score: np.ndarray, item_group_masking: np.ndarray, group_fairness: np.ndarray, gamma: np.ndarray):
    n_doc = item_group_masking.shape[0]

    expohedron_complement = np.asarray([[1.0] * n_doc])
    expohedron_basis = null_space(expohedron_complement, HIGH_TOLERANCE)

    # print("Fairness_level_direction_space")
    # Since the vector space is orthogonal with a subspace in the expohedron space
    # The intersection space will also be the projection space
    fairness_level_projection_space = intersect_vector_space(expohedron_basis, item_group_masking)

    # Random point in fairness surface
    # print('Initiate point')
    initiate_fair_point = np.asarray([gamma.sum() / n_doc] * n_doc)
    initiate_fair_point = project_point_on_plane(initiate_fair_point, item_group_masking, group_fairness)
    assert majorized(initiate_fair_point, gamma), "Initiate point is not in the expohedron"

    end_point = gamma[invert_permutation(np.argsort(-relevance_score))]
    end_fairness = np.sum((item_group_masking.T @ end_point - group_fairness) ** 2)
   
    # print("Optimal_fairness_direction")
    fixed_direction = project_vector_on_subspace(relevance_score,
                                                fairness_level_projection_space)
    # Post-correction optimal fairness level direction in case starting point is in opposite direction with relevance direction
    optimal_fairness_direction = project_vector_on_subspace(fixed_direction, (end_point-initiate_fair_point).reshape(n_doc, 1))

    direction = relevance_score

    # print("Start search for pareto front")
    pareto_set = []
    objectives = []
    pareto_point = optimal_utility_point_in_fair_level(initiate_fair_point, item_group_masking,
                                                       direction, gamma, False)
    # pareto_set.append(pareto_point)
    # pareto_set.append(end_point)

    # _set = optimal_utility_point_in_fair_level(initiate_fair_point, expohedron_complement.T, optimal_fairness_direction + relevance_score, gamma, True)
    # pareto_set += _set
    # _set = optimal_utility_point_in_fair_level(pareto_point, expohedron_complement.T, optimal_fairness_direction + relevance_score, gamma, True)
    # pareto_set += _set
    _set = optimal_utility_point_in_fair_level(initiate_fair_point, expohedron_complement.T, optimal_fairness_direction + relevance_score, gamma, True)
    pareto_set += _set

    for point in pareto_set:
        utils = relevance_score @ point
        unfairness = np.sum((item_group_masking.T @ point - group_fairness) ** 2)
        objectives.append([utils, unfairness])
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

    n_doc = 100
    n_group = 25

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
