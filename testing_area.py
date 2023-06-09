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


def optimal_utility_point_in_fair_level(start_point: np.ndarray, orthogonal_vectors: np.ndarray,
                                        direction: np.ndarray, gamma: np.ndarray):
    current_point = start_point
    previous_face = identify_face(gamma, current_point)
    current_direction = direction
    complement_basis = orthogonal_complement(orthogonal_vectors)
    # current_face = previous_face
    current_face = identify_face(gamma, current_point)
    corrected_point = post_correction(current_face, current_point)
    k = 0

    while True:
        k += 1
        face_complement = find_face_subspace_without_parent(current_face)
        face_orth = orthogonal_complement(face_complement)
        vertex_of_face = current_face.gamma[invert_permutation(current_face.zone)]
        A = np.concatenate((face_orth, complement_basis), axis=1)
        b = np.concatenate((face_orth.T @ vertex_of_face, complement_basis.T @ start_point), axis=0)
        current_search_faces = intersect_vector_space(face_orth, orthogonal_vectors)

        current_direction = project_vector_on_subspace(direction, current_search_faces)
        current_direction[np.abs(current_direction) < 1e-13] = 0.
        new_point = find_face_intersection_bisection(gamma, corrected_point, current_direction)
        new_face = identify_face(gamma, new_point)

        current_point = project_point_onto_plane(current_point, A, b)

        if current_search_faces.shape[1] == 0:
            break

        # # TODO: error here
        if current_face.dim == previous_face.dim:
            print(current_face.splits)
            print(previous_face.splits)
            # print(current_point)
            print("--------------")
            break

    #     current_point = project_point_onto_plane(current_point, A, b)
        if not new_face.dim < current_face.dim:
            raise Exception("if not face.dim < current_dim","A precision error is likely to have occurred")

        current_point = new_point
        current_face = new_face

    # while True:
    #     print(current_point)
    #     current_point = find_face_intersection_bisection(gamma, current_point, current_direction)
    #     print("binary search: ", (current_point @ complement_basis - start_point @ complement_basis))
    #     print(current_face.contains(current_point))
    #     current_point = current_point / np.sum(current_point) * np.sum(gamma)
    #     print("expo project: ", (current_point @ complement_basis - start_point @ complement_basis))
    #     current_face = identify_face(gamma, current_point)
    #     print(current_point)

    #     if not previous_face.dim >= current_face.dim:
    #         raise Exception("if not face.dim < current_dim", "A precision error is likely to have occurred")

    #     # Post-correction
    #     face_complement = find_face_subspace_without_parent(current_face)
    #     face_orth = orthogonal_complement(face_complement)
    #     vertex_of_face = current_face.gamma[invert_permutation(current_face.zone)]

    #     # current_face.contains(current_point)
    #     # print(pareto_face_basis_vectors.T @ vertex_of_face - pareto_face_basis_vectors.T @ current_point)

    #     A = np.concatenate((face_complement, complement_basis), axis=1)
    #     b = np.concatenate((face_complement.T @ vertex_of_face, complement_basis.T @ start_point), axis=0)
    #     current_search_faces = intersect_vector_space(face_orth, orthogonal_vectors)
    #     # current_search_faces = orth(np.concatenate([face_orth, orthogonal_vectors], axis=1))
    #     # print(np.concatenate([face_orth, orthogonal_vectors], axis=1).shape)

    #     if current_search_faces.shape[1] == 0:
    #         break

    #     # # TODO: error here
    #     if current_face.dim == previous_face.dim:
    #         print(current_face.splits)
    #         print(previous_face.splits)
    #         # print(current_point)
    #         print("--------------")
    #         break

    #     current_point = project_point_onto_plane(current_point, A, b)
    #     # current_point = project_point_onto_plane(current_point, face_complement,
    #     #                                          face_complement.T @ vertex_of_face)
    #     print("plane project: ", (face_complement.T @ vertex_of_face - face_complement.T @ current_point), " ", (complement_basis.T @ start_point - complement_basis.T @ current_point) )
    #     assert current_face.contains(current_point), "Float point error"

    #     previous_face = current_face
    #     current_direction = project_vector_on_subspace(direction, current_search_faces)
    #     current_direction[np.abs(current_direction) < 1e-13] = 0
    #     print(current_direction)

    return current_point


def example(relevance_score: np.ndarray, item_group_masking: np.ndarray, group_fairness: np.ndarray, gamma: np.ndarray):
    n_doc = item_group_masking.shape[0]

    expohedron_basis = orthogonal_complement(np.asarray([[1.0] * n_doc]).T)
    fairness_level_basis_vector = orthogonal_complement(item_group_masking)
    print("Fairness_level_direction_space")

    # Since the vector space is orthogonal with a subspace in the expohedron space
    # The intersection space will also be the projection space
    fairness_level_projection_space = intersect_vector_space(item_group_masking, expohedron_basis)
    # print(fairness_level_projection_space)

    # Random point in fairness surface
    print('Initiate point')
    initiate_fair_point = np.asarray([gamma.sum() / n_doc] * n_doc)
    initiate_fair_point = project_point_onto_plane(initiate_fair_point, item_group_masking, group_fairness)
    assert majorized(initiate_fair_point, gamma), "Initiate point is not in the expohedron"

    # optimal_fairness_direction = project_vector_on_subspace(relevance_score, fairness_level_projection_space)
    end_point = gamma[invert_permutation(np.argsort(-relevance_score))]
    optimal_fairness_direction = project_vector_on_subspace(end_point - initiate_fair_point,
                                                            fairness_level_projection_space)
    print("Optimal_fairness_direction")
    # print(fairness_level_projection_space)

    # direction_space = intersect_vector_space(fairness_level_basis_vector, expohedron_basis)
    # direction = project_vector_on_subspace(relevance_score, direction_space)
    # print(check_spanned_vector(direction, fairness_level_basis_vector))
    direction = project_vector_on_subspace(relevance_score, fairness_level_basis_vector)

    pareto_set = []
    pareto_point = optimal_utility_point_in_fair_level(initiate_fair_point, fairness_level_basis_vector,
                                                       direction, gamma)
    pareto_set.append(pareto_point)

    # print("Start search for pareto front")
    # step = 0.05
    # nb_iteration = 1

    # while True:
    #     nb_iteration += 1
    #     # print(nb_iteration)
    #     start_point = initiate_fair_point + (nb_iteration * step) * optimal_fairness_direction
    #     start_point = start_point / np.sum(start_point) * np.sum(gamma)
    #     if not majorized(start_point, gamma):
    #         break
    #     pareto_point = optimal_utility_point_in_fair_level(start_point, fairness_level_basis_vector,
    #                                                        direction, gamma)
    #     assert majorized(pareto_point, gamma), "Projection went wrong, new point is out of the hedron."
    #     pareto_set.append(pareto_point)
    #     # break

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

    relevance_score = np.loadtxt("data_error/relevance_score.csv", delimiter=",").astype(np.double)
    item_group_masking = np.loadtxt("data_error/item_group.csv", delimiter=",").astype(np.double)
    n_doc = item_group_masking.shape[0]

    # n_doc = 40
    # n_group = 2
    
    # np.random.seed(n_doc)
    # relevance_score = np.random.rand(n_doc)
    # # np.savetxt("data_error/relevance_score.csv", relevance_score, delimiter=",")
    
    # item_group_masking = np.zeros((n_doc, n_group))
    # for i in range(n_doc):
    #     j = np.random.randint(n_group, size=1)
    #     item_group_masking[i][j[0]] = 1
    # # np.savetxt("data_error/item_group.csv", item_group_masking, delimiter=",")

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
    # print("Start QP experiment:")
    # qp_start = time.time()
    # base_qp = QP.experiment(_relevance_score, item_group)
    # qp_end = time.time()
    # print("Done")
    # print((hedron_end - hedron_start) / len(objs))
    # print((qp_end - qp_start) / len(base_qp))
    # draw(objs, base_qp)
