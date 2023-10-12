import numpy as np
import pandas as pd

from scipy.linalg import null_space, norm, orth

from helpers import project_point_on_plane, project_on_vector_space, invert_permutation, intersect_vector_space
from helpers import Objective, project_vector_on_subspace
from sphereCoordinator import BasisTransformer, SphereCoordinator
from expohedron import find_face_subspace_without_parent_2, update_face_by_point, Expohedron


HIGH_TOLERANCE = 1e-12


def projected_path(relevance_score: np.ndarray, item_group_masking: np.ndarray, group_fairness: np.ndarray, gamma: np.ndarray):
    n_doc, _ = item_group_masking.shape
    hedron = Expohedron(gamma)
    objs = Objective(relevance_score, group_fairness, item_group_masking, gamma)

    if n_doc < 12:
        expohedron_complement = np.asarray([[1.0] * n_doc])
        x = intersect_vector_space(null_space(expohedron_complement), item_group_masking).T

    # print("Start search for pareto front")
    pareto_set = []
    objectives = []

    nb_iteration = 0
    pareto_point = objs.convex_constraints_prob(group_fairness)
    assert hedron.contains(pareto_point), "Projection went wrong, new point is out of the hedron."

    objectives.append(objs.objectives(pareto_point))
    pareto_set.append(pareto_point)

    face_orth = find_face_subspace_without_parent_2(pareto_point, gamma)
    next_face = face_orth

    optimal_fairness_direction = relevance_score
    # optimal_fairness_direction = project_on_vector_space(relevance_score, item_group_masking.T)
    while True:
        if not hedron.contains(pareto_point):
            break
        _, n_col = face_orth.shape
        max_utils = objs.utils(pareto_point)
        pre_util = objs.utils(pareto_point)
        # print("Current optimal: ", max_utils)
        next_point = None
        for exclude in range(-1, n_col-1):
            _face_orth = face_orth[:, np.arange(n_col) != exclude]
            check_dir = project_on_vector_space(optimal_fairness_direction, _face_orth.T)

            if np.all(np.abs(check_dir) < 1e-13):
                continue
            check_dir[np.abs(check_dir) < 1e-13] = 0

            intersect_point = hedron.find_face_intersection_bisection(pareto_point, check_dir)

            # Post-correction for small N
            if n_doc < 12:
                y = null_space(_face_orth.T)
                check = intersect_vector_space(null_space(relevance_score.reshape((1, n_doc))), y)
                if check.shape[1] != 0:
                    k = np.asarray([project_vector_on_subspace(vec, check)
                                    for vec in x])
                    k = orth(k.T)
                    for vec in k.T:
                        sign = objs.unfairness(intersect_point + 0.0001*vec) - objs.unfairness(intersect_point)
                        intersect_point = hedron.find_face_intersection_bisection(intersect_point, -sign * vec)

            if norm(intersect_point - pareto_point) < 1e-6:
                continue

            user_utilities = objs.utils(intersect_point)
            x1 = (user_utilities - pre_util) / (objs.unfairness(intersect_point) - objs.unfairness(pareto_point))
            x2 = (max_utils - pre_util) / (objs.unfairness(next_point) - objs.unfairness(pareto_point)) \
                if next_point is not None else 0
            if x1 > x2:
                max_utils = user_utilities
                next_point = intersect_point
                next_face = update_face_by_point(intersect_point, _face_orth, gamma)

            # print(user_utilities)
            # if user_utilities - max_utils > 1e-6:
            #     max_utils = user_utilities
            #     next_point = intersect_point
        if next_point is None:
            break
        nb_iteration += 1
        # if nb_iteration == 30:
        #     break
        pareto_set.append(next_point)
        objectives.append(objs.objectives(next_point))
        pareto_point = next_point
        face_orth = next_face

    # print(objectives)
    return objectives, pareto_set


def sphere_path(relevance_score: np.ndarray, item_group_masking: np.ndarray, group_fairness: np.ndarray, gamma: np.ndarray, n_divided=3, n_sample=5):
    n_doc, _ = item_group_masking.shape

    expohedron_complement = np.asarray([[1.0] * n_doc])
    hedron = Expohedron(gamma)
    objs = Objective(relevance_score, group_fairness, item_group_masking, gamma)

    # print('Initiate point')
    center_point = np.asarray([gamma.sum() / n_doc] * n_doc)
    initiate_fair_point = project_point_on_plane(center_point, item_group_masking, group_fairness)
    assert hedron.contains(initiate_fair_point), "Initiate point is not in the expohedron"

    # print("Start search for pareto front")

    end_point = gamma[invert_permutation(np.argsort(-relevance_score))]
    radius = norm(center_point - end_point)

    b = item_group_masking.T @ initiate_fair_point
    pareto_point = objs.convex_constraints_prob(b)
    assert hedron.contains(pareto_point), "Projection went wrong, new point is out of the hedron."

    basis_transform = BasisTransformer(center_point, null_space(expohedron_complement), compress=radius)
    sphere_coor = SphereCoordinator(center_point, radius, basis_transform, n_sample)

    t_starting_point, t_end_point = basis_transform.transform([sphere_coor.line_intersect_sphere(pareto_point), end_point])

    pareto_set = sphere_coor.geodesic_binary_approximate(n_divided, t_starting_point, t_end_point, objs, hedron)
    pareto_set = [pareto_point,] + pareto_set + [end_point,]
    objectives = [objs.objectives(point) for point in pareto_set]

    # print(objectives)
    return objectives, pareto_set
