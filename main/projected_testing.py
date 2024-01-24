import numpy as np
import pandas as pd

from scipy.linalg import null_space, norm, orth

from main.helpers import project_on_vector_space, invert_permutation, intersect_vector_space
from main.helpers import Objective, project_vector_on_subspace
from main.expohedron import find_face_subspace_without_parent_2, update_face_by_point, Expohedron


def projected_path(relevance_score: np.ndarray, item_group_masking: np.ndarray, group_fairness: np.ndarray, gamma: np.ndarray):
    n_doc, _ = item_group_masking.shape
    hedron = Expohedron(gamma)
    objs = Objective(relevance_score, group_fairness, item_group_masking, gamma)

    # print("Start search for pareto front")
    pareto_set = []
    objectives = []
    sum_gamma = -np.cumsum(np.sort(-gamma))

    nb_iteration = 0
    pareto_point = objs.optimal_utility_at_fairness_level(group_fairness)
    assert hedron.contains(pareto_point), "Projection went wrong, new point is out of the hedron."

    objectives.append(objs.objectives(pareto_point))
    pareto_set.append(pareto_point)

    min_utils = objs.utils(pareto_point)*1.0001
    another_point = objs.optimal_fairness_at_utility_level(min_utils)
    face_orth = find_face_subspace_without_parent_2(another_point, gamma)
    
    x = face_orth.sum(axis=0) - np.ones(face_orth.shape[1])
    _b = np.concatenate([group_fairness, sum_gamma[x.astype(int)]], axis=0)
    _A = np.concatenate([item_group_masking, face_orth], axis=1)
    in_point, residuals, rank, _ = np.linalg.lstsq(_A.T, _b, rcond=None)

    check_dir = another_point - in_point

    pareto_point = hedron.find_face_intersection_bisection(another_point, check_dir)

    face_orth = find_face_subspace_without_parent_2(pareto_point, gamma)
    next_face = face_orth

    pareto_set.append(pareto_point)
    objectives.append(objs.objectives(pareto_point))
    
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
            x = _face_orth.sum(axis=0) - np.ones(_face_orth.shape[1])
            _b = np.concatenate([group_fairness, sum_gamma[x.astype(int)]], axis=0)
            _A = np.concatenate([item_group_masking, _face_orth], axis=1)
            
            try:
                _c = objs.custom_optimal(_A, _b)
            except Exception as error:
                print(error)
                _c = None
            if _c is None:
                check_dir = project_on_vector_space(relevance_score, _face_orth.T)
            else:
                check_dir = -_c + pareto_point

            intersect_point = hedron.find_face_intersection_bisection(pareto_point, check_dir)
            # print(np.linalg.norm(intersect_point-pareto_point))
            if np.linalg.norm(intersect_point-pareto_point) < 1e-6:
                continue

            user_utilities = objs.utils(intersect_point)
            x1 = (user_utilities - pre_util) / (objs.unfairness(intersect_point) - objs.unfairness(pareto_point))
            x2 = (max_utils - pre_util) / (objs.unfairness(next_point) - objs.unfairness(pareto_point)) \
                if next_point is not None else 0
            
            if x1 > x2:
                max_utils = user_utilities
                next_point = intersect_point
                next_face = update_face_by_point(intersect_point, _face_orth, gamma)
                # next_face = find_face_subspace_without_parent_2(intersect_point, gamma)

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

    end_point = gamma[invert_permutation(np.argsort(-relevance_score))]
    pareto_set.append(end_point)
    objectives.append(objs.objectives(end_point))

    print(objectives)
    return objectives, pareto_set