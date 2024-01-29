import numpy as np
import pandas as pd

from scipy.linalg import null_space, norm

from .helpers import project_point_on_plane, invert_permutation
from .helpers import Objective
from .sphereCoordinator import BasisTransformer, SphereCoordinator
from .expohedron import Expohedron


HIGH_TOLERANCE = 1e-12


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
    end_point = objs.optimal_fairness_at_utility_level(objs.utils(end_point))
    radius = norm(center_point - end_point)

    b = item_group_masking.T @ initiate_fair_point
    pareto_point = objs.optimal_utility_at_fairness_level(b)
    assert hedron.contains(pareto_point), "Projection went wrong, new point is out of the hedron."

    basis_transform = BasisTransformer(center_point, null_space(expohedron_complement), compress=radius)
    sphere_coor = SphereCoordinator(center_point, radius, basis_transform, n_sample)

    t_starting_point, t_end_point = basis_transform.transform([sphere_coor.line_intersect_sphere(pareto_point), end_point])

    pareto_set = sphere_coor.geodesic_binary_approximate(n_divided, t_starting_point, t_end_point, objs, hedron)
    pareto_set = [pareto_point,] + pareto_set + [end_point,]
    objectives = [objs.objectives(point) for point in pareto_set]

    # print(objectives)
    return objectives, pareto_set
