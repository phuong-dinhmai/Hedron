import numpy as np
from scipy.linalg import null_space, norm

from .exposure_evaluation import normalize_evaluation, evaluation, utils, unfairness
from main.sphereCoordinator import BasisTransformer, SphereCoordinator
from main.expohedron import Expohedron


def sphere_get_pareto_point_for_scalarization(pareto_curve, gamma, group_fairness, target_exposure, rel, item_group_masking, optimal_util_point):
    # Find the line segment on which the optimal exposure lies
    if len(pareto_curve) == 1:  # pathological case
        exposure_opt = pareto_curve[0]
        print("here")
        return exposure_opt, normalize_evaluation(exposure_opt, rel, item_group_masking, group_fairness, optimal_util_point)
    
    target_utils = utils(target_exposure, rel)

    up_bound = None
    low_bound = None
    for i in np.arange(1, len(pareto_curve)):
        o1 = utils(pareto_curve[i], rel)
        if o1 >= target_utils:
            low_bound = pareto_curve[i-1]
            up_bound = pareto_curve[i]
            break  # optimal point is in line segment [i, i+1]
        
    if low_bound is None:
        return pareto_curve[-1], normalize_evaluation(pareto_curve[-1], rel, item_group_masking, group_fairness, optimal_util_point)
    
    n_doc = len(rel)
    expohedron_complement = np.asarray([[1.0] * n_doc])
    center_point = np.asarray([gamma.sum() / n_doc] * n_doc)
    radius = norm(center_point - target_exposure)

    hedron = Expohedron(gamma)
    basis_transform = BasisTransformer(center_point, null_space(expohedron_complement), compress=radius)
    sphere_coor = SphereCoordinator(center_point, radius, basis_transform, None)

    # print(evaluation(low_bound, rel, item_group_masking, group_fairness, optimal_util_point))
    # print(evaluation(target_exposure, rel, item_group_masking, group_fairness, optimal_util_point))
    # print(evaluation(up_bound, rel, item_group_masking, group_fairness, optimal_util_point))

    low_bound, up_bound = basis_transform.transform([sphere_coor.line_intersect_sphere(low_bound), 
                                                     sphere_coor.line_intersect_sphere(up_bound)])

    while True:
        mid = sphere_coor.geodesic_sample(low_bound, up_bound, 3)[1]
        intersect = sphere_coor.basis_transform.re_transform([mid])[0]
        intersect = hedron.find_face_intersection_bisection(center_point, intersect-center_point)

        f = utils(intersect, rel)
        # print(f)
        # r_low_bound, r_up_bound = sphere_coor.basis_transform.re_transform([low_bound, up_bound])
        # r_low_bound = hedron.find_face_intersection_bisection(center_point, r_low_bound-center_point)
        # r_up_bound = hedron.find_face_intersection_bisection(center_point, r_up_bound-center_point)
        if norm(up_bound - low_bound) < 1e-6:
            return intersect, normalize_evaluation(intersect, rel, item_group_masking, group_fairness, optimal_util_point)
        if f > target_utils:
            up_bound = mid
        else:
            low_bound = mid