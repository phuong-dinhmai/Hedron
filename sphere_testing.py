import matplotlib.pyplot as plt
import numpy as np
import time
from scipy.linalg import null_space, orth, norm
import cvxpy as cp
import geomstats.backend as gs
from geomstats.geometry.hypersphere import Hypersphere, HypersphereMetric

from helpers import project_point_on_plane, project_on_vector_space
from helpers import majorized, find_face_intersection_bisection

from expohedron_face import find_face_subspace_without_parent_2

import QP



HIGH_TOLERANCE = 1e-12


class BasisTransformer:
    
    def __init__(self, point, basis_vector, compress) -> None:
        """
        Change basis of coordinator - reduce dimension in case point cloud is proved to be in the same hyper plane 
        New coorordinator will have basis vector is np.eye(dim)
        :param point: origin of new coordinate O(0, ..., 0)
        :type point: np.ndarray
        :param basis_vector: basis vector of hyper plane which all point belongs to 
        :type basis_vector: np.ndarray
        """
        self.original_dim, self.transformed_dim = basis_vector.shape
        self.compress = compress

        new_points = []
        for i in range(self.transformed_dim):
            _point = point + basis_vector[:, i]
            new_points.append(_point)
        new_points.append(point)
        S = np.asarray(new_points)
        D = np.eye(self.transformed_dim)

        D = np.concatenate((D, np.zeros((self.original_dim-self.transformed_dim, self.transformed_dim))))
        self.M = np.linalg.inv(S) @ D

        D = np.concatenate((D, np.ones((self.original_dim, self.original_dim-self.transformed_dim))), axis=1)
        self.M_inv = np.linalg.inv(D) @ S
        
    def transform(self, points):
        """
        Return points new coordinate 
        :param points: each row is the coordinate of a point
        :type points: np.ndarray
        """
        return np.asarray(points) @ self.M / self.compress

    def re_transform(self, points):
        """
        Return transformed points original coordinate
        :param points: each row is the coordinate of a point
        :type points: np.ndarray
        """
        _points = np.asarray(points)
        n_point = _points.shape[0]
        _points = np.concatenate((_points * self.compress, np.ones((n_point, self.original_dim-self.transformed_dim))), axis=1)

        return _points @ self.M_inv 


def draw(a, b):
    a = np.asarray(a)
    b = np.asarray(b)
    b = b[b[:, 1].argsort()]
    a = a[a[:, 1].argsort()]

    plt.plot(b[:, 1], b[:, 0], label="QP")
    plt.plot(a[:, 1], a[:, 0], label="Hedron")
    plt.ylabel("User utility")
    plt.xlabel("Unfairness")
    plt.legend(loc='lower right')
    plt.show()


def line_intersect_sphere(center_point, radius, outside_point):
    dir = outside_point - center_point
    dis = norm(dir)
    return center_point + dir * (radius / dis)


def example2(relevance_score: np.ndarray, item_group_masking: np.ndarray, group_fairness: np.ndarray, gamma: np.ndarray):
    n_doc, n_group = item_group_masking.shape

    expohedron_complement = np.asarray([[1.0] * n_doc])
    expohedron_basis = null_space(expohedron_complement, HIGH_TOLERANCE)

    # print("Fairness_level_direction_space")
    # Since the vector space is orthogonal with a subspace in the expohedron space
    # The intersection space will also be the projection space
    # fairness_level_projection_space = intersect_vector_space(expohedron_basis, item_group_masking)
    # optimal_fairness_direction = project_vector_on_subspace(relevance_score,
    #                                                         fairness_level_projection_space)

    # Random point in fairness surface
    print('Initiate point')
    center_point = np.asarray([gamma.sum() / n_doc] * n_doc)
    initiate_fair_point = project_point_on_plane(center_point, item_group_masking, group_fairness)
    assert majorized(initiate_fair_point, gamma), "Initiate point is not in the expohedron"

    print("Start search for pareto front")
    pareto_set = []
    objectives = []

    nb_iteration = 0
    starting_point = initiate_fair_point
    b = item_group_masking.T @ starting_point
    pareto_point = convex_constraints_prob(relevance_score, item_group_masking, gamma, b)
    assert majorized(pareto_point, gamma), "Projection went wrong, new point is out of the hedron."

    unfairness = np.sum((item_group_masking.T @ pareto_point - group_fairness) ** 2)
    user_utilities = relevance_score @ pareto_point
    objectives.append([user_utilities, unfairness])

    # end_point = gamma[np.argsort(-relevance_score)]
    # optimal_fairness_direction = project_vector_on_subspace(end_point-pareto_point, fairness_level_projection_space)
    # optimal_fairness_direction = project_vector_on_subspace(end_point - pareto_point, item_group_masking)
    optimal_fairness_direction = relevance_score
    # print(end_point - pareto_point)
    # print(optimal_fairness_direction)

    # return []
    while True:
        # face = identify_face(gamma, pareto_point)
        # face_orth = find_face_subspace_without_parent(face)
        face_orth = find_face_subspace_without_parent_2(pareto_point, gamma)
        if not majorized(pareto_point, gamma):
            break
        # print(optimal_fairness_direction)
        # print(face_orth)
        n_row, n_col = face_orth.shape
        max_utils = relevance_score @ pareto_point
        # print("Current optimal: ", max_utils)
        next_point = None
        for exclude in range(-1, n_col-1):
            _face_orth = face_orth[:, np.arange(n_col) != exclude]
            # optimal_fairness_direction = project_vector_on_subspace(relevance_score, item_group_masking)
            # optimal_fairness_direction = project_vector_on_subspace(end_point - pareto_point, item_group_masking)
            check_dir = project_on_vector_space(optimal_fairness_direction, _face_orth.T)

            if np.all(np.abs(check_dir) < 1e-9):
                continue
            check_dir[np.abs(check_dir) < 1e-9] = 0

            intersect_point = find_face_intersection_bisection(gamma, pareto_point, check_dir)

            if not majorized(intersect_point, gamma):
                continue
            user_utilities = relevance_score @ intersect_point
            # print(user_utilities)
            if user_utilities - max_utils > 1e-6:
                max_utils = user_utilities
                next_point = intersect_point
        if next_point is None:
            print(nb_iteration)
            break
        nb_iteration += 1
        pareto_set.append(next_point)
        unfairness = np.sum((item_group_masking.T @ next_point - group_fairness) ** 2)
        user_utilities = relevance_score @ next_point
        objectives.append([user_utilities, unfairness])
        pareto_point = next_point
        # print(nb_iteration)
        # print(user_utilities)

        # break
    return objectives


def sphere_check(relevance_score: np.ndarray, item_group_masking: np.ndarray, group_fairness: np.ndarray, gamma: np.ndarray):
    n_doc, n_group = item_group_masking.shape

    expohedron_complement = np.asarray([[1.0] * n_doc])
    
    print('Initiate point')
    center_point = np.asarray([gamma.sum() / n_doc] * n_doc)
    initiate_fair_point = project_point_on_plane(center_point, item_group_masking, group_fairness)
    assert majorized(initiate_fair_point, gamma), "Initiate point is not in the expohedron"

    print("Start search for pareto front")
    pareto_set = []
    objectives = []

    end_point = gamma[np.argsort(-relevance_score)]
    radius = norm(center_point - end_point)
    starting_point = initiate_fair_point
    b = item_group_masking.T @ starting_point
    pareto_point = convex_constraints_prob(relevance_score, item_group_masking, gamma, b)
    assert majorized(pareto_point, gamma), "Projection went wrong, new point is out of the hedron."

    unfairness = np.sum((item_group_masking.T @ pareto_point - group_fairness) ** 2)
    user_utilities = relevance_score @ pareto_point
    objectives.append([user_utilities, unfairness])
    pareto_set.append(pareto_point)
    
    basis_transform = BasisTransformer(center_point, null_space(expohedron_complement), compress=radius)
    t_starting_point, t_end_point = basis_transform.transform([line_intersect_sphere(center_point, radius, pareto_point), end_point])


    sphere = Hypersphere(dim=n_doc-1)
    project_dir = project_on_vector_space(relevance_score, expohedron_complement)
    starting_tangent_vec = sphere.to_tangent(basis_transform.transform(project_dir), t_starting_point)
    # geodesic_func = sphere.metric.geodesic(initial_point=t_starting_point, initial_tangent_vec=starting_tangent_vec) 
    geodesic_func = sphere.metric.geodesic(initial_point=t_starting_point, end_point=t_end_point)
    
    points = geodesic_func(gs.linspace(0, 1, 25))
    for point in points:
        ori_point = basis_transform.re_transform([point])[0]
        intersect_point = find_face_intersection_bisection(gamma, center_point, ori_point-center_point)

        unfairness = np.sum((item_group_masking.T @ intersect_point - group_fairness) ** 2)
        user_utilities = relevance_score @ intersect_point
        objectives.append([user_utilities, unfairness])
        pareto_set.append(intersect_point)

    # print(pareto_set)
    print(objectives)
    return objectives



def convex_constraints_prob(relevance_score, item_group_masking, gamma, group_fairness):
    n_doc, n_group = item_group_masking.shape
    gamma_sum = np.cumsum(gamma)
    vars = cp.Variable(n_doc)
    constrs = [cp.sum_largest(vars, i) <= gamma_sum[i-1] for i in range(1, n_doc)]
    constrs.append(item_group_masking.T @ vars == group_fairness)
    obj_func = cp.Maximize(cp.sum(relevance_score.T @ vars))
    prob = cp.Problem(obj_func, constrs)
    prob.solve(verbose=False)  # Returns the optimal value.
    # print("status:", prob.status)
    if prob.status == cp.OPTIMAL:
        return vars.value
    return None


def load_data():
    # n_doc = 4
    # n_group = 2
    # relevance_score = np.asarray([0.7, 0.8, 1, 0.4])
    # item_group_masking = np.asarray([[0, 1], [0, 1], [1, 0], [1, 0]])
    # gamma = np.asarray([4, 3, 2, 1])
    # group_fairness = np.asarray([6.5, 3.5])

    # relevance_score = np.loadtxt("data_error/relevance_score.csv", delimiter=",").astype(np.double)
    # item_group_masking = np.loadtxt("data_error/item_group.csv", delimiter=",").astype(np.double)
    # n_doc = item_group_masking.shape[0]

    n_doc = 6
    n_group = 2

    np.random.seed(n_doc)
    relevance_score = np.random.rand(n_doc)
    # np.savetxt("data_error/relevance_score.csv", relevance_score, delimiter=",")

    item_group_masking = np.zeros((n_doc, n_group))
    for i in range(n_doc):
        j = np.random.randint(n_group, size=1)
        item_group_masking[i][j[0]] = 1
    cnt_col = item_group_masking.sum(axis=0)
    item_group_masking = np.delete(item_group_masking, cnt_col == 0, 1)
    # np.savetxt("data_error/item_group.csv", item_group_masking, delimiter=",")

    gamma = 1 / np.log(np.arange(0, n_doc) + 2)
    group_size = item_group_masking.sum(axis=0)
    group_fairness = group_size / np.sum(group_size) * np.sum(gamma)

    return relevance_score, item_group_masking, group_fairness, gamma


if __name__ == "__main__":
    # point = np.asarray((7/3, 7/3, 7/3))
    # expohedron_complement = np.asarray([[1.0] * 3])
    # basis = null_space(expohedron_complement)
    # points = np.asarray([(1, 2, 4), (2, 1, 4), (4, 2, 1), (4, 1, 2), (2, 4, 1)])
    # transform = BasisTransformer(point, basis, compress=norm(point - points[0]))

    # _p = transform.transform(points)
    # print(_p)
    # print(transform.re_transform(_p))

    print("Load data")
    _relevance_score, item_group, _group_fairness, _gamma = load_data()
    print("Start hedron experiment:")
    hedron_start = time.time()
    objs = sphere_check(_relevance_score, item_group, _group_fairness, _gamma)
    # objs = example2(_relevance_score, item_group, _group_fairness, _gamma)
    hedron_end = time.time()

    print("Start QP experiment:")
    qp_start = time.time()
    base_qp = QP.experiment(_relevance_score, item_group)
    qp_end = time.time()
    print("Done")
    print((hedron_end - hedron_start) / len(objs))
    print((qp_end - qp_start) / len(base_qp))
    draw(objs, base_qp)
