import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time
import pickle
from scipy.linalg import null_space, orth, norm
import scipy.stats as ss
import cvxpy as cp
import geomstats.backend as gs
from geomstats.geometry.hypersphere import Hypersphere, HypersphereMetric
import torch

from helpers import project_point_on_plane, project_on_vector_space
from helpers import majorized, find_face_intersection_bisection, invert_permutation
from expohedron import find_face_subspace_without_parent_2, identify_face

from BPR.evaluate import get_relevance

import QP


HIGH_TOLERANCE = 1e-12


class BasisTransformer:

    def __init__(self, point, basis_vector, compress) -> None:
        """
        Change basis of coordinator - reduce dimension in case point cloud is proved to be in the same hyper plane
        New coordinator will have basis vector is np.eye(dim)
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
        :type points: np.ndarray, list
        """
        return np.asarray(points) @ self.M / self.compress

    def re_transform(self, points):
        """
        Return transformed points original coordinate
        :param points: each row is the coordinate of a point
        :type points: np.ndarray, list
        """
        _points = np.asarray(points)
        n_point = _points.shape[0]
        _points = np.concatenate((_points * self.compress, np.ones((n_point, self.original_dim-self.transformed_dim))),
                                 axis=1)

        return _points @ self.M_inv


def draw(a, b):
    a = np.asarray(a)
    b = np.asarray(b)
    b = b[b[:, 1].argsort()]
    a = a[a[:, 1].argsort()]

    plt.plot(b[:, 1], b[:, 0], label="check")
    plt.plot(a[:, 1], a[:, 0], label="optimal")
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

    end_point = gamma[np.argsort(-relevance_score)]
    unfairness = np.sum((item_group_masking.T @ end_point - group_fairness) ** 2)
    user_utilities = relevance_score @ end_point
    print(user_utilities)
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
            break
        nb_iteration += 1
        pareto_set.append(next_point)
        unfairness = np.sum((item_group_masking.T @ next_point - group_fairness) ** 2)
        user_utilities = relevance_score @ next_point
        objectives.append([user_utilities, unfairness])
        pareto_point = next_point
        print(nb_iteration)
        # print(user_utilities)

        # break
    print(objectives)
    return objectives, pareto_set


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

    end_point = gamma[invert_permutation(np.argsort(-relevance_score))]
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

    t_starting_point, t_end_point, t_center = basis_transform.transform(
        [line_intersect_sphere(center_point, radius, pareto_point), end_point, center_point])
    sphere = Hypersphere(dim=n_doc-1)

    geodesic_func = sphere.metric.geodesic(initial_point=t_starting_point, end_point=t_end_point)
    points = geodesic_func(gs.linspace(0, 1, 25))

    for point in points:
        ori_point = basis_transform.re_transform([point])[0]
        intersect_point = find_face_intersection_bisection(gamma, center_point, ori_point-center_point)

        unfairness = np.sum((item_group_masking.T @ intersect_point - group_fairness) ** 2)
        user_utilities = relevance_score @ intersect_point
        objectives.append([user_utilities, unfairness])
        pareto_set.append(intersect_point)

    print(objectives)
    return objectives, pareto_set


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

    n_doc = 40
    n_group = 8

    np.random.seed(n_doc)
    relevance_score = np.random.rand(n_doc)
    # np.savetxt("data_error/relevance_score.csv", relevance_score, delimiter=",")

    item_group_masking = np.zeros((n_doc, n_group))
    x = np.arange(-n_group/2, n_group/2)
    xU, xL = x + 0.5, x - 0.5
    prob = ss.norm.cdf(xU, scale=3) - ss.norm.cdf(xL, scale=3)
    prob = prob / prob.sum()
    for i in range(n_doc):
        # j = np.random.randint(n_group, size=1)
        j = np.random.choice(range(n_group), size=1, p=prob)
        item_group_masking[i][j[0]] = 1
    cnt_col = item_group_masking.sum(axis=0)
    item_group_masking = np.delete(item_group_masking, cnt_col == 0, 1)
    # np.savetxt("data_error/item_group.csv", item_group_masking, delimiter=",")

    gamma = 1 / np.log(np.arange(0, n_doc) + 2)
    group_size = item_group_masking.sum(axis=0)
    group_fairness = group_size / np.sum(group_size) * np.sum(gamma)

    return relevance_score, item_group_masking, group_fairness, gamma


def test(relevance_score, item_group_masking, group_fairness, gamma, points):
    n_doc, n_group = item_group_masking.shape

    expohedron_complement = np.asarray([[1.0] * n_doc])
    center_point = np.asarray([gamma.sum() / n_doc] * n_doc)
    end_point = gamma[invert_permutation(np.argsort(-relevance_score))]
    radius = norm(center_point - end_point)

    basis_transform = BasisTransformer(center_point, null_space(expohedron_complement), compress=radius)
    # t_start, t_center, t_endpoint = basis_transform.transform([points[0], center_point, end_point])
    sphere = Hypersphere(dim=n_doc-1)
    # for i in range(1, len(points)):
        # b = item_group_masking.T @ point
        # pareto_point = convex_constraints_prob(relevance_score, item_group_masking, gamma, b)
        # x = identify_face(gamma, pareto_point)
        # y = identify_face(gamma, point)
        # dir = points[i] - points[i-1]
        # intersect_start = lineline_intersect_sphere(center_point, radius, points[i-1])
        # intersect_start = basis_transform.transform([intersect_start])
        # intersect_end = line_intersect_sphere(center_point, radius, points[i])
        # intersect_end = basis_transform.transform([intersect_end])
        # tangent_vec = sphere.metric.log(base_point=intersect_start, point=intersect_end)
        # geodesic = sphere.metric.geodesic(initial_point=intersect_start, end_point=intersect_end)
        # sample = geodesic(gs.linspace(0, 1, 5))
        # face_1 = find_face_subspace_without_parent_2(points[i-1], gamma)
        # face_2 = find_face_subspace_without_parent_2(points[i], gamma)
        #
        # aset = set([tuple(x) for x in face_1.T])
        # bset = set([tuple(x) for x in face_2.T])
        # face = np.array([x for x in aset & bset])
        # print(face_1.shape)
        # print(face_2.shape)
        # print(face.shape)
        #
        # for x in range(1, len(sample)):
        #     p = basis_transform.re_transform([sample[x]])[0]
        #     # print((p-center_point).shape)
        #     intersec = find_face_intersection_bisection(gamma, center_point, p-center_point)
        #     k = ((intersec-points[i-1]) / norm(intersec-points[i-1])) - (dir / norm(dir))
        #     # print(np.any(np.abs(k) > 1e-9))
        #     print(k)

        # c = sphere.to_tangent(intersect_start, basis_transform.transform(dir))
        # print("-----------------------------")

    for j in range(1, len(points)):
        intersect_points = []
        mid_point = (points[j] + points[j-1]) / 2
        check_point = (points[j-1], mid_point, points[j])
        for i in range(0, 3):
            point = check_point[i]
            face = identify_face(gamma, point)
            # print(face.splits)
            # print(face.zone)
            # print("----------------------")
            face = find_face_subspace_without_parent_2(point, gamma)
            p_center = project_point_on_plane(center_point, face, face.T @ point)
            dir = p_center - center_point
            up = radius
            down = 0
            k = (up + down) / 2
            while True:
                p = point + k*dir
                dis = norm(p-center_point)
                if np.abs(dis - radius) < 1e-9:
                    break
                if dis > radius:
                    up = k
                else:
                    down = k
                k = (up + down) / 2
            intersect_points.append(point + k*dir)

        intersect_points = basis_transform.transform(intersect_points)

        d = sphere.metric.dist(intersect_points[0], intersect_points[2])
        d_1 = sphere.metric.dist(intersect_points[0], intersect_points[1])
        d_2 = sphere.metric.dist(intersect_points[1], intersect_points[2])
        print(d - d_1 - d_2)


def movielen100k_testing():
    from BPR.model import BPR

    with open("/home/phuong/Documents/expohedron/BPR/output/data.pkl", 'rb') as f:
        dataset = pickle.load(f)
        user_size, item_size = dataset['user_size'], dataset['item_size']
        train_user_list, test_user_list = dataset['train_user_list'], dataset['test_user_list']
        # train_pair = dataset['train_pair']

    model = BPR(user_size, item_size, dim=512, weight_decay=0.025)
    model.load_state_dict(torch.load("/home/phuong/Documents/expohedron/BPR/output/bpr.pt"))
    model.eval()

    relevance_matrix, item_idx = get_relevance(50, model.W, model.H, train_user_list, batch=512)
    item_idx = item_idx.numpy()
    relevance_matrix = relevance_matrix.detach().numpy()
    data_anal = pd.read_csv("/home/phuong/Documents/expohedron/BPR/output/data_analysis.csv", index_col=0)
    
    gamma = 1 / np.log(np.arange(0, 50) + 2)
    
    for i in range(relevance_matrix.shape[0]):
        idx = item_idx[i]
        group_masking = data_anal[["popular", "unpopular"]].loc[idx].to_numpy()
        group_size = group_masking.sum(axis=0)
        group_fairness = group_size / np.sum(group_size) * np.sum(gamma)
        optimal = example2(relevance_matrix[i], item_group_masking=group_masking, group_fairness=group_fairness, gamma=gamma)
        sphere_pareto = sphere_check(relevance_matrix[i], item_group_masking=group_masking, group_fairness=group_fairness, gamma=gamma)
        draw(optimal[0], sphere_pareto[0])
        # draw(sphere_pareto[0], optimal[0])
        break


if __name__ == "__main__":
    # print("Load data")
    # _relevance_score, item_group, _group_fairness, _gamma = load_data()
    # print("Start hedron experiment:")
    # hedron_start = time.time()
    # # objs, points = sphere_check(_relevance_score, item_group, _group_fairness, _gamma)
    # objs, points = example2(_relevance_score, item_group, _group_fairness, _gamma)
    # test(_relevance_score, item_group, _group_fairness, _gamma, points)

    # print(len(points))
    # hedron_end = time.time()

    # print("Start QP experiment:")
    # qp_start = time.time()
    # base_qp = QP.experiment(_relevance_score, item_group)
    # qp_end = time.time()
    # print("Done")
    # print((hedron_end - hedron_start) / len(objs))
    # print(hedron_end - hedron_start)
    # print((qp_end - qp_start) / len(base_qp))
    # print(qp_end - qp_start)
    # draw(objs, base_qp)

    movielen100k_testing()