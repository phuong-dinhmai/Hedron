import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import pickle
from scipy.linalg import null_space, norm

import scipy.stats as ss
import time

from helpers import project_point_on_plane, project_on_vector_space, invert_permutation
from helpers import Objective
from sphereCoordinator import BasisTransformer, SphereCoordinator
from expohedron import find_face_subspace_without_parent_2, Expohedron

from BPR.evaluate import get_relevance

import QP


HIGH_TOLERANCE = 1e-12



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


def example2(relevance_score: np.ndarray, item_group_masking: np.ndarray, group_fairness: np.ndarray, gamma: np.ndarray):
    n_doc, _ = item_group_masking.shape
    hedron = Expohedron(gamma)
    objs = Objective(relevance_score, group_fairness, item_group_masking, gamma)

    # Random point in fairness surface
    print('Initiate point')
    center_point = np.asarray([gamma.sum() / n_doc] * n_doc)
    initiate_fair_point = project_point_on_plane(center_point, item_group_masking, group_fairness)
    assert hedron.contains(initiate_fair_point), "Initiate point is not in the expohedron"

    print("Start search for pareto front")
    pareto_set = []
    objectives = []

    nb_iteration = 0
    starting_point = initiate_fair_point
    b = item_group_masking.T @ starting_point
    pareto_point = objs.convex_constraints_prob(b)
    assert hedron.contains(pareto_point), "Projection went wrong, new point is out of the hedron."

    objectives.append(objs.objectives(pareto_point))

    optimal_fairness_direction = relevance_score

    while True:
        face_orth = find_face_subspace_without_parent_2(pareto_point, gamma)
        if not hedron.contains(pareto_point):
            break
        _, n_col = face_orth.shape
        max_utils = objs.utils(pareto_point)
        # print("Current optimal: ", max_utils)
        next_point = None
        for exclude in range(-1, n_col-1):
            _face_orth = face_orth[:, np.arange(n_col) != exclude]
            check_dir = project_on_vector_space(optimal_fairness_direction, _face_orth.T)

            if np.all(np.abs(check_dir) < 1e-9):
                continue
            check_dir[np.abs(check_dir) < 1e-9] = 0

            intersect_point = hedron.find_face_intersection_bisection(pareto_point, check_dir)

            if not hedron.contains(intersect_point):
                continue
            user_utilities = objs.utils(intersect_point)
            if user_utilities - max_utils > 1e-9:
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

    print(objectives)
    return objectives, pareto_set


def sphere_check(relevance_score: np.ndarray, item_group_masking: np.ndarray, group_fairness: np.ndarray, gamma: np.ndarray, n_divided=3, n_sample=5):
    n_doc, _ = item_group_masking.shape

    expohedron_complement = np.asarray([[1.0] * n_doc])
    hedron = Expohedron(gamma)
    objs = Objective(relevance_score, group_fairness, item_group_masking, gamma)

    print('Initiate point')
    center_point = np.asarray([gamma.sum() / n_doc] * n_doc)
    initiate_fair_point = project_point_on_plane(center_point, item_group_masking, group_fairness)
    assert hedron.contains(initiate_fair_point), "Initiate point is not in the expohedron"

    print("Start search for pareto front")

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

    print(objectives)
    return objectives, pareto_set


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

    n_doc = 50
    n_group = 2

    np.random.seed(n_doc)
    relevance_score = np.arange(1, 51) / 50
    # print(relevance_score)
    # relevance_score = np.random.rand(n_doc)
    # np.savetxt("data_error/relevance_score.csv", relevance_score, delimiter=",")

    item_group_masking = np.zeros((n_doc, n_group))
    item_group_masking[:10, 0] = 1
    item_group_masking[10:, 1] = 1
    # x = np.arange(-n_group/2, n_group/2)
    # xU, xL = x + 0.5, x - 0.5
    # prob = ss.norm.cdf(xU, scale=3) - ss.norm.cdf(xL, scale=3)
    # prob = prob / prob.sum()
    # for i in range(n_doc):
    #     j = np.random.randint(n_group, size=1)
    #     # j = np.random.choice(range(n_group), size=1, p=prob)
    #     item_group_masking[i][j[0]] = 1
    cnt_col = item_group_masking.sum(axis=0)
    item_group_masking = np.delete(item_group_masking, cnt_col == 0, 1)
    # np.savetxt("data_error/item_group.csv", item_group_masking, delimiter=",")

    gamma = 1 / np.log(np.arange(0, n_doc) + 2)
    group_size = item_group_masking.sum(axis=0)
    group_fairness = group_size / np.sum(group_size) * np.sum(gamma)

    return relevance_score, item_group_masking, group_fairness, gamma


def movielen100k_testing():
    from BPR.model import BPR

    with open("BPR/output/data.pkl", 'rb') as f:
        dataset = pickle.load(f)
        user_size, item_size = dataset['user_size'], dataset['item_size']
        train_user_list, test_user_list = dataset['train_user_list'], dataset['test_user_list']
        # train_pair = dataset['train_pair']

    model = BPR(user_size, item_size, dim=512, weight_decay=0.025)
    model.load_state_dict(torch.load("BPR/output/bpr.pt"))
    model.eval()

    relevance_matrix, item_idx = get_relevance(50, model.W, model.H, train_user_list, batch=512)
    item_idx = item_idx.numpy()
    relevance_matrix = relevance_matrix.detach().numpy()
    data_anal = pd.read_csv("BPR/output/data_analysis.csv", index_col=0)
    
    gamma = 1 / np.log(np.arange(0, 50) + 2)
    
    for i in range(relevance_matrix.shape[0]):
        idx = item_idx[i]
        # print(relevance_matrix[i])
        group_masking = data_anal[["popular", "unpopular"]].loc[idx].to_numpy()
        group_size = group_masking.sum(axis=0)
        group_fairness = group_size / np.sum(group_size) * np.sum(gamma)
        optimal = example2(relevance_matrix[i], item_group_masking=group_masking, group_fairness=group_fairness, gamma=gamma)
        sphere_pareto = sphere_check(relevance_matrix[i], item_group_masking=group_masking, group_fairness=group_fairness, gamma=gamma, n_divided=3)
        base_qp = QP.experiment(relevance_matrix[i], group_masking)
        draw(base_qp[-5:], sphere_pareto[0])
        # draw(base_qp[-5:], optimal[0])
        break


if __name__ == "__main__":
    print("Load data")
    _relevance_score, item_group, _group_fairness, _gamma = load_data()
    print("Start hedron experiment:")
    hedron_start = time.time()
    objs, points = sphere_check(_relevance_score, item_group, _group_fairness, _gamma, n_divided=0, n_sample=25)
    # objs, points = example2(_relevance_score, item_group, _group_fairness, _gamma)
    # test(_relevance_score, item_group, _group_fairness, _gamma, points)
    
    # print(len(points))
    hedron_end = time.time()
    
    print("Start QP experiment:")
    qp_start = time.time()
    base_qp = QP.experiment(_relevance_score, item_group)
    qp_end = time.time()
    print("Done")
    print((hedron_end - hedron_start) / len(objs))
    print(hedron_end - hedron_start)
    print((qp_end - qp_start) / len(base_qp))
    print(qp_end - qp_start)
    draw(base_qp[5:], objs)

    # movielen100k_testing()