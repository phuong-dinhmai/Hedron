import matplotlib.pyplot as plt
import numpy as np
import time
from scipy.linalg import null_space
import cvxpy as cp

from helpers import project_vector_on_subspace, project_point_on_plane
from helpers import majorized, invert_permutation, intersect_vector_space

from expohedron import caratheodory_decomposition_pbm_gls

import QP


HIGH_TOLERANCE = 1e-12


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


def example(relevance_score: np.ndarray, item_group_masking: np.ndarray, group_fairness: np.ndarray, gamma: np.ndarray):
    n_doc = item_group_masking.shape[0]
    n_group = item_group_masking.shape[1]

    expohedron_complement = np.asarray([[1.0] * n_doc])
    expohedron_basis = null_space(expohedron_complement, HIGH_TOLERANCE)
    group_dict = [[] for j in range(n_group)]
    for i in range(n_doc):
        g = np.where(item_group_masking[i, :] == 1)[0][0]
        group_dict[g].append(i)

    print("Fairness_level_direction_space")
    # Since the vector space is orthogonal with a subspace in the expohedron space
    # The intersection space will also be the projection space
    fairness_level_projection_space = intersect_vector_space(expohedron_basis, item_group_masking)

    # Random point in fairness surface
    print('Initiate point')
    center_point = np.asarray([gamma.sum() / n_doc] * n_doc)
    initiate_fair_point = project_point_on_plane(center_point, item_group_masking, group_fairness)
    assert majorized(initiate_fair_point, gamma), "Initiate point is not in the expohedron"

    end_point = gamma[invert_permutation(np.argsort(-relevance_score))]
    # _t = linear_programming_check(relevance_score, item_group_masking, gamma, item_group_masking.T @ end_point)
    # print(relevance_score @ _t[:n_doc])
   
    print("Optimal_fairness_direction")
    # TODO: This direction do not lead to optimal level in L1
    optimal_fairness_direction = project_vector_on_subspace(end_point-initiate_fair_point,
                                                            fairness_level_projection_space)
    # optimal_fairness_direction = end_point - initiate_fair_point
    # optimal_fairness_direction = project_vector_on_subspace(relevance_score, fairness_level_projection_space)

    optimal_fairness_direction /= np.linalg.norm(optimal_fairness_direction)
    direction = relevance_score


    print("Start search for pareto front")
    pareto_set = []
    objectives = []

    step = 0.1
    nb_iteration = 1

    # return []
    while True:
        starting_point = initiate_fair_point + (nb_iteration * step) * optimal_fairness_direction
        nb_iteration += 1
        if not majorized(starting_point, gamma):
            print(nb_iteration)
            break
        b = item_group_masking.T @ starting_point
        pareto_point = convex_constraints_prob(relevance_score, item_group_masking, gamma, b)
        assert majorized(pareto_point, gamma), "Projection went wrong, new point is out of the hedron."

        unfairness = np.sum((item_group_masking.T @ pareto_point - group_fairness) ** 2)
        user_utilities = relevance_score @ pareto_point

        pareto_set.append(pareto_point)
        objectives.append([user_utilities, unfairness])
        # break
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

    # relevance_score = np.loadtxt("data_error/relevance_score.csv", delimiter=",").astype(np.double)
    # item_group_masking = np.loadtxt("data_error/item_group.csv", delimiter=",").astype(np.double)
    # n_doc = item_group_masking.shape[0]

    n_doc = 100
    n_group = 40

    np.random.seed(n_doc)
    relevance_score = np.random.rand(n_doc)
    # np.savetxt("data_error/relevance_score.csv", relevance_score, delimiter=",")

    item_group_masking = np.zeros((n_doc, n_group))
    for i in range(n_doc):
        j = np.random.randint(n_group, size=1)
        item_group_masking[i][j[0]] = 1
    # np.savetxt("data_error/item_group.csv", item_group_masking, delimiter=",")

    gamma = 1 / np.log(np.arange(0, n_doc) + 2)
    # gamma = np.asarray([4, 3, 2, 1])
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
