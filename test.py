from main import projected_testing, projected_testing_test, sphere_testing, helpers
from baseline import QP
from data.synthetic.load_data import load_data
import numpy as np
import time
from itertools import permutations 
import matplotlib.pyplot as plt


def draw(curves, names):
    for i in range(len(names)):
        curve = np.array(curves[i])
        name = names[i]
        plt.plot(curve[:, 1], curve[:, 0], marker='o', label=name)

    plt.ylabel("User utility")
    plt.xlabel("Unfairness")
    plt.legend(loc='lower right')
    plt.show()

if __name__ == "__main__":
    # n_doc = 4
    # n_group = 2
    # gamma = np.array([4, 3, 2, 1])
    # rel = np.array([2, 5, 8, 4])
    # item_group_masking = np.array([[0, 1], [0, 1], [1, 0], [1, 0]])
    # group_fairness = [4.5, 5.5]
    # print(item_group_masking)
    n_doc = 20
    n_group = 8
    rel, item_group_masking, group_fairness, gamma = load_data(n_doc, n_group, 5)
    # print(item_group_masking)

    start = time.time()
    p_obj, p_points = projected_testing_test.projected_path(rel, item_group_masking, group_fairness, gamma)
    end = time.time()
    print(end - start)
    start = time.time()
    pc_obj, pc_points = projected_testing.projected_path(rel, item_group_masking, group_fairness, gamma)
    end = time.time()
    print(end - start)
    # s_0_obj, s_0_points = sphere_path(rel, item_group_masking, group_fairness, gamma, 0, 20)
    # s_1_obj, s_1_points = sphere_path(rel, item_group_masking, group_fairness, gamma, 1, 10)
    # s_2_obj, s_2_points = sphere_path(rel, item_group_masking, group_fairness, gamma, 2, 5)
    # s_3_obj, s_3_points = sphere_testing.sphere_path(rel, item_group_masking, group_fairness, gamma, 3, 3)

    # baseline
    qp_objs, qp_points = QP.experiment(rel, item_group_masking, gamma, group_fairness, np.arange(1, 21) / 20)
    draw([qp_objs, p_obj, pc_obj], ["baseline", "P-Expo", "New P Expo"])
    # x = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
    # y = np.array([[1, 1, 1, 1], [1, 0, -1, 0], [0, 1, 0, -1], [1, -1, 0, 0]])
    # print(np.linalg.inv(y).dot(x))
    
    # transistion_matrix = np.array([[0.25, 0, 0.5, -np.sqrt(2)/4], 
    #                                [0.25, 0.5, 0, np.sqrt(2)/4],
    #                                [0.25, 0, -0.5, -np.sqrt(2)/4],
    #                                [0.25, -0.5, 0, np.sqrt(2)/4]])
    # permu = permutations(gamma)
    # vertex = []
    # s_point = []
    # p_point = []
    # for point in list(permu):
    #     vertex.append(point @ transistion_matrix)
    # for point in s_3_points:
    #     s_point.append(point @ transistion_matrix)
    # for point in p_points:
    #     p_point.append(point @ transistion_matrix)
    
    # # print(qp_objs)
    # # print(np.array(vertex)[:, 1:])
    # # print(np.array(s_point)[:, 1:])
    # # print(np.array(p_point)[:, 1:])
    # rel = helpers.project_on_vector_space(rel, np.array([[1,1,1,1]]))
    # print(rel @ transistion_matrix)
    # test = np.array([2.5, 0.75, -1.25, 0.25]) @ np.linalg.inv(transistion_matrix)
    # print(item_group_masking.T @ test - group_fairness)
    # print(rel @ test.T)


    
