import time
import numpy as np
from scipy.linalg import norm, null_space

from main.sphereCoordinator import SphereCoordinator, BasisTransformer
from main.expohedron import Expohedron
from main.helpers import Objective
from main.sphere_testing import sphere_path


def s_short_cut_accuracy(rel, item_group, group_fairness, gamma, n_divided, n_sample):
    n_doc = len(rel)
    start = time.time()
    s3_objs, s3_points = sphere_path(rel, item_group, group_fairness, gamma, n_divided, n_sample)
    center_point = np.asarray([gamma.sum() / n_doc] * n_doc)
    end = time.time()
    # print(len(s3_objs))

    expohedron_complement = np.asarray([[1.0] * n_doc])
    center_point = np.asarray([gamma.sum() / n_doc] * n_doc)
    radius = norm(center_point - gamma)

    hedron = Expohedron(gamma)
    objs = Objective(rel, group_fairness, item_group, gamma)
    basis_transform = BasisTransformer(center_point, null_space(expohedron_complement), compress=radius)
    sphere_coor = SphereCoordinator(center_point, radius, basis_transform, None)
    # print(end - start)

    total_point = np.power(2, n_divided) * (n_sample + 1) + 1
    # print(total_point)
    t_start, t_end = sphere_coor.basis_transform.transform([sphere_coor.line_intersect_sphere(s3_points[0]), s3_points[-1]])
    pareto_set = sphere_coor.geodesic_sample(t_start, t_end, total_point)
    pareto_set = sphere_coor.basis_transform.re_transform(pareto_set)
    spoints = []
    for point in pareto_set:
        spoints.append(hedron.find_face_intersection_bisection(center_point, point - center_point))
    sobjs = np.array([list(objs.objectives(spoint)) for spoint in spoints])

    mid = int((total_point - 1) / 2) + 1
    marked_points_ids = [0, mid, total_point-1]
    x = [sphere_coor.line_intersect_sphere(s3_points[i]) for i in marked_points_ids]
    t_point = sphere_coor.basis_transform.transform(x)
    s1_points = [s3_points[0]]
    for i in range(2):
        pareto_set = sphere_coor.geodesic_sample(t_point[i], t_point[i + 1], mid-1)
        pareto_set = list(sphere_coor.basis_transform.re_transform(pareto_set))
        for point in pareto_set:
            s1_points.append(hedron.find_face_intersection_bisection(center_point, point - center_point))
        s1_points.append(s3_points[marked_points_ids[i + 1]])
    s1_objs = np.array([list(objs.objectives(spoint)) for spoint in s1_points])

    quarter = int((mid - 1) / 2) + 1
    marked_points_ids = [0, quarter, mid, mid+quarter, total_point-1]
    x = [sphere_coor.line_intersect_sphere(s3_points[i]) for i in marked_points_ids]
    t_point = sphere_coor.basis_transform.transform(x)
    s2_points = [s3_points[0]]
    for i in range(4):
        pareto_set = sphere_coor.geodesic_sample(t_point[i], t_point[i + 1], quarter-1)
        pareto_set = list(sphere_coor.basis_transform.re_transform(pareto_set))
        for point in pareto_set:
            s2_points.append(hedron.find_face_intersection_bisection(center_point, point - center_point))
        s2_points.append(s3_points[marked_points_ids[i + 1]])
    s2_objs = np.array([list(objs.objectives(spoint)) for spoint in s2_points])

    s3_objs = np.array(s3_objs)

    return spoints, s1_points, s2_points, s3_points
    # plt.plot(sobjs[:, 1], sobjs[:, 0], label="Sphere-Expo_0", marker="o")
    # plt.plot(s1_objs[:, 1], s1_objs[:, 0], label="Sphere-Expo_1", marker="o")
    # plt.plot(s2_objs[:, 1], s2_objs[:, 0], label="Sphere-Expo_3", marker="o")
    # plt.plot(s3_objs[:, 1], s3_objs[:, 0], label="Sphere-Expo_7", marker="o")
    # plt.ylabel("User utility")
    # plt.xlabel("Unfairness")
    # plt.legend(loc='lower right')
    # plt.show()