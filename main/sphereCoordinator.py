import geomstats.backend as gs
from geomstats.geometry.hypersphere import Hypersphere

import numpy as np
from scipy.linalg import norm


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


class SphereCoordinator:
    def __init__(self, center_point, radius, basis_transformer, n_point):
        self.center_point = center_point
        dim = len(center_point)
        self.radius = radius
        self.hypersphere = Hypersphere(dim=dim-1)
        self.n_sample_in_geodesic = n_point
        self.basis_transform = basis_transformer

    def line_intersect_sphere(self, outside_point):
        dir = outside_point - self.center_point
        dis = norm(dir)
        return self.center_point + dir * (self.radius / dis)

    def geodesic_binary_approximate(self, n_divided, s_start_point, s_end_point, objs, hedron):
        result = []
        if n_divided == 0:
            s_points = self.geodesic_sample(s_start_point, s_end_point, self.n_sample_in_geodesic)
            s_points = self.basis_transform.re_transform(s_points)
            for point in s_points:
                result.append(hedron.find_face_intersection_bisection(self.center_point, point-self.center_point))
            return result

        mid = self.geodesic_sample(s_start_point, s_end_point, 3)[1]
        mid = self.basis_transform.re_transform([mid])[0]
        corrected_point, s_mid_point = self.post_correction_point_2(mid, hedron, objs)
        # corrected_point, s_mid_point = self.post_correction_point(mid, hedron, objs)
        s_mid_point = self.basis_transform.transform([s_mid_point])[0]
        result += self.geodesic_binary_approximate(n_divided-1, s_start_point, s_mid_point, objs, hedron)
        result.append(corrected_point)
        result += self.geodesic_binary_approximate(n_divided-1, s_mid_point, s_end_point, objs, hedron)
        return result

    def geodesic_sample(self, s_starting_point, s_end_point, n_point=3):
        geodesic_func = self.hypersphere.metric.geodesic(initial_point=s_starting_point, end_point=s_end_point)
        return geodesic_func(gs.linspace(0, 1, n_point))

    def post_correction_point(self, sphere_point, hedron, objs):
        intersect = hedron.find_face_intersection_bisection(self.center_point, sphere_point-self.center_point)
        b = objs.group_masking.T @ intersect
        pareto_point = objs.optimal_utility_at_fairness_level(b)
        revert = self.line_intersect_sphere(pareto_point)
        # print(norm(revert-sphere_point))
        return pareto_point, revert

    def post_correction_point_2(self, sphere_point, hedron, objs):
        intersect = hedron.find_face_intersection_bisection(self.center_point, sphere_point - self.center_point)
        objectives = objs.objectives(intersect)
        pareto_point = objs.optimal_fairness_at_utility_level(objectives[0])
        revert = self.line_intersect_sphere(pareto_point)
        # print(norm(revert-sphere_point))
        return pareto_point, revert
