import numpy as np
from scipy.optimize import minimize, Bounds

from evaluation.exposure_evaluation import normalize_evaluation, evaluation


def scalarized_objective_within_pareto_segment(alpha: float, scalarization: float,
                                               point1: np.ndarray, point2: np.ndarray,
                                               target_exposure: np.ndarray,
                                               relevance_vector: np.ndarray,
                                               item_list: np.ndarray,
                                               optimal_util_point: np.ndarray) -> float:
    """
        given a convex combination of two exposure vectors, computes the value of the scalarized objective

            min α (-U) + (1-α) F

    :param alpha: The convex combination parameter of the two exposure vectors
    :type alpha: float
    :param scalarization: The scalarization parameter
    :type scalarization: float
    :param point1: An exposure vector
    :type point1: numpy.ndarray
    :param point2: An exposure vector
    :type point2: numpy.ndarray
    :param target_exposure: The expected group exposure
    :type target_exposure: numpy.ndarray
    :param relevance_vector: The item relevance vector
    :type relevance_vector: numpy.ndarray
    :param item_list: The group binary matrix
    :type item_list: numpy.ndarray
    :param optimal_util_point: The expected exposure at optimal utils point
    :type optimal_util_point: numpy.ndarray
    :return: The value of the objective function
    :rtype: float
    """
    exposure = alpha * point1 + (1-alpha) * point2
    n_utils, n_unfair = normalize_evaluation(exposure, relevance_vector, item_list, target_exposure, optimal_util_point)
    return scalarization * (-n_utils) + (1-scalarization) * n_unfair


def get_pareto_point_for_scalarization(pareto_curve: list, target_exposure: np.ndarray,
                                       alpha: float, relevance_vector: np.ndarray,
                                       item_list: np.ndarray, optimal_util_point: np.ndarray) -> tuple:
    """
        Given a Pareto front in an expohedron, and a scalarization parameter `alpha`
        computes the optimum of the scalarized problem

            min α (-U) + (1-α) F

    :param pareto_curve: The pareto curve in the expohedron
    :type pareto_curve: list
    :param target_exposure: The target exposure vector
    :type target_exposure: numpy.ndarray
    :param relevance_vector: The item relevance vector
    :type relevance_vector: numpy.ndarray
    :param item_list: The group binary matrix
    :type item_list: numpy.ndarray
    :param optimal_util_point: The expected exposure at optimal utils point
    :type optimal_util_point: numpy.ndarray
    :param alpha: The scalarization parameter
    :type alpha: float
    :return: The optimal utility, the optimal unfairness and the optimal exposure
    :rtype: Tuple[float, float, numpy.ndarray]
    """

    def objective(exposure):
        n_utils, n_unfair = normalize_evaluation(exposure, relevance_vector, item_list, target_exposure,
                                                 optimal_util_point)
        return alpha * (-n_utils) + (1-alpha) * n_unfair

    bounds = Bounds(0, 1)
    # Find the line segment on which the optimal exposure lies
    if len(pareto_curve) == 1:  # pathological case
        exposure_opt = pareto_curve[0]
        return normalize_evaluation(exposure_opt, relevance_vector, item_list, target_exposure, optimal_util_point)
    for i in np.arange(0, len(pareto_curve)-1):
        o1 = objective(pareto_curve[i])
        o2 = objective(pareto_curve[i+1])
        if o2 > o1:
            break  # optimal point is in line segment [i, i+1]
    # try:
    #     a = pareto_curve[i]
    # except UnboundLocalError:
    #     a = 1
    sol = minimize(scalarized_objective_within_pareto_segment, 0,
                   (alpha, pareto_curve[i], pareto_curve[i+1], target_exposure,
                    relevance_vector, item_list, optimal_util_point),
                   method="Nelder-Mead", bounds=bounds)
    exposure_opt = sol.x[0] * pareto_curve[i] + (1-sol.x[0]) * pareto_curve[i+1]
    assert objective(exposure_opt) == sol.fun
    return exposure_opt, normalize_evaluation(exposure_opt, relevance_vector, item_list, target_exposure, optimal_util_point)
