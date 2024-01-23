import time
import json
from tqdm import tqdm

import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st

from data.synthetic.load_data import load_data
from main.projected_testing import projected_path
from main.sphere_testing import sphere_path
from evaluation.exposure_to_bistochastic import get_pareto_point_for_scalarization
from evaluation.exposure_evaluation import evaluation, normalize_evaluation

from baseline import QP


def draw(check_curve, optimal_curve):
    check_curve = np.asarray(check_curve)
    optimal_curve = np.asarray(optimal_curve)
    optimal_curve = optimal_curve[optimal_curve[:, 1].argsort()]
    # check_curve = check_curve[check_curve[:, 1].argsort()]

    plt.plot(optimal_curve[:, 1], optimal_curve[:, 0], label="optimal_curve")
    plt.plot(check_curve[:, 1], check_curve[:, 0], label="check_curve")
    plt.ylabel("User utility")
    plt.xlabel("Unfairness")
    plt.legend(loc='lower right')
    plt.show()


def single_test(n_doc, n_group):
    rel, item_group_masking, group_fairness, gamma = load_data(n_doc, n_group)

    # obj, points = projected_path(rel, item_group_masking, group_fairness, gamma)
    obj, points = sphere_path(rel, item_group_masking, group_fairness, gamma, 2)

    # baseline
    qp_objs, qp_points = QP.experiment(rel, item_group_masking, group_fairness, np.arange(1, 21) / 20)
    draw(obj, qp_objs)


def s_short_cut_accuracy(n_doc, n_group):
    rel, item_group, group_fairness, gamma = load_data(n_doc, n_group)

    start = time.time()
    s3_objs, s3_points, sphere_coor, objs, hedron = sphere_path(rel, item_group,
                                                                group_fairness,
                                                                gamma, n_divided=3,
                                                                n_sample=6)
    center_point = np.asarray([gamma.sum() / n_doc] * n_doc)
    end = time.time()
    print(end - start)

    t_start = sphere_coor.basis_transform.transform([s3_points[0]])[0]
    t_end = sphere_coor.basis_transform.transform([s3_points[56]])[0]
    pareto_set = sphere_coor.geodesic_sample(t_start, t_end, 55)
    pareto_set = list(sphere_coor.basis_transform.re_transform(pareto_set))
    spoints = [s3_points[0]]
    for point in pareto_set:
        spoints.append(hedron.find_face_intersection_bisection(center_point, point - center_point))
    spoints.append(s3_points[56])
    _sobjs = np.array([list(objs.objectives(spoint)) for spoint in spoints])
    sobjs = [_sobjs[0]]
    for i in range(0, len(_sobjs) - 1):
        if _sobjs[i, 0] > _sobjs[i + 1, 0] or _sobjs[i, 1] > _sobjs[i + 1, 1]:
            continue
        sobjs.append(_sobjs[i + 1])
    sobjs = np.array(sobjs)

    marked_points_ids = [0, 28, 56]
    x = [s3_points[i] for i in marked_points_ids]
    t_point = sphere_coor.basis_transform.transform(x)
    s1_points = [s3_points[0]]
    for i in range(2):
        pareto_set = sphere_coor.geodesic_sample(t_point[i], t_point[i + 1], 27)
        pareto_set = list(sphere_coor.basis_transform.re_transform(pareto_set))
        for point in pareto_set:
            s1_points.append(hedron.find_face_intersection_bisection(center_point, point - center_point))
        s1_points.append(s3_points[marked_points_ids[i + 1]])
    _s1_objs = np.array([list(objs.objectives(spoint)) for spoint in s1_points])
    s1_objs = [_s1_objs[0]]
    for i in range(0, len(_s1_objs) - 1):
        if _s1_objs[i, 0] > _s1_objs[i + 1, 0] or _s1_objs[i, 1] > _s1_objs[i + 1, 1]:
            continue
        s1_objs.append(_s1_objs[i + 1])
    s1_objs = np.array(s1_objs)

    marked_points_ids = [0, 14, 28, 42, 56]
    x = [s3_points[i] for i in marked_points_ids]
    t_point = sphere_coor.basis_transform.transform(x)
    s2_points = [s3_points[0]]
    for i in range(4):
        pareto_set = sphere_coor.geodesic_sample(t_point[i], t_point[i + 1], 13)
        pareto_set = list(sphere_coor.basis_transform.re_transform(pareto_set))
        for point in pareto_set:
            s2_points.append(hedron.find_face_intersection_bisection(center_point, point - center_point))
        s2_points.append(s3_points[marked_points_ids[i + 1]])
    _s2_objs = np.array([list(objs.objectives(spoint)) for spoint in s2_points])
    s2_objs = [_s2_objs[0]]
    for i in range(0, len(_s1_objs) - 1):
        if _s2_objs[i, 0] > _s2_objs[i + 1, 0] or _s2_objs[i, 1] > _s2_objs[i + 1, 1]:
            continue
        s2_objs.append(_s2_objs[i + 1])
    s2_objs = np.array(s2_objs)

    s3_objs = np.array(s3_objs)
    plt.plot(sobjs[:, 1], sobjs[:, 0], label="SphereExpo_0_marker", marker="o")
    plt.plot(s1_objs[:, 1], s1_objs[:, 0], label="SphereExpo_1_marker", marker="o")
    # plt.plot(s1_check_objs[:, 1], s1_check_objs[:, 0], label="SphereExpo_1_check")
    plt.plot(s2_objs[:, 1], s2_objs[:, 0], label="SphereExpo_2_marker", marker="o")
    plt.plot(s3_objs[:, 1], s3_objs[:, 0], label="SphereExpo_3_marker", marker="o")
    plt.ylabel("User utility")
    plt.xlabel("Unfairness")
    plt.legend(loc='lower right')
    plt.show()


def massive_running_time_test(n_item_group_range, repeated_time, file_name):
    variance_total_running_time = {
        "s_0_markers": {"bound": [], "mean": []},
        "s_1_markers": {"bound": [], "mean": []},
        "s_3_markers": {"bound": [], "mean": []},
        "s_5_markers": {"bound": [], "mean": []},
        "qp_bistochastic": {"bound": [], "mean": []},
        "qp_expohedron": {"bound": [], "mean": []},
        "p_expo": {"bound": [], "mean": []}
    }
    for n_doc, n_group in tqdm(n_item_group_range):
        total_running_time = {
            "s_0_markers": [],
            "s_1_markers": [],
            "s_3_markers": [],
            "s_5_markers": [],
            "qp_bistochastic": [],
            "qp_expohedron": [],
            "p_expo": []
        }
        for t in tqdm(range(repeated_time)):
            rel, item_group_masking, group_fairness, gamma = load_data(n_doc, n_group)

            # start = time.time()
            # obj, points = projected_path(rel, item_group_masking, group_fairness, gamma)
            # end = time.time()
            # total_running_time["p_expo"].append([(end-start), len(obj)])

            n_samples = [55, 27, 13, 6]
            for n_divided in range(0, 4):
                start = time.time()
                obj, points = sphere_path(rel, item_group_masking, group_fairness, gamma,
                                          n_divided, n_samples[n_divided])
                end = time.time()
                total_running_time[f"s_{pow(2, n_divided)-1}_markers"].append([(end-start), len(obj)])

            # baseline
            start = time.time()
            qp_objs, qp_points = QP.experiment(rel, item_group_masking, group_fairness, np.arange(1, 56) / 55)
            end = time.time()
            total_running_time["qp_bistochastic"].append([(end-start), len(qp_objs)])

        for key in total_running_time:
            arr = total_running_time[key]
            variance_total_running_time[key]["bound"] = st.norm.interval(confidence=0.90, loc=np.mean(arr),
                                                                         scale=st.sem(arr))
            variance_total_running_time[key]["avg"] = np.mean(arr)

    with open(f"../results/{file_name}_result.json", "w") as f_out:
        json.dump(variance_total_running_time, f_out)

    with open(f"../data/synthetic/{file_name}_n_item_group_range.json", "w") as f_out:
        json.dump(n_item_group_range, f_out)


if __name__ == "__main__":
    # single_test(10, 5)
    # s_short_cut_accuracy(10, 5)
    from main.helpers import invert_permutation

    rel, item_group_masking, group_fairness, _gamma = load_data(10, 5)

    obj, points = projected_path(rel, item_group_masking, group_fairness, _gamma)
    # obj, points = sphere_path(rel, item_group_masking, group_fairness, _gamma, 2)

    # baseline
    alpha_arr = np.arange(1, 21) / 20
    qp_objs, qp_points = QP.experiment(rel, item_group_masking, _gamma, group_fairness, alpha_arr)

    optimal_util_point = _gamma[invert_permutation(np.argsort(-rel))]
    for i in range(len(alpha_arr)):
        alpha = alpha_arr[i]
        point, normalize_objs = get_pareto_point_for_scalarization(points, group_fairness, alpha, rel,
                                                                   item_group_masking, optimal_util_point)
        opt_exposure = qp_points[i] @ _gamma
        expo_util, expo_unfair = normalize_evaluation(point, rel, item_group_masking,
                                                      group_fairness, optimal_util_point)
        opt_util, opt_unfair = normalize_evaluation(opt_exposure, rel, item_group_masking,
                                                    group_fairness, optimal_util_point)
        print(expo_util - opt_util, " ", expo_unfair - opt_unfair)
        print(alpha * opt_util - (1-alpha) * opt_unfair, " ", alpha * expo_util - (1-alpha) * expo_unfair)


