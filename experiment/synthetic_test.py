import time
import json
from tqdm import tqdm

import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st
from scipy.linalg import norm, null_space

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from data.synthetic.load_data import load_data
from main.projected_testing import projected_path
from main.sphere_testing import sphere_path
from evaluation.exposure_to_bistochastic import get_pareto_point_for_scalarization
from evaluation.exposure_evaluation import evaluation, normalize_evaluation
from evaluation.sphere_evaluation import sphere_get_pareto_point_for_scalarization

from baseline import QP


class json_serialize(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def draw(curves, names):
    for i in range(len(names)):
        curve = np.array(curves[i])
        name = names[i]
        plt.plot(curve[:, 1], curve[:, 0], marker='o', label=name)

    plt.ylabel("User utility")
    plt.xlabel("Unfairness")
    plt.legend(loc='lower right')
    plt.show()


def single_test(n_doc, n_group):
    rel, item_group_masking, group_fairness, gamma = load_data(n_doc, n_group, 10)
    print(item_group_masking)

    p_obj, p_points = projected_path(rel, item_group_masking, group_fairness, gamma)
    # s_0_obj, s_0_points = sphere_path(rel, item_group_masking, group_fairness, gamma, 0, 20)
    # s_1_obj, s_1_points = sphere_path(rel, item_group_masking, group_fairness, gamma, 1, 10)
    # s_2_obj, s_2_points = sphere_path(rel, item_group_masking, group_fairness, gamma, 2, 5)
    s_3_obj, s_3_points = sphere_path(rel, item_group_masking, group_fairness, gamma, 3, 2)

    # baseline
    qp_objs, qp_points = QP.experiment(rel, item_group_masking, gamma, group_fairness, np.arange(1, 21) / 20)
    # 
    draw([qp_objs, p_obj, s_3_obj], ["baseline", "P-Expo", "Sphere-Expo_7"])

    # draw([qp_objs, s_0_obj, s_1_obj, s_2_obj, s_3_obj], 
    #      ["QP", "Sphere-Expo_0", "Sphere-Expo_1", "Sphere-Expo_3", "Sphere-Expo_7"])


def massive_running_time_test(n_item_group_range, repeated_time, file_name):
    variance_total_running_time = {
        "s_0_markers": {"bound": [], "mean": []},
        "s_1_markers": {"bound": [], "mean": []},
        "s_3_markers": {"bound": [], "mean": []},
        "s_7_markers": {"bound": [], "mean": []},
        "qp_bistochastic": {"bound": [], "mean": []},
        "qp_expohedron": {"bound": [], "mean": []},
        "p_expo": {"bound": [], "mean": []}
    }
    for n_doc, n_group in tqdm(n_item_group_range):
        total_running_time = {
            "s_0_markers": [],
            "s_1_markers": [],
            "s_3_markers": [],
            "s_7_markers": [],
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


def aggregation_test(n_doc_range):
    results = {
        "alpha": np.arange(1, 21) / 20,
        "qp": [[] for i in range(20)],
        "p_expo": [[] for i in range(20)],
        "s_expo_1": [[] for i in range(20)],
        "s_expo_2": [[] for i in range(20)],
        "s_expo_3": [[] for i in range(20)]
    }
    num_doc_list = set(n_doc_range)
    running_time = {
        "qp": {i:[] for i in num_doc_list},
        "p_expo": {i:[] for i in num_doc_list},
        "s_expo_1": {i:[] for i in num_doc_list},
        "s_expo_2": {i:[] for i in num_doc_list},
        "s_expo_3": {i:[] for i in num_doc_list}
    }
    for n_doc in n_doc_range:
        n_group = np.random.randint(3, n_doc/2)
        print(n_doc, n_group)
        # n_group = 3
        rel, item_group_masking, group_fairness, gamma = load_data(n_doc, n_group)

        start = time.time()
        qp_objs, qp_points = QP.experiment(rel, item_group_masking, gamma, group_fairness, results["alpha"])
        end = time.time()
        running_time["qp"][n_doc].append(end - start)

        start = time.time()
        p_obj, p_points = projected_path(rel, item_group_masking, group_fairness, gamma)
        end = time.time()
        running_time["p_expo"][n_doc].append(end - start)

        start = time.time()
        s_1_obj, s_1_points = sphere_path(rel, item_group_masking, group_fairness, gamma, 1, 3)
        end = time.time()
        running_time["s_expo_1"][n_doc].append(end - start)

        start = time.time()
        s_2_obj, s_2_points = sphere_path(rel, item_group_masking, group_fairness, gamma, 2, 3)
        end = time.time()
        running_time["s_expo_2"][n_doc].append(end - start)

        start = time.time()
        s_3_obj, s_3_points = sphere_path(rel, item_group_masking, group_fairness, gamma, 3, 3)
        end = time.time()
        running_time["s_expo_3"][n_doc].append(end - start)


        optimal_util_point = s_1_points[-1]
        # draw([qp_objs, p_obj], ["baseline", "P-Expo"])

        for i in range(0, len(results["alpha"])):
            alpha = results["alpha"][i]
            optimal = qp_points[i] @ gamma
            results["qp"][i].append(normalize_evaluation(optimal, rel, item_group_masking, group_fairness, optimal_util_point))

            point, objs = sphere_get_pareto_point_for_scalarization(p_points, gamma, group_fairness, optimal, rel, item_group_masking, optimal_util_point)
            results["p_expo"][i].append(objs)

            point, objs = sphere_get_pareto_point_for_scalarization(s_1_points, gamma, group_fairness, optimal, rel, item_group_masking, optimal_util_point)
            results["s_expo_1"][i].append(objs)
            point, objs = sphere_get_pareto_point_for_scalarization(s_2_points, gamma, group_fairness, optimal, rel, item_group_masking, optimal_util_point)
            results["s_expo_2"][i].append(objs)
            point, objs = sphere_get_pareto_point_for_scalarization(s_3_points, gamma, group_fairness, optimal, rel, item_group_masking, optimal_util_point)
            results["s_expo_3"][i].append(objs)
            # raise Exception("test")

    aggregation_result = {
        "qp": [],
        "p_expo": [],
        "s_expo_1": [],
        "s_expo_2": [],
        "s_expo_3": [],
    }

    aggregation_running_time = {
        "qp": [],
        "p_expo": [],
        "s_expo_1": [],
        "s_expo_2": [],
        "s_expo_3": [],
    }

    for key in aggregation_result:    
        for i in range(len(results["alpha"])):
            aggregation_result[key].append(np.mean(np.array(results[key][i]), axis=0))
        aggregation_result[key] = np.array(aggregation_result[key])

    num_doc_list = list(num_doc_list)
    num_doc_list.sort()
    for key in aggregation_running_time:
        for n_doc in num_doc_list:
            aggregation_running_time[key].append(np.array(running_time[key][n_doc]).mean())

    # print(aggregation_result["p_expo"])
    # print(aggregation_result["qp"])
    # print(aggregation_result["s_expo_3"])
    # print(aggregation_running_time["qp"])

    summary = {
        "accuracy": aggregation_result,
        "running_time": aggregation_running_time
    }
    with open('results/synthetic/small.json', "w") as log_file:
        json.dump(summary, log_file, cls=json_serialize)


    fig, ax = plt.subplots(nrows=1, ncols=2)

    ax[0].plot(aggregation_result["qp"][:, 1], aggregation_result["qp"][:, 0], "o-b", label="QP")
    ax[0].plot(aggregation_result["p_expo"][:, 1], aggregation_result["p_expo"][:, 0], "o-g", label="P-Expo")
    # ax[0].plot(aggregation_result["s_expo_1"][:, 1], aggregation_result["s_expo_1"][:, 0], "x--y", label="Sphere-Expo_1")
    # ax[0].plot(aggregation_result["s_expo_2"][:, 1], aggregation_result["s_expo_2"][:, 0], "x--m", label="Sphere-Expo_5")
    # ax[0].plot(aggregation_result["s_expo_3"][:, 1], aggregation_result["s_expo_3"][:, 0], "x--r", label="Sphere-Expo_7")
    ax[0].set_ylabel("Normalized User utility")
    ax[0].set_xlabel("Normalized Unfairness (nDCG)")
    ax[0].legend(loc='lower right')
    ax[0].set_title('Aggregated Pareto fronts')
    
    ax[1].plot(num_doc_list, aggregation_running_time["qp"], "o-b", label="QP")
    ax[1].plot(num_doc_list, aggregation_running_time["p_expo"], "o-g", label="P-Expo")
    # ax[1].plot(num_doc_list, aggregation_running_time["s_expo_1"], "x--y", label="Sphere-Expo_1")
    # ax[1].plot(num_doc_list, aggregation_running_time["s_expo_2"], "x--m", label="Sphere-Expo_5")
    # ax[1].plot(num_doc_list, aggregation_running_time["s_expo_3"], "x--r", label="Sphere-Expo_7")
    ax[1].set_ylabel("Avg running time (s)")
    ax[1].set_xlabel("Number of items")
    ax[1].legend(loc='lower right')
    ax[1].set_title('Aggregated running time')
    
    fig.suptitle("Aggregation results in a small synthetic dataset")
    plt.show()

if __name__ == "__main__":
    n_doc = 7
    n_group = 35
    # n_group = np.random.randint(3, n_doc-2)
    # print(n_group)
    single_test(n_doc, n_group)
    # n_docs = np.arange(8, 21)
    # # n_docs = 5 * np.arange(2, 21)
    # n_docs = np.repeat(n_docs, 3)
    # aggregation_test(n_docs)
    # # for n_doc in np.arange(8, 10):
    # #     n_group = np.random.randint(3, n_doc-2)
    # #     single_test(n_doc, n_group)
