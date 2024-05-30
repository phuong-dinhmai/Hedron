import numpy as np
import pandas as pd
import json 
import matplotlib.pyplot as plt
import os
import sys
import time

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from main.projected_testing_test import projected_path
from main.sphere_testing import sphere_path
from baseline import QP

from evaluation.exposure_evaluation import normalize_evaluation, evaluation
from evaluation.exposure_to_bistochastic import get_pareto_point_for_scalarization
from evaluation.sphere_evaluation import sphere_get_pareto_point_for_scalarization

from sphere_shortcut_experiment import s_short_cut_accuracy


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


def experiment(query_file_path, item_group_file):
    query_file = open(query_file_path, "r", encoding="utf8")
    queries = [json.loads(line) for line in query_file.readlines()]
    queries = {query['qid']: query for query in queries}  # The data of unique queries

    all_group_masking = pd.read_csv(item_group_file, index_col=0)

    doc_id_dict = {}
    results = {
        "alpha": np.arange(1, 21) / 20,
        "qp": [[] for i in range(20)],
        "s_expo_1": [[] for i in range(20)],
        "s_expo_2": [[] for i in range(20)],
        "s_expo_3": [[] for i in range(20)],
    }
    norm_term = {}
    running_time = {}
    cnt = 0

    for qid in queries.keys():
        query = queries[qid]
        docs = query["documents"]
        n_doc = len(docs)
        rel = []
        if n_doc < 5 or n_doc > 100:
            continue

        group_matrix = np.zeros((n_doc, all_group_masking.shape[1]))
        for ord in range(n_doc):
            doc_id_dict[qid] = {docs[ord]["doc_id"]: ord}
            rel.append(docs[ord]["relevance"])
            # rel.append(np.random.rand())
            group_matrix[ord, :] = all_group_masking.loc[docs[ord]["doc_id"]]

        # print(rel)
        gamma = 1 / np.log(np.arange(0, n_doc) + 2)
        group_size = group_matrix.sum(axis=0)
        item_group_masking = np.delete(group_matrix, group_size == 0, 1)
        if (item_group_masking.shape[1] < 2 or item_group_masking.shape[1] == n_doc): 
            continue
        group_size = item_group_masking.sum(axis=0)
        group_fairness = group_size / np.sum(group_size) * np.sum(gamma)
        rel = np.array(rel)
        # print(group_matrix)
        print(qid, " ", n_doc)
        
        # p_obj, p_points = projected_path(rel, item_group_masking, group_fairness, gamma)
        q_start = time.time()
        qp_objs, qp_points = QP.experiment(rel, item_group_masking, gamma, group_fairness, results["alpha"])
        q_end = time.time()
        
        if np.linalg.norm(qp_objs[0][0] - qp_objs[-1][0]) < 1e-4: 
            print("skip utils")
            continue
        if np.linalg.norm(qp_objs[0][1] - qp_objs[-1][1]) < 1e-4: 
            print("skip unfair")
            continue
        # print(qp_objs)
        # raise Exception("test")

        s_start = time.time()
        spoints, s_1_points, s_2_points, s_3_points = s_short_cut_accuracy(rel, item_group_masking, group_fairness, gamma, 3, 2)
        s_end = time.time()
        optimal_util_point = s_1_points[-1]
        norm_term[qid] = evaluation(optimal_util_point, rel, item_group_masking, group_fairness, None)

        s_1_objs = [evaluation(point, rel, item_group_masking, group_fairness, None) for point in s_1_points]
        # s_2_objs = [evaluation(point, rel, item_group_masking, group_fairness, None) for point in s_2_points]
        s_3_objs = [evaluation(point, rel, item_group_masking, group_fairness, None)for point in s_3_points]
        # print(qid)
        # draw([qp_objs, s_2_objs], ["baseline", "Sphere-Expo_2"])

        qp_objs, qp_points = qp_objs[:-1], qp_points[:-1]
        qp_objs.append(s_3_objs[-1])
        qp_points.append(None)

        running_time[str(qid)] = [q_end-q_start, s_end-s_start]

        for i in range(0, len(results["alpha"])):
            alpha = results["alpha"][i]
            if alpha != 1:
                optimal = qp_points[i] @ gamma
            else:
                optimal = optimal_util_point
            # point, objs = get_pareto_point_for_scalarization(p_points, group_fairness, alpha, rel, item_group_masking, optimal_util_point)
            # results["p_expo"][i].append(objs)
            # optimal = qp_points[i] @ gamma
            results["qp"][i].append(normalize_evaluation(optimal, rel, item_group_masking, group_fairness, optimal_util_point))

            point, objs = sphere_get_pareto_point_for_scalarization(s_1_points, gamma, group_fairness, optimal, rel, item_group_masking, optimal_util_point)
            results["s_expo_1"][i].append(objs)
            point, objs = sphere_get_pareto_point_for_scalarization(s_2_points, gamma, group_fairness, optimal, rel, item_group_masking, optimal_util_point)
            results["s_expo_2"][i].append(objs)
            point, objs = sphere_get_pareto_point_for_scalarization(s_3_points, gamma, group_fairness, optimal, rel, item_group_masking, optimal_util_point)
            results["s_expo_3"][i].append(objs)

        cnt += 1
        # break

    aggregation_result = {
        "qp": [],
        "s_expo_1": [],
        "s_expo_2": [],
        "s_expo_3": [],
    }

    for key in aggregation_result:    
        for i in range(len(results["alpha"])):
            aggregation_result[key].append(np.mean(np.array(results[key][i]), axis=0))
        aggregation_result[key] = np.array(aggregation_result[key])


    print(cnt)
    summary = {
        "accuracy": aggregation_result,
        "norm": norm_term,
        "running_time": running_time
    }
    with open("results/TREC/aggregate.json", "w") as f_out:
        json.dump(summary, f_out, cls=json_serialize)

    # plt.plot(aggregation_result["p_expo"][:, 1], aggregation_result["p_expo"][:, 0], "o-g",, label="P-Expo")
    plt.plot(aggregation_result["qp"][:, 1], aggregation_result["qp"][:, 0], "o-b", label="QP")
    plt.plot(aggregation_result["s_expo_1"][:, 1], aggregation_result["s_expo_1"][:, 0], "x--y", label="Sphere-Expo_1")
    plt.plot(aggregation_result["s_expo_2"][:, 1], aggregation_result["s_expo_2"][:, 0], "x--m", label="Sphere-Expo_5")
    plt.plot(aggregation_result["s_expo_3"][:, 1], aggregation_result["s_expo_3"][:, 0], "x--r", label="Sphere-Expo_7")
    plt.ylabel("Normalized User utility")
    plt.xlabel("Normalized Unfairness")
    plt.legend(loc='lower right')
    plt.show()

if __name__ == "__main__":
    group_file = "data/TREC/year_group.csv"
    queries_file = "data/TREC/year_queries.json"

    experiment(queries_file, group_file)