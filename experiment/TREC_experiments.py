import numpy as np
import pandas as pd
import json 
import matplotlib.pyplot as plt
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from main.projected_testing import projected_path
from main.sphere_testing import sphere_path
from baseline import QP

from evaluation.exposure_evaluation import normalize_evaluation, evaluation
from evaluation.exposure_to_bistochastic import get_pareto_point_for_scalarization
from evaluation.sphere_evaluation import sphere_get_pareto_point_for_scalarization

from sphere_shortcut_experiment import s_short_cut_accuracy


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
    cnt = 10

    for qid in queries.keys():
        query = queries[qid]
        docs = query["documents"]
        n_doc = len(docs)
        rel = []

        group_matrix = np.zeros((n_doc, all_group_masking.shape[1]))
        for ord in range(n_doc):
            doc_id_dict[qid] = {docs[ord]["doc_id"]: ord}
            # rel.append(docs[ord]["relevance"])
            rel.append(np.random.rand())
            group_matrix[ord, :] = all_group_masking.loc[docs[ord]["doc_id"]]

        # print(rel)
        gamma = 1 / np.log(np.arange(0, n_doc) + 2)
        group_size = group_matrix.sum(axis=0)
        item_group_masking = np.delete(group_matrix, group_size == 0, 1)
        group_size = item_group_masking.sum(axis=0)
        group_fairness = group_size / np.sum(group_size) * np.sum(gamma)
        rel = np.array(rel)
        # print(group_matrix)
        # print(group_size)
        
        # p_obj, p_points = projected_path(rel, item_group_masking, group_fairness, gamma)
        qp_objs, qp_points = QP.experiment(rel, item_group_masking, gamma, group_fairness, results["alpha"])
        
        if np.linalg.norm(qp_objs[0][0] - qp_objs[-1][0]) < 1e-4: 
            print("skip utils")
            continue
        if np.linalg.norm(qp_objs[0][1] - qp_objs[-1][1]) < 1e-4: 
            print("skip unfair")
            continue
        # print(qp_objs)
        # raise Exception("test")
        cnt -= 1
        spoints, s_1_points, s_2_points, s_3_points = s_short_cut_accuracy(rel, item_group_masking, group_fairness, gamma, 3, 2)
        optimal_util_point = s_1_points[-1]

        s_1_objs = [evaluation(point, rel, item_group_masking, group_fairness, None) for point in s_1_points]
        s_2_objs = [evaluation(point, rel, item_group_masking, group_fairness, None) for point in s_2_points]
        s_3_objs = [evaluation(point, rel, item_group_masking, group_fairness, None)for point in s_3_points]
        # draw([qp_objs, s_2_objs], ["baseline", "Sphere-Expo_2"])

        qp_objs, qp_points = qp_objs[:-1], qp_points[:-1]
        qp_objs.append(s_3_objs[-1])
        qp_points.append(None)

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

        if cnt == 0:
            break

    aggregation_result = {
        "qp": [],
        # "p_expo": [],
        "s_expo_1": [],
        "s_expo_2": [],
        "s_expo_3": [],
    }

    for i in range(len(results["alpha"])):
        aggregation_result["qp"].append(np.mean(np.array(results["qp"][i]), axis=0))
        # aggregation_result["p_expo"].append(np.mean(np.array(results["p_expo"][i]), axis=0))
        aggregation_result["s_expo_1"].append(np.mean(np.array(results["s_expo_1"][i]), axis=0))
        aggregation_result["s_expo_2"].append(np.mean(np.array(results["s_expo_2"][i]), axis=0))
        aggregation_result["s_expo_3"].append(np.mean(np.array(results["s_expo_3"][i]), axis=0))


    aggregation_result["qp"] = np.array(aggregation_result["qp"])
    # aggregation_result["p_expo"] = np.array(aggregation_result["p_expo"])
    aggregation_result["s_expo_1"] = np.array(aggregation_result["s_expo_1"])
    aggregation_result["s_expo_2"] = np.array(aggregation_result["s_expo_2"])
    aggregation_result["s_expo_3"] = np.array(aggregation_result["s_expo_3"])
    # print(aggregation_result["p_expo"])
    # print(aggregation_result["qp"])
    # print(aggregation_result["s_expo_3"])

    # with open("results/TREC_result.json", "w") as f_out:
    #     json.dump(aggregation_result.tolist(), f_out)

    # plt.plot(aggregation_result["p_expo"][:, 1], aggregation_result["p_expo"][:, 0], marker='o', label="P-Expo")
    plt.plot(aggregation_result["qp"][:, 1], aggregation_result["qp"][:, 0], marker='o', label="QP")
    plt.plot(aggregation_result["s_expo_1"][:, 1], aggregation_result["s_expo_1"][:, 0], marker='o', label="Sphere-Expo_1")
    plt.plot(aggregation_result["s_expo_2"][:, 1], aggregation_result["s_expo_2"][:, 0], marker='o', label="Sphere-Expo_5")
    plt.plot(aggregation_result["s_expo_3"][:, 1], aggregation_result["s_expo_3"][:, 0], marker='o', label="Sphere-Expo_7")
    plt.ylabel("Normalized User utility")
    plt.xlabel("Normalized Unfairness")
    plt.legend(loc='lower right')
    plt.show()

if __name__ == "__main__":
    group_file = "D:\VinUni_researchAssistant\Hedron\data\TREC\year_group.csv"
    queries_file = "D:\VinUni_researchAssistant\Hedron\data\TREC\year_queries.json"

    experiment(queries_file, group_file)