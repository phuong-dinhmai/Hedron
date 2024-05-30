import os
import sys
import time
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import pandas as pd
import numpy as np
import json
from sklearn.metrics import ndcg_score
import matplotlib.pyplot as plt

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


def df_transform(df):
    df[1] = df[1].apply(lambda x: x[4:])
    df[df.columns[2:-1]] = df[df.columns[2:-1]].applymap(lambda x: x.split(':')[1])
    df = df.drop(138, axis=1)
    return df


def experiment(file_path):
    train = pd.read_csv(file_path, header=None, sep=" ")
    train_df = df_transform(train)
    train_df = train_df.astype(float)
    train_df[1] = train_df[1].astype(int)

    x = train_df[1].unique()
    bins = [i for i in range(0, 270, 10)]
    labels = [i for i in range(0, 26)]
    train_df['binned'] = pd.cut(train_df[133], bins=bins, labels=labels)

    results = {
        "alpha": np.arange(1, 21) / 20,
        "qp": [[] for i in range(20)],
        "s_expo_1": [[] for i in range(20)],
        "s_expo_2": [[] for i in range(20)],
        "s_expo_3": [[] for i in range(20)],
    }
    normalize_term = {}
    running_time = {}
    cnt = 0

    for qid in x[:1]:
        items = train_df.loc[train_df[1] == qid, :]
        rel = items[0] / 5
        group = items["binned"]
        n_doc = len(rel)

        if len(rel.unique()) == 1 or len(group.unique()) == 1 or n_doc >= 100 or n_doc < 5:
            continue

        group_masking = np.zeros((len(rel), len(labels)))
        for i in range(n_doc):
            group_masking[i][group.iloc[i]] = 1

        # gamma = 1 / np.log(np.arange(0, n_doc) + 2)
        gamma = 1 / (np.arange(0, n_doc) + 1)
        group_size = group_masking.sum(axis=0)
        item_group_masking = np.delete(group_masking, group_size == 0, 1)
        if (item_group_masking.shape[1] < 2) or (item_group_masking.shape[1] == n_doc): 
            continue

        group_size = item_group_masking.sum(axis=0)
        group_fairness = group_size / np.sum(group_size) * np.sum(gamma)
        rel = rel.to_numpy()
        print(group_fairness)

        q_start = time.time()
        qp_objs, qp_points = QP.experiment(rel, item_group_masking, gamma, group_fairness, results["alpha"])
        q_end = time.time()

        if qp_objs is None:
            continue
        if np.linalg.norm(qp_objs[0][0] - qp_objs[-1][0]) < 1e-3: 
            print("skip utils")
            continue
        if np.linalg.norm(qp_objs[0][1] - qp_objs[-1][1]) < 1e-3: 
            print("skip unfair")
            continue
        
        print(qid, " ", n_doc)
        try:
            s_start = time.time()
            spoints, s_1_points, s_2_points, s_3_points = s_short_cut_accuracy(rel, item_group_masking, group_fairness, gamma, 3, 2)
            s_end = time.time()
        except Exception as error:
            print("Sphere error at: ", qid)
            continue
        s3_objs = [evaluation(point, rel, item_group_masking, group_fairness, None) for point in s_3_points]
        draw([qp_objs, s3_objs], ["baseline", "Sphere-Expo_5"])
        optimal_util_point = s_3_points[-1]

        qp_objs, qp_points = qp_objs[:-1], qp_points[:-1]
        qp_objs.append(s3_objs[-1])
        qp_points.append(None)
        normalize_term[str(qid)] = evaluation(optimal_util_point, rel, item_group_masking, group_fairness, None)
        print(normalize_term[str(qid)])
        running_time[str(qid)] = [q_end - q_start, s_end - s_start]

        for i in range(0, len(results["alpha"])):
            alpha = results["alpha"][i]
            # point, objs = get_pareto_point_for_scalarization(p_points, group_fairness, alpha, rel, item_group_masking, optimal_util_point)
            # results["p_expo"][i].append(objs)
            qp_objs[i] = list(qp_objs[i])
            if alpha != 1:     
                if qp_objs[i][0] < qp_objs[i-1][0]:
                    qp_objs[i][0] = qp_objs[i-1][0]
                if qp_objs[i][1] < qp_objs[i-1][1]:
                    qp_objs[i][1] = qp_objs[i-1][1]         
                optimal = qp_points[i] @ gamma
            else:
                # continue
                optimal = optimal_util_point
            results["qp"][i].append(normalize_evaluation(optimal, rel, item_group_masking, group_fairness, optimal_util_point))

            point, objs = sphere_get_pareto_point_for_scalarization(s_1_points, gamma, group_fairness, optimal, rel, item_group_masking, optimal_util_point)
            results["s_expo_1"][i].append(objs)
            point, objs = sphere_get_pareto_point_for_scalarization(s_2_points, gamma, group_fairness, optimal, rel, item_group_masking, optimal_util_point)
            results["s_expo_2"][i].append(objs)
            point, objs = sphere_get_pareto_point_for_scalarization(s_3_points, gamma, group_fairness, optimal, rel, item_group_masking, optimal_util_point)
            results["s_expo_3"][i].append(objs)

        cnt += 1

    aggregation_result = {
        "qp": [],
        # "p_expo": [],
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
        "norm": normalize_term,
        "running_time": running_time
    }
    # with open("results/MSLR/aggregate.json", "w") as f_out:
    #     json.dump(summary, f_out, cls=json_serialize)

    # # plt.plot(aggregation_result["p_expo"][:, 1], aggregation_result["p_expo"][:, 0], "o-g",, label="P-Expo")
    # plt.plot(aggregation_result["qp"][:, 1], aggregation_result["qp"][:, 0], "o-b", label="QP")
    # plt.plot(aggregation_result["s_expo_1"][:, 1], aggregation_result["s_expo_1"][:, 0], "x--y", label="Sphere-Expo_1")
    # plt.plot(aggregation_result["s_expo_2"][:, 1], aggregation_result["s_expo_2"][:, 0], "x--m", label="Sphere-Expo_5")
    # plt.plot(aggregation_result["s_expo_3"][:, 1], aggregation_result["s_expo_3"][:, 0], "x--r", label="Sphere-Expo_7")
    # plt.ylabel("Normalized User utility")
    # plt.xlabel("Normalized Unfairness")
    # plt.legend(loc='lower right')
    # plt.show()


if __name__ == "__main__":
    experiment("D:\VinUni_researchAssistant\Hedron\data\MSLR\MSLR-WEB10K\Fold1\\vali.txt")