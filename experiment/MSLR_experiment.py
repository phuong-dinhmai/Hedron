import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import pandas as pd
import numpy as np
from sklearn.metrics import ndcg_score
import matplotlib.pyplot as plt

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
    bins = [i for i in range(0, 260, 10)]
    labels = [i for i in range(0, 25)]
    train_df['binned'] = pd.cut(train_df[133], bins=bins, labels=labels)

    results = {
        "alpha": np.arange(1, 21) / 20,
        "qp": [[] for i in range(20)],
        "s_expo_1": [[] for i in range(20)],
        "s_expo_2": [[] for i in range(20)],
        "s_expo_3": [[] for i in range(20)],
    }
    cnt = 10

    for qid in x:
        items = train_df.loc[train_df[1] == qid, :]
        rel = items[0] / 5
        group = items["binned"]
        n_doc = len(rel)

        if len(rel.unique()) == 1 or len(group.unique()) == 1 or n_doc > 50 or n_doc < 5:
            continue

        group_masking = np.zeros((len(rel), len(labels)))
        for i in range(n_doc):
            group_masking[i][group.iloc[i]] = 1

        # gamma = 1 / np.log(np.arange(0, n_doc) + 2)
        gamma = 1 / (np.arange(0, n_doc) + 1)
        group_size = group_masking.sum(axis=0)
        item_group_masking = np.delete(group_masking, group_size == 0, 1)
        group_size = item_group_masking.sum(axis=0)
        group_fairness = group_size / np.sum(group_size) * np.sum(gamma)
        rel = rel.to_numpy()

        qp_objs, qp_points = QP.experiment(rel, item_group_masking, gamma, group_fairness, results["alpha"])

        if (np.array(qp_objs) == None).any():
            continue
        if np.linalg.norm(qp_objs[0][0] - qp_objs[-1][0]) < 1e-3: 
            print("skip utils")
            continue
        if np.linalg.norm(qp_objs[0][1] - qp_objs[-1][1]) < 1e-3: 
            print("skip unfair")
            continue
        print(qid)
        
        cnt -= 1
        spoints, s_1_points, s_2_points, s_3_points = s_short_cut_accuracy(rel, item_group_masking, group_fairness, gamma, 3, 2)
        
        optimal_util_point = s_3_points[-1]

        for i in range(0, len(results["alpha"])-1):
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
                continue
                optimal = optimal_util_point
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

    for i in range(len(results["alpha"])-1):
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
    experiment("D:\VinUni_researchAssistant\Hedron\data\MSLR\MSLR-WEB10K\Fold1\\vali.txt")