import numpy as np
import pandas as pd
import pickle
import json
from os import listdir
from os.path import isfile, join

import matplotlib.pyplot as plt
import brokenaxes as box
import scipy.stats as st


def synthetic_ploting(file_name):
    aggregation_result = None
    aggregation_running_time = None
    with open(file_name, "r") as log_file:
        summary = json.load(log_file)
        aggregation_result = summary["accuracy"]
        aggregation_running_time = summary["running_time"]

    for key in aggregation_result.keys():
        aggregation_result[key] = np.array(aggregation_result[key])
        
    fig, ax = plt.subplots(nrows=1, ncols=2)
    fig.subplots_adjust(wspace=0.35)

    ax[0].plot(aggregation_result["qp"][:, 0], aggregation_result["qp"][:, 1], "o-b", label="QP")
    # ax[0].plot(aggregation_result["p_expo"][:, 0], aggregation_result["p_expo"][:, 1], "o-g", label="P-Expo")
    ax[0].plot(aggregation_result["s_expo_1"][:, 0], aggregation_result["s_expo_1"][:, 1], "x--y", label="Sphere-Expo (K=1)")
    ax[0].plot(aggregation_result["s_expo_2"][:, 0], aggregation_result["s_expo_2"][:, 1], "x--r", label="Sphere-Expo (K=3)")
    ax[0].plot(aggregation_result["s_expo_3"][:, 0], aggregation_result["s_expo_3"][:, 1], "x--m", label="Sphere-Expo (K=7)")
    ax[0].set_xlabel("Normalized User utility")
    ax[0].set_ylabel("Normalized Unfairness (nDCG)")
    ax[0].legend(loc='upper left')
    ax[0].grid()
    ax[0].set_title('Aggregated Pareto front')
    
    num_doc_list = 5 * np.arange(2, 21)
    # num_doc_list = np.arange(8, 21)
    ax[1].plot(num_doc_list, aggregation_running_time["qp"], "o-b", label="QP")
    # ax[1].plot(num_doc_list, aggregation_running_time["p_expo"], "o-g", label="P-Expo")
    ax[1].plot(num_doc_list, aggregation_running_time["s_expo_1"], "x--y", label="Sphere-Expo (K=1)")
    ax[1].plot(num_doc_list, aggregation_running_time["s_expo_2"], "x--r", label="Sphere-Expo (K=3)")
    ax[1].plot(num_doc_list, aggregation_running_time["s_expo_3"], "x--m", label="Sphere-Expo (K=7)")
    ax[1].set_ylabel("Avg running time (s)")
    ax[1].set_xlabel("Number of items")
    ax[1].grid()
    ax[1].legend(loc='upper left')
    ax[1].set_title('Aggregated running time')
    
    # fig.suptitle("Aggregation results in a small synhetic dataset")
    plt.show()

    # # plt.plot(aggregation_result["p_expo"][:, 1], aggregation_result["p_expo"][:, 0], "o-g",, label="P-Expo")
    # plt.plot(aggregation_result["qp"][:, 1], aggregation_result["qp"][:, 0], "o-b", label="QP")
    # plt.plot(aggregation_result["s_expo_1"][:, 1], aggregation_result["s_expo_1"][:, 0], "x--y", label="Sphere-Expo_1")
    # # plt.plot(aggregation_result["s_expo_2"][:, 1], aggregation_result["s_expo_2"][:, 0], "x--m", label="Sphere-Expo_5")
    # plt.plot(aggregation_result["s_expo_3"][:, 1], aggregation_result["s_expo_3"][:, 0], "x--r", label="Sphere-Expo_7")
    # plt.ylabel("Normalized User utility")
    # plt.xlabel("Normalized Unfairness")
    # plt.legend(loc='lower right')
    # plt.show()


def real_world_plotting(TREC_folder_path, TREC_file_path, MSLR_folder_path, MSLR_file_path):
    mrfr_aggregate = {"trec": [], "mslr": []}

    aggregation_result = None
    normalize_term = None
    with open(TREC_file_path, "r") as log_file:
        summary = json.load(log_file)
        aggregation_result = summary["accuracy"]
        normalize_term = summary["norm"]

    for key in aggregation_result.keys():
        aggregation_result[key] = np.array(aggregation_result[key])

    file_paths = [(f, join(TREC_folder_path, f)) for f in listdir(TREC_folder_path) if isfile(join(TREC_folder_path, f))]
    file_paths.sort(key=lambda x: x[1])

    cnt = 0
    for file_name, file_path in file_paths:
        with open(file_path, "r") as log_file:
            if ("b1" not in file_name) or ("n10" not in file_name):
                continue
            results = json.load(log_file)
            utils = results["utility"]
            unfairness = results["unfairness"]
            trec_mslr = []
            for sequence_id in utils.keys():
                query_id = list(utils[sequence_id].keys())[0]
                util = utils[sequence_id][query_id]
                unfair = unfairness[sequence_id][query_id]
                if query_id not in normalize_term.keys():
                    continue
                norm = normalize_term[query_id]
                trec_mslr.append((util / norm[0], unfair / norm[1]))
            mrfr_aggregate["trec"].append(np.mean(np.array(trec_mslr), axis=0))
    mrfr_aggregate["trec"] = np.array(mrfr_aggregate["trec"])
    fig, ax = plt.subplots(nrows=2, ncols=2)
    fig.subplots_adjust(hspace=0.05)  # adjust space between axes

    ax[0][0].plot(mrfr_aggregate["trec"][:, 0], mrfr_aggregate["trec"][:, 1], "o-g", label="Ctrl")

    ax[1][0].plot(aggregation_result["qp"][:, 0], aggregation_result["qp"][:, 1], "o-b", label="QP")
    ax[1][0].plot(aggregation_result["s_expo_1"][:, 0], aggregation_result["s_expo_1"][:, 1], "x--y", label="Sphere-Expo_1")
    ax[1][0].plot(aggregation_result["s_expo_2"][:, 0], aggregation_result["s_expo_2"][:, 1], "x--m", label="Sphere-Expo_5")
    ax[1][0].plot(aggregation_result["s_expo_3"][:, 0], aggregation_result["s_expo_3"][:, 1], "x--r", label="Sphere-Expo_7")

    ax[0][0].spines.bottom.set_visible(False)
    ax[1][0].spines.top.set_visible(False)
    ax[0][0].xaxis.tick_top()
    ax[0][0].tick_params(labeltop=False)  # don't put tick labels at the top
    ax[1][0].xaxis.tick_bottom()

    ax[1][0].set_xlabel("Normalized User utility")
    ax[1][0].set_ylabel("Normalized Unfairness (nDCG)")
    ax[1][0].yaxis.set_label_coords(0.05, 0.5, transform=fig.transFigure)

    ax[1][0].legend(loc='upper left')
    ax[1][0].grid()
    ax[0][0].grid()
    ax[1][0].set_xlim([0.95, 1])
    ax[0][0].set_xlim([0.95, 1])
    ax[0][0].set_title('TREC aggregated Pareto front')

    aggregation_result = None
    normalize_term = None
    with open(MSLR_file_path, "r") as log_file:
        summary = json.load(log_file)
        aggregation_result = summary["accuracy"]
        normalize_term = summary["norm"]

    for key in aggregation_result.keys():
        aggregation_result[key] = np.array(aggregation_result[key])

    file_paths = [(f, join(MSLR_folder_path, f)) for f in listdir(MSLR_folder_path) if isfile(join(MSLR_folder_path, f))]
    file_paths.sort(key=lambda x: x[1])

    for file_name, file_path in file_paths:
        # if "l0.2" not in file_name:
        #     continue
        with open(file_path, "r") as log_file:
            results = json.load(log_file)
            utils = results["utility"]
            unfairness = results["unfairness"]
            trec_mslr = []
            for sequence_id in utils.keys():
                query_id = list(utils[sequence_id].keys())[0]
                util = utils[sequence_id][query_id]
                unfair = unfairness[sequence_id][query_id]
                if query_id not in normalize_term:
                    continue
                norm = normalize_term[query_id]
                # print(norm)
                # print(util, " ", unfair)
                if (util / norm[0]) > 1:
                    print(query_id)
                trec_mslr.append((util / norm[0], unfair / norm[1]))
            mrfr_aggregate["mslr"].append(np.mean(np.array(trec_mslr), axis=0))
            print(file_name)
            print(mrfr_aggregate["mslr"][-1])
    mrfr_aggregate["mslr"] = np.array(mrfr_aggregate["mslr"])
    # print(mrfr_aggregate["mslr"])

    ax[0][1].spines.bottom.set_visible(False)
    ax[1][1].spines.top.set_visible(False)
    ax[0][1].xaxis.tick_top()
    ax[0][1].tick_params(labeltop=False)  # don't put tick labels at the top
    ax[1][1].xaxis.tick_bottom()

    ax[0][1].plot(mrfr_aggregate["mslr"][:, 0], mrfr_aggregate["mslr"][:, 1], "o-g", label="Ctrl")
    # ax[1].plot(aggregation_result["qp"][:, 0], aggregation_result["qp"][:, 1], "o-b", label="QP")
    ax[1][1].plot(aggregation_result["s_expo_3"][:, 0], aggregation_result["s_expo_3"][:, 1], "o-b", label="QP")
    ax[1][1].plot(aggregation_result["s_expo_1"][:, 0], aggregation_result["s_expo_1"][:, 1], "x--y", label="Sphere-Expo_1")
    ax[1][1].plot(aggregation_result["s_expo_2"][:, 0], aggregation_result["s_expo_2"][:, 1], "x--m", label="Sphere-Expo_5")
    ax[1][1].plot(aggregation_result["s_expo_3"][:, 0], aggregation_result["s_expo_3"][:, 1], "x--r", label="Sphere-Expo_7")
    # ax[1].set_xlabel("Normalized User utility")
    # ax[1].set_ylabel("Normalized Unfairness (nDCG)")
    # ax[1].legend(loc='upper left')
    # ax[1].grid()
    # ax[1].set_title('Aggregated Pareto front')

    plt.show()


def real_world_plotting_2(TREC_folder_path, TREC_file_path, MSLR_folder_path, MSLR_file_path):
    mrfr_aggregate = {"trec": [], "mslr": []}

    aggregation_result = None
    normalize_term = None
    with open(TREC_file_path, "r") as log_file:
        summary = json.load(log_file)
        aggregation_result = summary["accuracy"]
        normalize_term = summary["norm"]
        running_time = summary["running_time"]
    
    running_time = pd.DataFrame(running_time)
    # print(running_time.mean(axis=1))

    for key in aggregation_result.keys():
        aggregation_result[key] = np.array(aggregation_result[key])

    file_paths = [(f, join(TREC_folder_path, f)) for f in listdir(TREC_folder_path) if isfile(join(TREC_folder_path, f))]
    file_paths.sort(key=lambda x: x[1])

    cnt = 0
    for file_name, file_path in file_paths:
        with open(file_path, "r") as log_file:
            if ("b1" not in file_name) or ("n3" not in file_name):
                continue
            if "l0.8" in file_name:
                continue
            results = json.load(log_file)
            utils = results["utility"]
            unfairness = results["unfairness"]
            trec_mslr = []
            for sequence_id in utils.keys():
                query_id = list(utils[sequence_id].keys())[0]
                util = utils[sequence_id][query_id]
                unfair = unfairness[sequence_id][query_id]
                if query_id not in normalize_term.keys():
                    continue
                norm = normalize_term[query_id]
                trec_mslr.append((util / norm[0], unfair / norm[1]))
            mrfr_aggregate["trec"].append(np.mean(np.array(trec_mslr), axis=0))
            print(file_name)
            print(mrfr_aggregate["trec"][-1])
    mrfr_aggregate["trec"] = np.array(mrfr_aggregate["trec"])
    print("TREC")
    fig, ax = plt.subplots(nrows=1, ncols=2)

    ax[0].plot(aggregation_result["qp"][:, 0], aggregation_result["qp"][:, 1], "o-b", label="QP")
    ax[0].plot(aggregation_result["s_expo_1"][:, 0], aggregation_result["s_expo_1"][:, 1], "x--y", label="Sphere-Expo (K=1)")
    ax[0].plot(aggregation_result["s_expo_2"][:, 0], aggregation_result["s_expo_2"][:, 1], "x--r", label="Sphere-Expo (K=3)")
    ax[0].plot(mrfr_aggregate["trec"][:, 0], mrfr_aggregate["trec"][:, 1], "og", label="Ctrl")
    # ax[0].plot(aggregation_result["s_expo_3"][:, 0], aggregation_result["s_expo_3"][:, 1], "x--r", label="Sphere-Expo_7")
    ax[0].set_xlabel("Normalized User utility (nDCG)")
    ax[0].set_ylabel("Normalized Unfairness")
    ax[0].legend(loc='upper left')
    ax[0].grid()
    ax[0].set_title('TREC2020')

    aggregation_result = None
    normalize_term = None
    with open(MSLR_file_path, "r") as log_file:
        summary = json.load(log_file)
        aggregation_result = summary["accuracy"]
        normalize_term = summary["norm"]
        running_time = summary["running_time"]
    
    running_time = pd.DataFrame(running_time)
    # print(running_time.mean(axis=1))

    for key in aggregation_result.keys():
        aggregation_result[key] = np.array(aggregation_result[key])

    file_paths = [(f, join(MSLR_folder_path, f)) for f in listdir(MSLR_folder_path) if isfile(join(MSLR_folder_path, f))]
    file_paths.sort(key=lambda x: x[1])

    for file_name, file_path in file_paths:
        # if "l0.2" not in file_name:
        #     continue
        with open(file_path, "r") as log_file:
            results = json.load(log_file)
            utils = results["utility"]
            unfairness = results["unfairness"]
            trec_mslr = []
            for sequence_id in utils.keys():
                query_id = list(utils[sequence_id].keys())[0]
                util = utils[sequence_id][query_id]
                unfair = unfairness[sequence_id][query_id]
                if query_id not in normalize_term:
                    continue
                norm = normalize_term[query_id]
                if (norm[0] < 1e-6) or (norm[1] < 1e-6):
                    print(query_id)
                    continue
                # if (util / norm[0]) > 1:
                #     print(query_id)
                trec_mslr.append((util / norm[0], unfair / norm[1]))
            mrfr_aggregate["mslr"].append(np.mean(np.array(trec_mslr), axis=0))
            print(file_name)
            print(mrfr_aggregate["mslr"][-1])
    mrfr_aggregate["mslr"] = np.array(mrfr_aggregate["mslr"])
    # print(mrfr_aggregate["mslr"])

    ax[1].plot(aggregation_result["s_expo_3"][:, 0], aggregation_result["s_expo_3"][:, 1], "o-b", label="QP")
    ax[1].plot(aggregation_result["s_expo_1"][:, 0], aggregation_result["s_expo_1"][:, 1], "x--y", label="Sphere-Expo (K=1)")
    ax[1].plot(aggregation_result["s_expo_2"][:, 0], aggregation_result["s_expo_2"][:, 1], "x--r", label="Sphere-Expo (K=3)")
    ax[1].plot(mrfr_aggregate["mslr"][:, 0], mrfr_aggregate["mslr"][:, 1], "og", label="Ctrl")
    ax[1].set_xlabel("Normalized User utility (nDCG)")
    # ax[1].set_ylabel("Normalized Unfairness")
    ax[1].legend(loc='upper left')
    ax[1].grid()
    ax[1].set_title('MSLR')

    # plt.title("Aggregate Pareto fronts in real-world dataset", y=-2 )
    plt.show()


if __name__ == "__main__":
    # synthetic_ploting("results/synthetic/large.json")
    real_world_plotting_2("results/TREC/year_group/", "results/TREC/aggregate.json", "results/MSLR/133/", "results/MSLR/aggregate.json")




