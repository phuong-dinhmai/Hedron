import numpy as np
import pandas as pd
import pickle
import json 

import matplotlib.pyplot as plt
import scipy.stats as st
import time
from tqdm import tqdm

from scipy.linalg import norm
import torch

from main import sphere_testing
from main.sphere_testing import sphere_path

from BPR.evaluate import get_relevance
from baseline import QP


# from decomposition import doubly_matrix, caratheodory_decomposition_pbm_gls
# from sampling_strategy import billiard_word


def movielen100k_testing():
    from BPR.model import BPR

    with open("../BPR/output/data.pkl", 'rb') as f:
        dataset = pickle.load(f)
        user_size, item_size = dataset['user_size'], dataset['item_size']
        train_user_list, test_user_list = dataset['train_user_list'], dataset['test_user_list']
        # train_pair = dataset['train_pair']

    model = BPR(user_size, item_size, dim=512, weight_decay=0.025)
    model.load_state_dict(torch.load("../BPR/output/bpr.pt"))
    model.eval()

    relevance_matrix, item_idx = get_relevance(50, model.W, model.H, train_user_list, batch=512)
    item_idx = item_idx.numpy()
    relevance_matrix = relevance_matrix.detach().numpy()
    data_anal = pd.read_csv("../BPR/output/data_analysis.csv", index_col=0)
    
    gamma = 1 / np.log(np.arange(0, 50) + 2)
    
    for i in range(relevance_matrix.shape[0]):
        idx = item_idx[i]
        print(relevance_matrix[i])
        group_masking = data_anal[["popular", "unpopular"]].loc[idx].to_numpy()
        group_size = group_masking.sum(axis=0)
        group_fairness = group_size / np.sum(group_size) * np.sum(gamma)
        optimal = projected_path(relevance_matrix[i], item_group_masking=group_masking, group_fairness=group_fairness, gamma=gamma)
        sphere_pareto = sphere_path(20*relevance_matrix[i], item_group_masking=group_masking, group_fairness=group_fairness, gamma=gamma, n_divided=3)
        base_qp = QP.experiment(20 * relevance_matrix[i], group_masking)
        draw(base_qp, sphere_pareto[0])
        # draw(base_qp[-5:], optimal[0])
        break


def experiment_QP():
    query_file_path = "/home/phuong/Documents/expohedron/data/TREC_2020/TREC Fair Ranking 2020 Data/clean_queries.json"
    query_file = open(query_file_path, "r", encoding="utf8")
    queries = [json.loads(line) for line in query_file.readlines()]
    queries = {query['qid']: query for query in queries} # The data of unique queries

    group_dict = pd.read_csv("data/TREC_2020/TREC Fair Ranking 2020 Data/new_group_2.csv", index_col=0)

    n_sample = 40
    alpha_arr = np.arange(0, n_sample+1) / n_sample

    doc_id_dict = {}
    prob_matrices = {}
    running_time = []
    for qid in queries.keys():
        query = queries[qid]
        docs = query["documents"]
        n_doc = len(docs)
        rel = []
        
        group_matrix = np.zeros((n_doc, 2))
        for ord in range(n_doc): 
            doc_id_dict[qid] = {docs[ord]["doc_id"]: ord}
            rel.append(docs[ord]["relevance"])
            group_matrix[ord, group_dict.loc[docs[ord]["doc_id"]]] = 1

        gamma = 1 / np.log(np.arange(0, n_doc) + 2)
        group_size = group_matrix.sum(axis=0)
        group_fairness = group_size / np.sum(group_size) * np.sum(gamma)
        start = time.time()
        _, pareto_opt = QP.experiment(np.asarray(rel), group_matrix, group_fairness, alpha_arr)
        end = time.time()
        running_time.append(end-start)
        prob_matrices[qid] = pareto_opt
    
    objs = {
        "alpha_arr": alpha_arr,
        "group_masking": group_dict,
        "prob_matrices": prob_matrices,
        "total_running_time": running_time
    }
    with open("../results/QP.pkl", "wb") as f:
        pickle.dump(objs, f)


def experment_methods():
    query_file_path = "/home/phuong/Documents/expohedron/data/TREC_2020/TREC Fair Ranking 2020 Data/clean_queries.json"
    query_file = open(query_file_path, "r", encoding="utf8")
    queries = [json.loads(line) for line in query_file.readlines()]
    queries = {query['qid']: query for query in queries}  # The data of unique queries

    group_dict = pd.read_csv("data/TREC_2020/TREC Fair Ranking 2020 Data/new_group_2.csv", index_col=0)

    doc_id_dict = {}
    exposure_score_projected = {}
    exposure_score_sphere = {}
    project_running_time = []
    sphere_running_time = []
    for qid in queries.keys():
        query = queries[qid]
        docs = query["documents"]
        n_doc = len(docs)
        rel = []

        group_matrix = np.zeros((n_doc, 2))
        for ord in range(n_doc):
            doc_id_dict[qid] = {docs[ord]["doc_id"]: ord}
            rel.append(docs[ord]["relevance"])
            group_matrix[ord, group_dict.loc[docs[ord]["doc_id"]]] = 1

        gamma = 1 / np.log(np.arange(0, n_doc) + 2)
        group_size = group_matrix.sum(axis=0)
        group_fairness = group_size / np.sum(group_size) * np.sum(gamma)
        start = time.time()
        projected_pareto_opt, _ = projected_path(np.asarray(rel), group_matrix, group_fairness)
        end = time.time()
        project_running_time.append(end - start)
        exposure_score_projected[qid] = projected_pareto_opt

        start = time.time()
        sphere_approx, _ = sphere_path(np.asarray(rel), group_matrix, group_fairness)
        end = time.time()
        sphere_running_time.append(end - start)
        exposure_score_sphere[qid] = sphere_approx

    objs = {
        "group_masking": group_dict,
        "exposure_score": exposure_score_projected,
        "total_running_time": project_running_time
    }
    with open("results/projected.pkl", "wb") as f:
        pickle.dump(objs, f)

    objs = {
        "group_masking": group_dict,
        "exposure_score": exposure_score_sphere,
        "total_running_time": sphere_running_time
    }
    with open("results/sphere.pkl", "wb") as f:
        pickle.dump(objs, f)


def eval_QP():
    # query_file_path = "/home/phuong/Documents/expohedron/data/TREC_2020/TREC Fair Ranking 2020 Data/clean_queries.json"
    # query_file = open(query_file_path, "r", encoding="utf8")
    # queries = [json.loads(line) for line in query_file.readlines()]
    # queries = {query['qid']: query for query in queries} # The data of unique queries
    #
    # group_dict = pd.read_csv("data/TREC_2020/TREC Fair Ranking 2020 Data/new_group_2.csv", index_col=0)

    with open("/results/QP.pkl", "rb") as file:
        objs = pickle.load(file)
    
    expected_matrices = objs["prob_matrices"]
    alpha_arr = objs["alpha_arr"]
    
    time_horizon = 10
    for qid in expected_matrices.keys():
    # for qid in queries.keys():
    #     query = queries[qid]
    #     docs = query["documents"]
    #     n_doc = len(docs)
        rel = []
        
        # group_matrix = np.zeros((n_doc, 2))
        # for ord in range(n_doc):
        #     rel.append(docs[ord]["relevance"])
        #     group_matrix[ord, group_dict.loc[docs[ord]["doc_id"]]] = 1
        # gamma = 1 / np.log(np.arange(0, n_doc) + 2)
        # group_size = group_matrix.sum(axis=0)
        # group_fairness = group_size / np.sum(group_size) * np.sum(gamma)

        # for i in range(6, len(alpha_arr)):
        #     matrix = np.asarray(expected_matrices[qid][i])
        #     if matrix is None:
        #         continue
        #     coefs, permutations = doubly_matrix(matrix.round(4))
        #     err = 1 - sum(coefs)
        #     coefs = [ele + err/len(coefs) for ele in coefs]
        #     sampler = billiard_word(coefs)
        #     for t in np.arange(1, 1000):  # Try to achive balance at this time
        #         index = next(sampler)
        #     for t in np.arange(1, time_horizon):
        #         index = next(sampler)
        #         rank = permutations[index]
        #         # print(QP.evaluate_probabilty(rank, np.asarray(rel), group_matrix, group_fairness))
        break


def eval_expohedron(file_eval, target_exposure):
    query_file_path = "/home/phuong/Documents/expohedron/data/TREC_2020/TREC Fair Ranking 2020 Data/clean_queries.json"
    query_file = open(query_file_path, "r", encoding="utf8")
    queries = [json.loads(line) for line in query_file.readlines()]
    queries = [query['qid'] for query in queries]

    with open(file_eval, "rb") as file:
        objs = pickle.load(file)

    epxosure_scores = objs["exposure_score"]

    check = []
    for qid in queries:
        segments = epxosure_scores[qid]
        d_matrices = target_exposure[qid]
        n_doc = d_matrices.shape[1]
        gamma = 1 / np.log(np.arange(0, n_doc) + 2)
        for _id in range(len(d_matrices)):
            matrix = d_matrices[_id]
            p_point = matrix @ gamma
            flag = False
            for i in range(len(segments)-1):
                vec_1 = p_point - segments[i]
                vec_2 = segments[i+1] - p_point
                if norm(vec_1) < 1e-9 or norm(vec_2) < 1e-9:
                    flag = True
                    break
                scalar = vec_2[0] / vec_1[0]
                if scalar < 0 and norm(vec_2 - scalar*vec_1) < 1e-9:
                    flag = True
                    break
            if not flag:
                check.append((qid, _id))

    np.savetxt("results/error_points.txt", np.asarray(check))


def running_time_eval():
    avg = {
        "pexpo_avg": [],
        "pexpo_total": [],
        "sexpo_avg": [],
        "sexpo_total": [],
        "qp_avg": []
    }
    for n_doc in tqdm(range(40, 61, 5)):
        ptime = {"avg": [], "total": []}
        stime = {"avg": [], "total": []}
        qtime = []
        for repeated in tqdm(range(0, 50)):
            # print("Load data")
            _relevance_score, item_group, _group_fairness, _gamma = load_data(n_doc, 2)
            # print("Start hedron experiment:")
            phedron_start = time.time()
            pobjs, ppoints = projected_path(_relevance_score, item_group, _group_fairness, _gamma)
            phedron_end = time.time()
            ptime["total"].append(phedron_end - phedron_start)
            ptime["avg"].append((phedron_end - phedron_start) / len(pobjs))

            shedron_start = time.time()
            sobjs, spoints = sphere_path(_relevance_score, item_group, _group_fairness, _gamma, n_divided=3, n_sample=6)
            shedron_end = time.time()
            stime["total"].append(shedron_end - shedron_start)
            stime["avg"].append((shedron_end - shedron_start) / len(sobjs))

            # print("Start QP experiment:")
            qp_start = time.time()
            base_qp = QP.experiment(_relevance_score, item_group, _group_fairness, alpha_arr=[0.1, 0.3, 0.7, 1])
            qp_end = time.time()
            qtime.append((qp_end - qp_start) / 4)

        x = st.norm.interval(confidence=0.90, loc=np.mean(ptime["total"]), scale=st.sem(ptime["total"]))
        avg["pexpo_total"].append([np.mean(ptime["total"]), x])
        x = st.norm.interval(confidence=0.90, loc=np.mean(ptime["avg"]), scale=st.sem(ptime["avg"]))
        avg["pexpo_avg"].append([np.mean(ptime["avg"]), x])

        x = st.norm.interval(confidence=0.90, loc=np.mean(stime["total"]), scale=st.sem(stime["total"]))
        avg["sexpo_total"].append([np.mean(stime["total"]), x])
        x = st.norm.interval(confidence=0.90, loc=np.mean(stime["avg"]), scale=st.sem(stime["avg"]))
        avg["sexpo_avg"].append([np.mean(stime["avg"]), x])

        x = st.norm.interval(confidence=0.90, loc=np.mean(qtime), scale=st.sem(qtime))
        avg["qp_avg"].append([np.mean(qtime), x])

    with open("../results/time_40_60.json", "w") as f_out:
        json.dump(avg, f_out)


def accuracy():
    n_doc = 40
    _relevance_score, item_group, _group_fairness, _gamma = load_data(n_doc, 2)
    print(_relevance_score.shape)

    # pobjs, ppoints = projected_path(_relevance_score, item_group, _group_fairness, _gamma)
    sobjs_0, spoints_0 = sphere_path(_relevance_score, item_group, _group_fairness, _gamma, n_divided=0, n_sample=27)
    sobjs_1, spoints_1 = sphere_path(_relevance_score, item_group, _group_fairness, _gamma, n_divided=1, n_sample=13)
    sobjs_2, spoints_2 = sphere_path(_relevance_score, item_group, _group_fairness, _gamma, n_divided=2, n_sample=6)
    # sobjs_3, spoints_3 = sphere_path(_relevance_score, item_group, _group_fairness, _gamma, n_divided=3, n_sample=6)

    from main.helpers import Objective
    objs = Objective(_relevance_score, _group_fairness, item_group, _gamma)
    expohedron = np.loadtxt("D:\VinUni_researchAssistant\expohedron\examples\points.csv", delimiter=",").astype(np.double)
    expohedron_objs = np.array([objs.objectives(point) for point in expohedron])

    n_sample = 40
    alpha_arr = np.arange(0, n_sample + 1) / n_sample
    qpobjs, qpoints = QP.experiment(_relevance_score, item_group, _group_fairness, alpha_arr)

    # pobjs = np.asarray(pobjs)
    sobjs_0 = np.asarray(sobjs_0)
    sobjs_1 = np.asarray(sobjs_1)
    sobjs_2 = np.asarray(sobjs_2)
    # sobjs_3 = np.asarray(sobjs_3)
    qpobjs = np.asarray(qpobjs)

    # pobjs = pobjs[pobjs[:, 1].argsort()]
    sobjs_0 = sobjs_0[sobjs_0[:, 1].argsort()]
    sobjs_1 = sobjs_1[sobjs_1[:, 1].argsort()]
    sobjs_2 = sobjs_2[sobjs_2[:, 1].argsort()]
    # sobjs_3 = sobjs_3[sobjs_3[:, 1].argsort()]
    qpobjs = qpobjs[qpobjs[:, 1].argsort()]
    qpobjs = qpobjs[5:, :]

    # plt.plot(pobjs[:, 1], pobjs[:, 0] / pobjs[-1, 0], marker='o', label="PExpo")
    plt.plot(sobjs_0[:, 1], sobjs_0[:, 0] / sobjs_0[-1, 0], marker='o', label="SphereExpo")
    plt.plot(sobjs_1[:, 1], sobjs_1[:, 0] / sobjs_1[-1, 0], marker='o', label="SphereExpo_1")
    plt.plot(sobjs_2[:, 1], sobjs_2[:, 0] / sobjs_2[-1, 0], marker='o', label="SphereExpo_2")
    # plt.plot(sobjs_3[:, 1], sobjs_3[:, 0] / sobjs_3[-1, 0], marker='o', label="SExpo_marks_3")
    plt.plot(qpobjs[:, 1], qpobjs[:, 0] / qpobjs[-1, 0], marker='o', label="QP")
    plt.plot(expohedron_objs[:, 1], expohedron_objs[:, 0] / expohedron_objs[-1, 0], marker='o', label="ExpoIndividual")

    plt.ylabel("User utility")
    plt.xlabel("Unfairness")
    plt.legend(loc='lower right')
    plt.show()


def plot():
    # with open("D:\\VinUni_researchAssistant\\Hedron\\results\\time_0_50.json", "r") as f_in:
    #     rtime_50 = json.load(f_in)
    # with open("D:\\VinUni_researchAssistant\\Hedron\\results\\time_50_100.json", "r") as f_in:
    #     rtime_50_100 = json.load(f_in)
    # with open("D:\\VinUni_researchAssistant\\Hedron\\results\\running_LP.json", "r") as f_in:
    #     rtime_LP = json.load(f_in)
    with open("/results/stime_base_2.json", "r") as f_in:
        sexpo = json.load(f_in)

    # avg = {
    #     "pexpo_avg": np.asarray(rtime_50["pexpo_avg"] + rtime_50_100["pexpo_avg"]),
    #     "LP": rtime_LP,
    #     "sexpo_avg": np.asarray(rtime_50["sexpo_avg"] + rtime_50_100["sexpo_avg"]),
    #     "qp_avg": np.asarray(rtime_50["qp_avg"] + rtime_50_100["qp_avg"])
    # }
    # bound = np.zeros((20, 2))
    # for i in range(0, 20):
    #     bound[i][0] = avg["pexpo_avg"][i, 0] - avg["pexpo_avg"][i, 1][0]
    #     bound[i][1] = avg["pexpo_avg"][i, 1][1] - avg["pexpo_avg"][i, 0]
    # plt.plot(list(range(5, 101, 5)), avg["pexpo_avg"][:, 0], label="PExpo")
    # plt.fill_between(list(range(5, 101, 5)), bound[:, 0], bound[:, 1])
    #
    # bound = np.zeros((20, 2))
    # for i in range(0, 20):
    #     bound[i][0] = avg["sexpo_avg"][i, 0] - avg["sexpo_avg"][i, 1][0]
    #     bound[i][1] = avg["sexpo_avg"][i, 1][1] - avg["sexpo_avg"][i, 0]
    # plt.plot(list(range(5, 101, 5)), avg["sexpo_avg"][:, 0], label="SphereExpo")
    # plt.fill_between(list(range(5, 101, 5)), bound[:, 0], bound[:, 1])
    #
    # bound = np.zeros((20, 2))
    # for i in range(0, 20):
    #     bound[i][0] = avg["qp_avg"][i, 0] - avg["qp_avg"][i, 1][0]
    #     bound[i][1] = avg["qp_avg"][i, 1][1] - avg["qp_avg"][i, 0]
    # plt.plot(list(range(5, 101, 5)), avg["qp_avg"][:, 0], label="QP")
    # plt.fill_between(list(range(5, 101, 5)), bound[:, 0], bound[:, 1])
    #
    # bound = np.zeros((20, 3))
    # # print(avg["LP"][0])
    # for i in range(0, 20):
    #     bound[i][0] = avg["LP"][i][0] - avg["LP"][i][1][0]
    #     bound[i][1] = avg["LP"][i][1][1] - avg["LP"][i][0]
    #     bound[i][2] = avg["LP"][i][0]
    # plt.plot(list(range(5, 101, 5)), bound[:, 2], label="LP")
    # plt.fill_between(list(range(5, 101, 5)), bound[:, 0], bound[:, 1])

    total = {
        # "pexpo_total": np.asarray(rtime_50["pexpo_total"] + rtime_50_100["pexpo_total"]),
        # "sexpo_3_total": np.asarray(rtime_50["sexpo_total"] + rtime_50_100["sexpo_total"]),
        "sexpo_0_total": sexpo["sexpo_total"],
        "sexpo_1_total": sexpo["sexpo_1_total"],
        "sexpo_2_total": sexpo["sexpo_2_total"],
        "sexpo_3_total": sexpo["sexpo_3_total"]
    }

    # bound = np.zeros((11, 3))
    # for i in range(0, 11):
    #     bound[i][0] = total["pexpo_total"][i, 0] - total["pexpo_total"][i, 1][0]
    #     bound[i][1] = total["pexpo_total"][i, 1][1] - total["pexpo_total"][i, 0]
    #     bound[i][2] = total["pexpo_total"][i, 0]
    # plt.plot(list(range(5, 60, 5)), bound[:, 2], label="PExpo")
    # plt.fill_between(list(range(5, 60, 5)), bound[:, 0], bound[:, 1])

    bound = np.zeros((8, 3))
    for i in range(0, 7):
        # bound[i][0] = total["sexpo_0_total"][i][0] - total["sexpo_0_total"][i][1][0]
        # bound[i][1] = total["sexpo_0_total"][i][1][1] - total["sexpo_0_total"][i][0]
        bound[i][2] = total["sexpo_0_total"][i][0]
    bound[7][2] = total["sexpo_0_total"][6][0] * 8
    plt.plot(list(range(3, 11)), bound[:, 2], label="SphereExpo")
    # plt.fill_between(list(range(5, 60, 5)), bound[:, 0], bound[:, 1])

    bound = np.zeros((8, 3))
    for i in range(0, 7):
        # bound[i][0] = total["sexpo_1_total"][i][0] - total["sexpo_1_total"][i][1][0]
        # bound[i][1] = total["sexpo_1_total"][i][1][1] - total["sexpo_1_total"][i][0]
        bound[i][2] = total["sexpo_1_total"][i][0]
    bound[7][2] = total["sexpo_1_total"][6][0] * 8
    plt.plot(list(range(3, 11)), bound[:, 2], label="SphereExpo_1_marker")
    # plt.fill_between(list(range(5, 60, 5)), bound[:, 0], bound[:, 1])

    bound = np.zeros((8, 3))
    for i in range(0, 7):
        # bound[i][0] = total["sexpo_2_total"][i][0] - total["sexpo_2_total"][i][1][0]
        # bound[i][1] = total["sexpo_2_total"][i][1][1] - total["sexpo_2_total"][i][0]
        bound[i][2] = total["sexpo_2_total"][i][0]
    bound[7][2] = total["sexpo_2_total"][6][0] * 8
    plt.plot(list(range(3, 11)), bound[:, 2], label="SphereExpo_2_marker")
    # plt.fill_between(list(range(5, 60, 5)), bound[:, 0], bound[:, 1])

    bound = np.zeros((8, 3))
    for i in range(0, 7):
        # bound[i][0] = total["sexpo_3_total"][i, 0] - total["sexpo_3_total"][i, 1][0]
        # bound[i][1] = total["sexpo_3_total"][i, 1][1] - total["sexpo_3_total"][i, 0]
        bound[i][2] = total["sexpo_3_total"][i][0]
    bound[7][2] = total["sexpo_3_total"][6][0] * 8
    plt.plot(list(range(3, 11)), bound[:, 2], label="SphereExpo_3_marker")
    # plt.fill_between(list(range(5, 60, 5)), bound[:, 0], bound[:, 1])

    plt.ylabel("Running time (s)")
    plt.xlabel("Number of items (in log2)")
    plt.legend(loc='upper left')
    plt.show()


if __name__ == "__main__":
    # plot()
    # accuracy()
    # exit()
    # running_time_eval()
    # accuracy()
    # exit()
    n_doc = 10
    n_group = 10
    np.random.seed(100)
    rel = np.random.rand(n_doc)

    item_group_masking = np.zeros((n_doc, n_group))
    for i in range(n_doc):
        j = np.random.randint(n_group, size=1)
        # j = np.random.choice(range(n_group), size=1, p=(0.3, 0.7))
        item_group_masking[i][j[0]] = 1
    cnt_col = item_group_masking.sum(axis=0)
    item_group_masking = np.delete(item_group_masking, cnt_col == 0, 1)
    np.savetxt("data_error/item_group.csv", item_group_masking, delimiter=",")

    gamma = 1 / np.log(np.arange(0, n_doc) + 2)
    group_size = item_group_masking.sum(axis=0)
    print(group_size)
    group_fairness = group_size / np.sum(group_size) * np.sum(gamma)

    # obj, points = projected_path(rel, item_group_masking, group_fairness, gamma)
    obj, points = sphere_testing.sphere_path(rel, item_group_masking, group_fairness, gamma, 2)

    qp_objs, qp_points = QP.experiment(rel, item_group_masking, group_fairness, np.arange(1, 21) / 20)
    draw(obj, qp_objs)

    # movielen100k_testing()

    # experiment_QP()
    # eval_QP()

