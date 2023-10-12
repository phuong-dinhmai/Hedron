import time
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

from helpers import invert_permutation
from sphere_testing import projected_path, sphere_path

from BPR.evaluate import get_relevance
import QP

from decomposition import doubly_matrix, caratheodory_decomposition_pbm_gls
from sampling_strategy import billiard_word


def movielen100k_testing():
    from BPR.model import BPR

    with open("BPR/output/data.pkl", 'rb') as f:
        dataset = pickle.load(f)
        user_size, item_size = dataset['user_size'], dataset['item_size']
        train_user_list, test_user_list = dataset['train_user_list'], dataset['test_user_list']
        # train_pair = dataset['train_pair']

    model = BPR(user_size, item_size, dim=512, weight_decay=0.025)
    model.load_state_dict(torch.load("BPR/output/bpr.pt"))
    model.eval()

    relevance_matrix, item_idx = get_relevance(50, model.W, model.H, train_user_list, batch=512)
    item_idx = item_idx.numpy()
    relevance_matrix = relevance_matrix.detach().numpy()
    data_anal = pd.read_csv("BPR/output/data_analysis.csv", index_col=0)
    
    gamma = 1 / np.log(np.arange(0, 50) + 2)
    
    for i in range(relevance_matrix.shape[0]):
        idx = item_idx[i]
        print(relevance_matrix[i])
        group_masking = data_anal[["popular", "unpopular"]].loc[idx].to_numpy()
        group_size = group_masking.sum(axis=0)
        group_fairness = group_size / np.sum(group_size) * np.sum(gamma)
        optimal = projected_path(relevance_matrix[i], item_group_masking=group_masking, group_fairness=group_fairness, gamma=gamma)
        sphere_pareto = sphere_path(20*relevance_matrix[i], item_group_masking=group_masking, group_fairness=group_fairness, gamma=gamma, n_divided=3)
        base_qp = QP.experiment(20*relevance_matrix[i], group_masking)
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
        pareto_opt, _ = QP.experiment(np.asarray(rel), group_matrix, group_fairness, alpha_arr)
        end = time.time()
        running_time.append(end-start)
        prob_matrices[qid] = pareto_opt
    
    objs = {
        "alpha_arr": alpha_arr,
        "group_masking": group_dict,
        "prob_matrices": prob_matrices,
        "total_running_time": running_time
    }
    with open("results/QP.pkl", "wb") as f:
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

    with open("D:\\VinUni_researchAssistant\\Hedron\\results\\QP.pkl", "rb") as file:
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

        for i in range(6, len(alpha_arr)):
            matrix = np.asarray(expected_matrices[qid][i])
            if matrix is None:
                continue
            coefs, permutations = doubly_matrix(matrix.round(4))
            err = 1 - sum(coefs)
            coefs = [ele + err/len(coefs) for ele in coefs]
            sampler = billiard_word(coefs)
            for t in np.arange(1, 1000):  # Try to achive balance at this time
                index = next(sampler)
            for t in np.arange(1, time_horizon):
                index = next(sampler)
                rank = permutations[index]
                # print(QP.evaluate_probabilty(rank, np.asarray(rel), group_matrix, group_fairness))
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


def draw(a, b):
    a = np.asarray(a)
    b = np.asarray(b)
    b = b[b[:, 1].argsort()]
    a = a[a[:, 1].argsort()]

    plt.plot(b[:, 1], b[:, 0], label="check")
    plt.plot(a[:, 1], a[:, 0], label="optimal")
    plt.ylabel("User utility")
    plt.xlabel("Unfairness")
    plt.legend(loc='lower right')
    plt.show()


def load_data(n_doc, n_group):
    # n_doc = 4
    # n_group = 2
    # relevance_score = np.asarray([0.7, 0.8, 1, 0.4])
    # item_group_masking = np.asarray([[0, 1], [0, 1], [1, 0], [1, 0]])
    # gamma = np.asarray([4, 3, 2, 1])
    # group_fairness = np.asarray([6.5, 3.5])

    # relevance_score = np.loadtxt("data_error/relevance_score.csv", delimiter=",").astype(np.double)
    # item_group_masking = np.loadtxt("data_error/item_group.csv", delimiter=",").astype(np.double)
    # n_doc = item_group_masking.shape[0]

    np.random.seed(n_doc)
    # relevance_score = np.arange(1, n_doc+1) / n_doc
    # print(relevance_score)
    relevance_score = np.random.rand(n_doc)
    # np.savetxt("data_error/relevance_score.csv", relevance_score, delimiter=",")

    item_group_masking = np.zeros((n_doc, n_group))
    # item_group_masking[:10, 0] = 1
    # item_group_masking[10:, 1] = 1
    # x = np.arange(-n_group/2, n_group/2)
    # xU, xL = x + 0.5, x - 0.5
    # prob = ss.norm.cdf(xU, scale=3) - ss.norm.cdf(xL, scale=3)
    # prob = prob / prob.sum()
    for i in range(n_doc):
        j = np.random.randint(n_group, size=1)
        # j = np.random.choice(range(n_group), size=1, p=prob)
        item_group_masking[i][j[0]] = 1
    cnt_col = item_group_masking.sum(axis=0)
    item_group_masking = np.delete(item_group_masking, cnt_col == 0, 1)
    # np.savetxt("data_error/item_group.csv", item_group_masking, delimiter=",")

    gamma = 1 / np.log(np.arange(0, n_doc) + 2)
    group_size = item_group_masking.sum(axis=0)
    group_fairness = group_size / np.sum(group_size) * np.sum(gamma)
    # print(group_fairness)

    return relevance_score, item_group_masking, group_fairness, gamma


if __name__ == "__main__":
    avg = {
        "pexpo_avg": [],
        "pexpo_total": [],
        "sexpo_avg": [],
        "sexpo_total": [],
        "qp_avg": []
    }
    for n_doc in tqdm(range(4, 51, 5)):
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
            ptime["total"].append(phedron_end-phedron_start)
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
            qtime.append((qp_end-qp_start)/4)

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

    with open("results/time_50.json", "w") as f_out:
        json.dump(avg, f_out)

    # draw(base_qp[4:], objs)

    # movielen100k_testing()

    # experiment_QP()
    # eval_QP()

