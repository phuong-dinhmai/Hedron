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
from scipy.optimize import minimize, Bounds

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
        # if int(qid) == 38093:
        #     print(group_matrix)
        #     raise Exception ("test")

        gamma = 1 / np.log(np.arange(0, n_doc) + 2)
        group_size = group_matrix.sum(axis=0)
        group_fairness = group_size / np.sum(group_size) * np.sum(gamma)
        start = time.time()
        pobjs, projected_pareto_opt = projected_path(np.asarray(rel), group_matrix, group_fairness, gamma)
        end = time.time()
        project_running_time.append(end - start)
        exposure_score_projected[qid] = projected_pareto_opt

        start = time.time()
        sobjs, sphere_approx = sphere_path(np.asarray(rel), group_matrix, group_fairness, gamma)
        end = time.time()
        sphere_running_time.append(end - start)
        exposure_score_sphere[qid] = sphere_approx
        if int(qid) == 26561:
            print(pobjs)
            print(sobjs)
            raise Exception('s')

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
    query_file_path = "/home/phuong/Documents/expohedron/data/TREC_2020/TREC Fair Ranking 2020 Data/clean_queries.json"
    query_file = open(query_file_path, "r", encoding="utf8")
    queries = [json.loads(line) for line in query_file.readlines()]
    queries = {query['qid']: query for query in queries} # The data of unique queries
    
    group_dict = pd.read_csv("data/TREC_2020/TREC Fair Ranking 2020 Data/new_group_2.csv", index_col=0)

    with open("results/QP.pkl", "rb") as file:
        objs = pickle.load(file)
    
    expected_matrices = objs["prob_matrices"]
    alpha_arr = objs["alpha_arr"]
    
    result = []
    cnt = 0
    # for qid in expected_matrices.keys():
    for qid in queries.keys():
        query = queries[qid]
        docs = query["documents"]
        n_doc = len(docs)
        rel = []
        
        group_matrix = np.zeros((n_doc, 2))
        for ord in range(n_doc):
            rel.append(docs[ord]["relevance"])
            group_matrix[ord, group_dict.loc[docs[ord]["doc_id"]]] = 1
        gamma = 1 / np.log(np.arange(0, n_doc) + 2)
        group_size = group_matrix.sum(axis=0)
        group_fairness = group_size / np.sum(group_size) * np.sum(gamma)
        end_point = gamma[invert_permutation(np.argsort(-np.array(rel)))]

        max_unfair = np.sum((group_matrix.T @ end_point - group_fairness) ** 2)
        if max_unfair < 1e-5:
            continue
        result.append([])
        dcg = np.sort(gamma) @ np.sort(rel)

        for i in range(6, len(expected_matrices[qid])):
            rank = expected_matrices[qid][i]
            if rank is None:
                result[cnt].append(None)
                continue
            x = QP.evaluate_probabilty(rank.round(4), np.asarray(rel), group_matrix, group_fairness)
            result[cnt].append([x[0] / dcg, x[1] / max_unfair])
        cnt += 1
    
    utils = np.zeros(len(alpha_arr) - 5)
    unfaire = np.zeros(len(alpha_arr) - 5)
    cnt = np.zeros(len(alpha_arr) - 5)

    for i in range(len(result)):
        if result is None:
            continue
        for j in range(len(result[i])):
            if result[i][j] is None:
                continue
            cnt[j] += 1
            utils[j] += result[i][j][0]
            unfaire[j] += result[i][j][1]
    print(utils / cnt)
    print(unfaire / cnt)
    print("---------------------")


def get_pareto_point_for_scalarization(pareto_curve: list, target_exposure: np.ndarray,
                                       pbm: np.ndarray, prp_exposure, alpha: float,
                                       fairness: str, relevance_vector: np.ndarray) -> tuple:
    """
        Given a Pareto front in an expohedron, and a scalarization parameter `alpha` computes the optimum of the scalarized problem

            min α (-U) + (1-α) F

    :param pareto_curve: The pareto curve in the expohedron
    :type pareto_curve: list
    :param target_exposure: The target exposure vector
    :type target_exposure: numpy.ndarray
    :param expohedron: The expohedron
    :type expohedron: DBNexpohedron
    :param alpha: The scalarization parameter
    :type alpha: float
    :return: The optimal utility, the optimal unfairness and the optimal exposure
    :rtype: Tuple[float, float, numpy.ndarray]
    """
    def nutility(exposure, pbm, relevance_vector) -> float:
        return exposure @ relevance_vector / (np.sort(relevance_vector) @ np.sort(pbm))

    def scalarized_objective_within_pareto_segment(alpha: float, scalarization: float,
                                               point1: np.ndarray, point2: np.ndarray,
                                               target_exposure: np.ndarray,
                                               pbm: np.ndarray,
                                               relevance_vector: np.ndarray,
                                               prp_exposure,
                                               fairness: str) -> float:
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
        :param pbm: The PBM
        :type pbm: numpy.ndarray
        :return: The value of the objective function
        :rtype: float
        """
        exposure = alpha * point1 + (1-alpha) * point2
        return scalarization * (-nutility(exposure, pbm, relevance_vector)) + (1-scalarization) * (np.linalg.norm(exposure - target_exposure) / np.linalg.norm(prp_exposure - target_exposure)) ** 2  # expohedron.nunfairness(exposure, fairness) ** 2


    objective = lambda exposure: alpha * (-nutility(exposure, pbm, relevance_vector)) + \
                                 (1-alpha) * (np.linalg.norm(exposure - target_exposure) / np.linalg.norm(prp_exposure - target_exposure)) ** 2  # expohedron.nunfairness(exposure, fairness) ** 2
    bounds = Bounds(0, 1)
    # Find the line segment on which the optimal exposure lies
    if len(pareto_curve) == 1:  # pathological case
        exposure_opt = pareto_curve[0]
        return nutility(exposure_opt, pbm), (np.linalg.norm(exposure_opt - target_exposure) / np.linalg.norm(prp_exposure - target_exposure)) ** 2, exposure_opt
    for i in np.arange(0, len(pareto_curve)-1):
        o1 = objective(pareto_curve[i])
        o2 = objective(pareto_curve[i+1])
        if o2 > o1: break  # optimal point is in line segment [i, i+1]
    # try:
    #     a = pareto_curve[i]
    # except UnboundLocalError:
    #     a = 1
    
    sol = minimize(scalarized_objective_within_pareto_segment, 0,
                   (alpha, pareto_curve[i], pareto_curve[i+1], target_exposure, pbm, relevance_vector, prp_exposure, fairness),
                   method="Nelder-Mead", bounds=bounds)
    exposure_opt = sol.x[0] * pareto_curve[i] + (1-sol.x[0]) * pareto_curve[i+1]
    assert objective(exposure_opt) == sol.fun
    return nutility(exposure_opt, pbm, relevance_vector), (np.linalg.norm(exposure_opt - target_exposure) / np.linalg.norm(prp_exposure - target_exposure)) ** 2, exposure_opt




def eval_expohedron(file_eval):
    query_file_path = "/home/phuong/Documents/expohedron/data/TREC_2020/TREC Fair Ranking 2020 Data/clean_queries.json"
    query_file = open(query_file_path, "r", encoding="utf8")
    queries = [json.loads(line) for line in query_file.readlines()]
    queries = [query['qid'] for query in queries]

    with open(file_eval, "rb") as file:
        objs = pickle.load(file)

    epxosure_scores = objs["exposure_score"]

    with open("results/QP.pkl", "rb") as file:
        objs = pickle.load(file)
    
    target_exposure = objs["prob_matrices"]
    alpha_arr = objs["alpha_arr"]

    check = []
    for qid in queries:
        segments = epxosure_scores[qid]
        d_matrices = target_exposure[qid]
        n_doc = len(d_matrices[0])
        gamma = 1 / np.log(np.arange(0, n_doc) + 2)

        for _id in range(len(d_matrices)):
            matrix = d_matrices[_id]
            if matrix is None:
                continue
            p_point = matrix @ gamma
            flag = False
            x=999
            for i in range(len(segments)-1):
                vec_1 = p_point - segments[i]
                vec_2 = segments[i+1] - p_point
                if norm(vec_1) < 1e-6 or norm(vec_2) < 1e-6:
                    x=0
                    flag = True
                    break
                scalar = norm(vec_2) / norm(vec_1)
                x = min(x, norm(vec_2 - scalar*vec_1))
                if scalar < 0 and norm(vec_2 - scalar*vec_1) < 1e-4:
                    x=0
                    flag = True
                    break
            if not flag:
                if x == 999:
                    print(qid)
                    raise Exception("s")
                check.append((qid, _id))

    np.savetxt("results/error_points.txt", np.asarray(check).astype(int))


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

    # np.random.seed(n_doc)
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
    # experiment_QP()
    eval_QP()
    # eval_expohedron("results/projected.pkl")
    # experment_methods()
    exit()

    avg = {
        "sexpo_1_avg": [],
        "sexpo_1_total": [],
        "sexpo_2_avg": [],
        "sexpo_2_total": [],
    }
    for n_doc in tqdm(range(5, 101, 5)):
        ptime = {"avg": [], "total": []}
        stime = {"avg": [], "total": []}
        stime_2 = {"avg": [], "total": []}
        stime_1 = {"avg": [], "total": []}
        qtime = []
        for repeated in tqdm(range(0, 50)):
            # print("Load data")
            _relevance_score, item_group, _group_fairness, _gamma = load_data(n_doc, 2)
            # print("Start hedron experiment:")
            # phedron_start = time.time()
            # pobjs, ppoints = projected_path(_relevance_score, item_group, _group_fairness, _gamma)
            # phedron_end = time.time()
            # ptime["total"].append(phedron_end-phedron_start)
            # ptime["avg"].append((phedron_end - phedron_start) / len(pobjs))

            shedron_start = time.time()
            sobjs, spoints = sphere_path(_relevance_score, item_group, _group_fairness, _gamma, n_divided=2, n_sample=12)
            shedron_end = time.time()
            stime_2["total"].append(shedron_end - shedron_start)
            stime_2["avg"].append((shedron_end - shedron_start) / len(sobjs))

            shedron_start = time.time()
            sobjs, spoints = sphere_path(_relevance_score, item_group, _group_fairness, _gamma, n_divided=1, n_sample=24)
            shedron_end = time.time()
            stime_1["total"].append(shedron_end - shedron_start)
            stime_1["avg"].append((shedron_end - shedron_start) / len(sobjs))

            # print("Start QP experiment:")
            # qp_start = time.time()
            # base_qp = QP.experiment(_relevance_score, item_group, _group_fairness, alpha_arr=[0.1, 0.3, 0.7, 1])
            # qp_end = time.time()
            # qtime.append((qp_end-qp_start)/4)

        # x = st.norm.interval(confidence=0.90, loc=np.mean(ptime["total"]), scale=st.sem(ptime["total"]))
        # avg["pexpo_total"].append([np.mean(ptime["total"]), x])
        # x = st.norm.interval(confidence=0.90, loc=np.mean(ptime["avg"]), scale=st.sem(ptime["avg"]))
        # avg["pexpo_avg"].append([np.mean(ptime["avg"]), x])

        x = st.norm.interval(confidence=0.90, loc=np.mean(stime_1["total"]), scale=st.sem(stime_1["total"]))
        avg["sexpo_1_total"].append([np.mean(stime_1["total"]), x])
        x = st.norm.interval(confidence=0.90, loc=np.mean(stime_1["avg"]), scale=st.sem(stime_1["avg"]))
        avg["sexpo_1_avg"].append([np.mean(stime_1["avg"]), x])
        x = st.norm.interval(confidence=0.90, loc=np.mean(stime_2["total"]), scale=st.sem(stime_2["total"]))
        avg["sexpo_2_total"].append([np.mean(stime_2["total"]), x])
        x = st.norm.interval(confidence=0.90, loc=np.mean(stime_2["avg"]), scale=st.sem(stime_2["avg"]))
        avg["sexpo_2_avg"].append([np.mean(stime_2["avg"]), x])
        # x = st.norm.interval(confidence=0.90, loc=np.mean(stime["total"]), scale=st.sem(stime["total"]))
        # avg["sexpo_total"].append([np.mean(stime["total"]), x])
        # x = st.norm.interval(confidence=0.90, loc=np.mean(stime["avg"]), scale=st.sem(stime["avg"]))
        # avg["sexpo_avg"].append([np.mean(stime["avg"]), x])
        # x = st.norm.interval(confidence=0.90, loc=np.mean(stime["total"]), scale=st.sem(stime["total"]))
        # avg["sexpo_total"].append([np.mean(stime["total"]), x])
        # x = st.norm.interval(confidence=0.90, loc=np.mean(stime["avg"]), scale=st.sem(stime["avg"]))
        # avg["sexpo_avg"].append([np.mean(stime["avg"]), x])

        # x = st.norm.interval(confidence=0.90, loc=np.mean(qtime), scale=st.sem(qtime))
        # avg["qp_avg"].append([np.mean(qtime), x])

    with open("results/stime_12.json", "w") as f_out:
        json.dump(avg, f_out)

    # draw(base_qp[4:], objs)

    # movielen100k_testing()

