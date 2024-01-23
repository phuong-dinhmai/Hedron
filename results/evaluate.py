import numpy as np
import pandas as pd
import pickle
import json

import matplotlib.pyplot as plt
import scipy.stats as st

query_file_path = "D:\\VinUni_researchAssistant\\Hedron\\data\\TREC\\clean_queries.json"
query_file = open(query_file_path, "r", encoding="utf8")
queries = [json.loads(line) for line in query_file.readlines()]
queries = {query['qid']: query for query in queries}  # The data of unique queries

group_dict = pd.read_csv("D:\\VinUni_researchAssistant\\Hedron\\data\\TREC\\new_group_2.csv", index_col=0)

r_path = "D:\\VinUni_researchAssistant\\Hedron\\results\\new_group_2.csv\\summary.json"
r_file = open(r_path, "r", encoding="utf8")
results = [json.loads(line) for line in r_file.readlines()]

group_fairness = {}
dcg = {}

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
    group_fairness[qid] = group_size / np.sum(group_size) * np.sum(gamma)
    dcg[qid] = np.sort(gamma) @ np.sort(rel)

cnt = 0
y = np.zeros([len(results)//2, 2])
for result in results:
    if cnt % 2 == 0:
        utils = []
        for seq in result.values():
            qid = list(seq.keys())[0]
            values = list(seq.values())[0]
            utils.append(values / dcg[int(qid)])
        y[cnt//2, 0] = np.mean(utils)
        print(np.mean(utils))
    else:
        fair = []
        for seq in result.values():
            qid = list(seq.keys())[0]
            values = list(seq.values())[0]
            x = np.asarray(values) / 1000 - group_fairness[int(qid)]
            if np.abs(x[0]) > 6:
                continue
            fair.append(np.sum((np.asarray(values) / 1000 - group_fairness[int(qid)]) ** 2))
        y[(cnt-1)//2, 1] = np.mean(fair)
        print(np.mean(fair))
    cnt += 1
print(y)





