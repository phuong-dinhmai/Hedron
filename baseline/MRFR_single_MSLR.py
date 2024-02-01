import numpy as np
import pandas as pd
import json
import csv
import itertools
import argparse
import os
import sys
from time import time
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

parser = argparse.ArgumentParser()
parser.add_argument("-g", "--gamma", type=float, help="Continuation probability", default=0.9)
parser.add_argument("-l", "--lambd", type=float, help="Weight of fairness wrt utility", default=0.2)
parser.add_argument("-t", "--beta", type=float, help="Weight of deserved exposure wrt rel in init sorting", default=1.0)
parser.add_argument("-e", "--eps", type=float, help="Probability of satisfaction given a relevant document", default=1.0)
parser.add_argument("-b", "--bin_size", type=int, help="Size of the bins of documents", default=3)
parser.add_argument("-n", "--max_n_bins", type=int, help="Maximum number of bins", default=1)
parser.add_argument("-f", "--grouping_file", type=str, help="Grouping file for fairness definition")
args = parser.parse_args()

EPS = args.eps # Probability for the user to be satisfied given that the document is relevant
GAMMA = args.gamma # Continuation probability
LAMBDA = args.lambd # Tradeoff hyperparameter balancing utility and fairness
BETA = args.beta # Tradeoff hyperparameter balancing relevance and deserved exposure in the initial sorting of docs
BIN_SIZE = args.bin_size # Size of the bins of documents to split the rankings
MAX_N_BINS = args.max_n_bins # Maximum number of bins to permute over in the rankings (remaining bins are kept fixed)


def exposure(ranking, n_groups, doc_groups, rel):
    exps = [0.0] * n_groups
    prod = 1
    for i in range(len(ranking)):
        doc = ranking[i]
        if len(doc_groups[doc]) > 0:
            for group in doc_groups[doc]:
                exps[group] += 1/np.log(i+2)
        #         exps[group] += (GAMMA ** i) * prod
        # prod *= 1 - probability(doc, rel)
    return exps


def relevance(ranking, n_groups, doc_groups, rel):
    rels = [0.0] * n_groups
    for i in range(len(ranking)):
        doc = ranking[i]
        if len(doc_groups[doc]) > 0:
            for group in doc_groups[doc]:
                rels[group] += probability(doc, rel)
    return rels


def utility(ranking, rel):
    sum = 0
    prod = 1
    for i in range(len(ranking)):
        doc = ranking[i]
        sum += 1/np.log(i+2) * probability(doc, rel)
    # for i in range(len(ranking)):
    #     doc = ranking[i]
    #     sum += (GAMMA ** i) * prod * probability(doc, rel)
    #     prod *= 1 - probability(doc, rel)
    return sum


def probability(doc, rel):
    return EPS * rel[doc]


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

# Load the queries and documents
train = pd.read_csv("data/MSLR/MSLR-WEB10K/Fold1/vali.txt", header=None, sep=" ")
train_df = df_transform(train)
train_df = train_df.astype(float)
train_df[1] = train_df[1].astype(int)

x = train_df[1].value_counts()
bins = [i for i in range(0, 260, 10)]
labels = [i for i in range(0, 25)]
train_df['binned'] = pd.cut(train_df[133], bins=bins, labels=labels)
# Find a (potentially sub-)optimal ranking for every query
exposure_history = {} # Exposure history per group for each query, used for amortization in repeated queries
relevance_history = {} # Relevance history per group for each query, used for amortization in repeated queries
utility_history = {} # Utility history for each query, used for amortization in repeated queries
discrepancy_history = {} # Discrepancy history for each query
sequence_query_counts = {} # Number of time per sequence each query occurs in different searches
sequence_query_eval_scores = {} # Evaluation scores for each unique query, with amortization for repeated queries
search_rankings = [] # The rankings obtained for each search
unfairness = {}
init_time = time()
cnt = 0

x = x.loc[[175, 340, 445]]

for query_id in x.index:
    if x[query_id] >= 100 or x[query_id] <= 5:
        continue
    sequence_id = cnt

    # Fetch relevant information in the query
    query_docs = train_df.loc[train_df[1] == query_id] # Documents associated with the query
    doc_groups = []
    query_group_dict = {} # Mapping between global group ids and query group ids
    rel_probas = []
    filtered_query_docs = []
    cnt_group = []
    n_groups = 0

    for query_doc_id in query_docs.index:
        query_doc = query_docs.loc[query_doc_id]
        filtered_query_docs.append(query_doc)
        rel_probas.append(query_doc[0] / 5) # Use the groundtruth relevance
        group = int(query_doc["binned"])
        if group not in query_group_dict:
            query_group_dict[group] = n_groups
            cnt_group.append(0)
            n_groups = n_groups + 1
        doc_groups.append([query_group_dict[group]])
        cnt_group[query_group_dict[group]] += 1

    if n_groups < 2:
        continue
    cnt += 1
    for search_id in range(1000):
        if int(search_id) % 100 == 0:
            print("Processing sequence %s -- search %s -- time %.3fs" % (sequence_id, search_id, time() - init_time),
                flush=True)
        
        n_query_docs = len(filtered_query_docs)
        target_exposure = cnt_group / np.sum(cnt_group) * np.sum(1 / np.log(np.arange(0, n_query_docs) + 2))

        # Find a (potentially sub-)optimal ranking for this query
        ## Fetch the history of exposure and relevance if the query has already been processed
        if sequence_id in sequence_query_counts:
            if query_id in sequence_query_counts[sequence_id]:
                cumul_exposures = exposure_history[sequence_id][query_id]
                cumul_relevances = relevance_history[sequence_id][query_id]
                cumul_utilities = utility_history[sequence_id][query_id]
                query_count = sequence_query_counts[sequence_id][query_id]
            else: # First time the query is processed
                cumul_exposures = [0.0] * n_groups
                cumul_relevances = [0.0] * n_groups
                cumul_utilities = 0.0
                query_count = 0
        else: # First time the sequence is processed
            exposure_history[sequence_id] = {}
            relevance_history[sequence_id] = {}
            utility_history[sequence_id] = {}
            discrepancy_history[sequence_id] = {}
            sequence_query_counts[sequence_id] = {}
            sequence_query_eval_scores[sequence_id] = {}
            unfairness[sequence_id] = {}
            cumul_exposures = [0.0] * n_groups
            cumul_relevances = [0.0] * n_groups
            cumul_utilities = 0.0
            query_count = 0

        ## Sort the documents based on relevance and discrepancy
        ### Sort the SORT_CUTOFF top relevance documents
        rel_sort_scores = np.asarray(rel_probas)
        ### Sort the remaining documents according to deserved exposure in descending order, to identify documents with a
        ### higher potential to reduce the query discrepancy after they are re-ranked
        exposure_norm = sum(cumul_exposures)
        relevance_norm = sum(cumul_relevances)
        if exposure_norm > 0 and relevance_norm > 0:
            de_sort_scores = [] # Deserved exposure scores
            for id in range(n_query_docs):
                doc_discrepancy = 0.0
                for g in doc_groups[id]:
                    # print(g)
                    # print(target_exposure[g])
                    # print(cumul_exposures[g])
                    # past_exposure = cumul_exposures[g] / exposure_norm
                    # past_relevance = cumul_relevances[g] / relevance_norm
                    # doc_discrepancy += past_exposure - past_relevance # Past over-exposure
                    past_exposure = cumul_exposures[g] / (query_count + 1)
                    doc_discrepancy += np.square(past_exposure - target_exposure[g])
                doc_discrepancy
                de_sort_scores.append(np.sqrt(doc_discrepancy))
        else:
            de_sort_scores = [0.0] * n_query_docs # If no docs are relevant, consider that all docs get deserved exposure
        de_sort_scores = np.asarray(de_sort_scores)
        ### Split the documents into bins
        sort_scores = rel_sort_scores - BETA * de_sort_scores
        sorted_doc_ids = np.argsort(-sort_scores) # Sort by combination of rel and (negative) over-exposure
        doc_bins = [sorted_doc_ids[i:i + BIN_SIZE] for i in range(0, len(sorted_doc_ids), BIN_SIZE)]

        ## Generate the rankings as all the possible combinations of the bins
        bin_permutations = [list(itertools.permutations(doc_bin)) for doc_bin in doc_bins]
        rankings = [[]]
        for i in range(len(bin_permutations)):
            if i < MAX_N_BINS:
                rankings = [ranking + list(bin_permutations[i][j]) for ranking in rankings
                            for j in range(len(bin_permutations[i]))]
            else:
                rankings = [ranking + list(doc_bins[i]) for ranking in rankings]
        n_rankings = len(rankings)
        ## Evaluate every ranking
        optimal_ranking_id = -1
        optimal_ranking_score = -np.inf
        optimal_ranking_group_exposures = []
        optimal_ranking_group_relevances = []
        optimal_ranking_utility = 0.0
        optimal_ranking_discrepancy = 0.0
        for r in range(n_rankings):
            ranking = rankings[r]

            # Compute the ranking's utility
            util = 1.0 / (query_count + 1) * (utility(ranking, rel_probas) + query_count * cumul_utilities)

            # Compute the normalization constant for exposure
            current_exposures = exposure(ranking, n_groups, doc_groups, rel_probas)
            group_exposures = [current_exposures[g] + cumul_exposures[g] for g in range(n_groups)]
            exposure_norm = sum(group_exposures)

            # Compute the normalization constant for relevance
            current_relevances = relevance(ranking, n_groups, doc_groups, rel_probas)
            group_relevances = [current_relevances[g] + cumul_relevances[g] for g in range(n_groups)]
            relevance_norm = sum(group_relevances)

            # Compute the discrepancy for the current partition
            discrepancy = 0.0
            if exposure_norm > 0 and relevance_norm > 0:
                for g in range(n_groups):
                    # amortized_exposure = group_exposures[g] / exposure_norm
                    # amortized_relevance = group_relevances[g] / relevance_norm
                    # discrepancy += np.square(amortized_exposure - amortized_relevance)
                    amortized_exposure = group_exposures[g] / (query_count + 1)
                    discrepancy += np.square(amortized_exposure - target_exposure[g])
                discrepancy = np.sqrt(discrepancy)
                # print(util, " ", np.array(group_exposures) / (query_count + 1), " ", target_exposure)

            # Compute the overall score for the current ranking on the current partition
            eval_score = util - LAMBDA * discrepancy

            if eval_score > optimal_ranking_score:
                optimal_ranking_id = r
                optimal_ranking_score = eval_score
                optimal_ranking_group_exposures = group_exposures
                optimal_ranking_group_relevances = group_relevances
                optimal_ranking_utility = util
                optimal_ranking_discrepancy = discrepancy

        sequence_query_eval_scores[sequence_id][query_id] = optimal_ranking_score
        optimal_ranking = [filtered_query_docs[d] for d in rankings[optimal_ranking_id]]
        search_rankings.append({'q_num': str(sequence_id) + "." + str(search_id), 'qid': query_id, 'ranking': optimal_ranking})

        # Update history variables
        exposure_history[sequence_id][query_id] = optimal_ranking_group_exposures
        relevance_history[sequence_id][query_id] = optimal_ranking_group_relevances
        utility_history[sequence_id][query_id] = optimal_ranking_utility
        discrepancy_history[sequence_id][query_id] = optimal_ranking_discrepancy
        sequence_query_counts[sequence_id][query_id] = query_count + 1
        unfairness[sequence_id][query_id] = np.linalg.norm(np.array(optimal_ranking_group_exposures) / (query_count+1) - target_exposure)

elapsed_time = time() - init_time
print("Elapsed time (s):", elapsed_time, flush=True)
mean_utility = [np.mean(list(query_utilities.values())) for query_utilities in utility_history.values()]

output_path = f"results/MSLR/"

if not os.path.exists(output_path):
    os.makedirs(output_path)

# # Save the obtained rankings
# ranking_file_path = output_path + "evaluation-rankings-single-g" + str(GAMMA) + "-l" + str(LAMBDA) + "-b" + \
#                     str(BIN_SIZE) + "-n" + str(MAX_N_BINS) + "-t" + str(BETA) + "-e" + str(EPS) + ".json"
#
# with open(ranking_file_path, "w", encoding="utf8") as ranking_file:
#     for search_ranking in search_rankings:
#         ranking_file.write(json.dumps(search_ranking) + "\n")

# Save the log
log_file_path = output_path + "evaluation-log-single-g" + str(GAMMA) + "-l" + str(LAMBDA) + "-b" + \
                str(BIN_SIZE) + "-n" + str(MAX_N_BINS) + "-t" + str(BETA) + "-e" + str(EPS) + ".json"

results = {
    "utility": utility_history,
    "unfairness": unfairness,
    "exposure_history": exposure_history,
    # "sequence_cnt": sequence_query_counts
}

with open(log_file_path, "w", encoding="utf8") as log_file:
    json.dump(results, log_file)