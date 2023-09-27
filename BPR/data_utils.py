import os
import random
import pickle
import argparse
from collections import deque
import math

import numpy as np
import pandas as pd

from torch.utils.data import IterableDataset, get_worker_info


def convert_unique_idx(df, column_name):
    column_dict = {x: i for i, x in enumerate(df[column_name].unique())}
    df[column_name] = df[column_name].apply(column_dict.get)
    df[column_name] = df[column_name].astype('int')
    assert df[column_name].min() == 0
    assert df[column_name].max() == len(column_dict) - 1
    return df, column_dict


def create_user_list(df, user_size):
    user_list = [list() for u in range(user_size)]
    for row in df.itertuples():
        user_list[row.user].append((row.time, row.item))
    return user_list


def create_pair(user_list):
    pair = []
    for user, item_list in enumerate(user_list):
        pair.extend([(user, item) for item in item_list])
    return pair


def split_train_test(df, user_size, test_size=0.2, time_order=False):
    """Split a dataset into `train_user_list` and `test_user_list`.
    Because it needs `user_list` for splitting dataset as `time_order` is set,
    Returning `user_list` data structure will be a good choice."""
    # TODO: Handle duplicated items
    if not time_order:
        test_idx = np.random.choice(len(df), size=int(len(df)*test_size))
        train_idx = list(set(range(len(df))) - set(test_idx))
        test_df = df.loc[test_idx].reset_index(drop=True)
        train_df = df.loc[train_idx].reset_index(drop=True)
        test_user_list = create_user_list(test_df, user_size)
        train_user_list = create_user_list(train_df, user_size)
    else:
        total_user_list = create_user_list(df, user_size)
        train_user_list = [None] * user_size
        test_user_list = [None] * user_size
        for user, item_list in enumerate(total_user_list):
            # Choose latest item
            item_list = sorted(item_list, key=lambda x: x[0])
            # Split item
            test_item = item_list[math.ceil(len(item_list)*(1-test_size)):]
            train_item = item_list[:math.ceil(len(item_list)*(1-test_size))]
            # Register to each user list
            test_user_list[user] = test_item
            train_user_list[user] = train_item
    # Remove time
    test_user_list = [list(map(lambda x: x[1], l)) for l in test_user_list]
    train_user_list = [list(map(lambda x: x[1], l)) for l in train_user_list]
    return train_user_list, test_user_list


def load_all(test_num=100):
    data = pd.read_csv('../data/movielens_100k/ml-100k/u1.base', sep="\t", header=None)
    data.columns = ['user', 'item', 'rating', 'time']
    data = data[data["rating"] >= 3]
    
    data, user_dict = convert_unique_idx(data, "user")
    user_size = len(user_dict)
    data, movie_dict = convert_unique_idx(data, "item")
    movie_size = len(movie_dict)

    train_user_list, test_user_list = split_train_test(data,
                                                       user_size,
                                                       test_size=0.2,
                                                       time_order=True)

    train_pair = create_pair(train_user_list)
    dataset = {'user_size': user_size, 'item_size': movie_size, 
            'user_mapping': user_dict, 'item_mapping': movie_dict,
            'train_user_list': train_user_list, 'test_user_list': test_user_list,
            'train_pair': train_pair}
    with open("/home/phuong/Documents/expohedron/BPR/output/data.pkl", 'wb') as f:
        pickle.dump(dataset, f, protocol=pickle.HIGHEST_PROTOCOL)
    return dataset


class TripletUniformPair(IterableDataset):
    def __init__(self, num_item, user_list, pair, shuffle, num_epochs):
        self.num_item = num_item
        self.user_list = user_list
        self.pair = pair
        self.shuffle = shuffle
        self.num_epochs = num_epochs

    def __iter__(self):
        worker_info = get_worker_info()
        # Shuffle per epoch
        self.example_size = self.num_epochs * len(self.pair)
        self.example_index_queue = deque([])
        self.seed = 0
        if worker_info is not None:
            self.start_list_index = worker_info.id
            self.num_workers = worker_info.num_workers
            self.index = worker_info.id
        else:
            self.start_list_index = None
            self.num_workers = 1
            self.index = 0
        return self

    def __next__(self):
        if self.index >= self.example_size:
            raise StopIteration
        # If `example_index_queue` is used up, replenish this list.
        while len(self.example_index_queue) == 0:
            index_list = list(range(len(self.pair)))
            if self.shuffle:
                random.Random(self.seed).shuffle(index_list)
                self.seed += 1
            if self.start_list_index is not None:
                index_list = index_list[self.start_list_index::self.num_workers]
                # Calculate next start index
                self.start_list_index = (self.start_list_index + (self.num_workers - (len(self.pair) % self.num_workers))) % self.num_workers
            self.example_index_queue.extend(index_list)
        result = self._example(self.example_index_queue.popleft())
        self.index += self.num_workers
        return result

    def _example(self, idx):
        u = self.pair[idx][0]
        i = self.pair[idx][1]
        j = np.random.randint(self.num_item)
        while j in self.user_list[u]:
            j = np.random.randint(self.num_item)
        return u, i, j
		