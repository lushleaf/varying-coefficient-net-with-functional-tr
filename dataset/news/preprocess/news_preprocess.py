import numpy as np
import pandas as pd

import random

data_path = 'dataset/news/docword.nytimes.txt'
data = pd.read_csv(data_path, sep=' ', header=None)
data_ = data.to_numpy()

doc_id_list = list(set(data_[:,0]))
random.shuffle(doc_id_list)
doc_id_list = doc_id_list[0:3000]
doc_id_list = set(doc_id_list)

select_raw_idx = []

for _ in range(data_.shape[0]):
    if _ % 10000 == 0: print(_)
    if data_[_,0] in doc_id_list:
        select_raw_idx.append(_)

data_ = data_[select_raw_idx,:]

word_id_list = {}

for _ in range(data_.shape[0]):
    if _ % 10000 == 0: print(_)
    if data_[_, 1] not in word_id_list:
        word_id_list[data_[_, 1]] = data_[_, 2]
    else:
        word_id_list[data_[_, 1]] += data_[_, 2]

word_freq = []
for key, value in word_id_list.items():
    word_freq.append(value)
word_freq.sort(reverse=True)

freq_tol = word_freq[500]

selected_word_id_list = []
for key in word_id_list.keys():
    value = word_id_list[key]
    if value > freq_tol:
        selected_word_id_list.append(key)

selected_word_id_list = set(selected_word_id_list)

select_raw_idx2 = []

for _ in range(data_.shape[0]):
    if _ % 10000 == 0: print(_)
    if data_[_,1] in selected_word_id_list:
        select_raw_idx2.append(_)

data_ = data_[select_raw_idx2, :]

np.savetxt('news_p.csv', data_, delimiter=',')

# --------------------------------
doc_id_map = {}
word_id_map = {}
doc_id_list = list(set(data_[:,0]))
word_id_list = list(set(data_[:,1]))

count = 0
for doc in doc_id_list:
    doc_id_map[doc] = count
    count+=1

count = 0
for word in word_id_list:
    word_id_map[word] = count
    count +=1

data_use = np.zeros([len(doc_id_list), len(word_id_list)])
for _ in range(data_.shape[0]):
    doc_id = doc_id_map[data_[_, 0]]
    word_id = word_id_map[data_[_, 1]]
    word_freq = data_[_, 2]
    data_use[doc_id, word_id] = word_freq


np.save('news_pp', data_use)