##
import os
import numpy as np
import pandas as pd
import tensorflow as tf
from math import inf
from scipy.special import expit
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import Imputer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegressionCV
from itertools import product
from itertools import combinations
import matplotlib.pyplot as plt
import matplotlib

import model.model_dummy as M
import model.model_b as M_b

matplotlib.rcParams["text.usetex"] = True

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

np.random.seed(13377)

save_dir = "./save/"

##
data = pd.read_csv("gmscredit_dataset/cs-training.csv", index_col=0)[:600]
drop_features = ["SeriousDlqin2yrs"]
target = ["SeriousDlqin2yrs"]
X = data.drop(drop_features, axis=1)
y = data[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.2)

imp = Imputer()
X_train = imp.fit_transform(X_train)
X_valid = imp.transform(X_valid)
X_test = imp.transform(X_test)

mean = X_train.mean(axis=0)
std = X_train.std(axis=0)
X_train = (X_train - mean) / std
X_valid = (X_valid - mean) / std
X_test = (X_test - mean) / std
##

X_train_good = pd.DataFrame(X_train[y_train.squeeze() == 0])
#X_train_good = X_train_good.loc[0:1, :]
X_train_bad = pd.DataFrame(X_train[y_train.squeeze() == 1])
#X_train_bad = X_train_bad.loc[0:1, :]
#pairs_index_train = np.array(list(product(X_train_good.index, X_train_bad.index)))

X_valid_good = pd.DataFrame(X_valid[y_valid.squeeze() == 0])
X_valid_bad = pd.DataFrame(X_valid[y_valid.squeeze() == 1])
#pairs_index_valid = np.array(list(product(X_valid_good.index, X_valid_bad.index)))

X_test_good = pd.DataFrame(X_test[y_test.squeeze() == 0])
X_test_bad = pd.DataFrame(X_test[y_test.squeeze() == 1])
#pairs_index_test = np.array(list(product(X_test_good.index, X_test_bad.index)))

##
good_index = list(X_train_good.index)
np.random.shuffle(good_index)
good_index_valid = list(X_valid_good.index)
np.random.shuffle(good_index_valid)
good_index_test = list(X_test_good.index)
np.random.shuffle(good_index_test)
##


def run_dataset(left_pairs, A, cnt, summary_writer=None, epoch=0):
    if summary_writer:
        outputs, loss, summary = M.sess.run([M.logit, M.loss, M.summary],
                                            feed_dict={M.batch_left: left_pairs, M.A: A, M.cnt: cnt})
        summary_writer.add_summary(summary, epoch)
    else:
        outputs = M.sess.run(M.logit, feed_dict={M.batch_left: left_pairs})
    outputs = expit(np.mean(outputs, axis=1))
    return outputs, loss


def print_score(left_pairs, A, cnt, output_summary=None, epoch=0):
    outputs, loss = run_dataset(left_pairs, A, cnt, output_summary, epoch)
    y_score = np.sign(np.array(outputs) - 0.5)
    somers_d = sum(y_score) / len(y_score)
    return somers_d, loss

##

def get_index_batch(i, batch_size, num_of_batches, residue, good_index):
    p = i * batch_size
    r = (i + 1) * batch_size
    if (residue != 0 and i + 1 == num_of_batches) or batch_size == 0:
        r = p + residue
    return good_index[p:r]


def good_bad_pairs(good, bad, index_batch):
    lb = len(bad)
    lba = len(index_batch)

    x_good = np.repeat(np.array(good.loc[index_batch]), lb, axis=0).reshape(-1, 10)
    x_bad = np.tile(np.array(bad), (lba, 1))
    pairs = np.concatenate((x_good, x_bad), axis=1)
    #np.random.shuffle(pairs)

    c = np.repeat(np.array(index_batch).reshape(-1, 1), lb, axis=0)
    d = np.tile(np.array(bad.index).reshape(-1, 1), (lba, 1))
    pairs_index = np.concatenate((c, d), axis=1)

    return x_good, x_bad, pairs_index, pairs


def get_a_cnt(gb_index):
    list1 = np.unique(gb_index[:, 0]).tolist() + np.unique(gb_index[:, 1]).tolist()
    cnt = len(list1)
    A = np.zeros((len(gb_index), cnt), dtype=float)
    n = 0
    for pair in gb_index:
        A[n, list1.index(pair[0])] = 1
        A[n, cnt - 1 - list1[::-1].index(pair[1])] = -1
        n += 1
    return A, cnt


def validate_test(batch_size, good_index_valid, good, bad):
    sum_somers_d = 0
    sum_loss = 0
    num_of_batches = (len(good_index_valid)* len(bad)) // (batch_size * len(bad))
    residue = (len(good_index_valid) * len(bad)) % (batch_size * len(bad))

    for i in range(num_of_batches):
        index_batch = get_index_batch(i, batch_size, num_of_batches, residue, good_index_valid)
        x_good, x_bad, gb_index, pairs = good_bad_pairs(good, bad, index_batch)
        A, cnt = get_a_cnt(gb_index)
        somers_d, loss = print_score(pairs, A, cnt, M.test_writer)
        sum_somers_d += somers_d * len(gb_index)
        sum_loss += loss * len(gb_index)
        if batch_size == 0:
            break
    avg_loss = sum_loss / (num_of_batches * batch_size * len(bad) + residue)
    avg_sd = sum_somers_d / (num_of_batches * batch_size * len(bad) + residue)
    print("Loss:", avg_loss, "\nValid Somers' D ", avg_sd)
    return avg_loss, avg_sd

##

M.sess.run(tf.global_variables_initializer())
batch_size = 2
num_of_batches = (len(good_index) * len(X_train_bad)) // (batch_size*len(X_train_bad))
residue = (len(good_index) * len(X_train_bad)) % (batch_size*len(X_train_bad))
print("Number of batches: ", num_of_batches)

#old_sd = -1
for ep in range(1):
    print(ep)
    for i in range(num_of_batches):

        index_batch = get_index_batch(i, batch_size, num_of_batches, residue, good_index)
        x_good, x_bad, gb_index, pairs = good_bad_pairs(X_train_good, X_train_bad, index_batch)
        A, cnt = get_a_cnt(gb_index)

        M.sess.run(M.iterator.initializer, feed_dict={M.input_l_placeholder: pairs, M.batch_size: len(gb_index)})

        while True:
            try:
                #M.sess.run(M.train_op, feed_dict={M.learning_rate: 1e-3, M.input_keep_prob: 1.0, M.keep_prob: 1.0, M.l1_decay: 0})
                M.sess.run(M.train_op, feed_dict={M.A: A, M.cnt: cnt, M.learning_rate: 1e-3, M.input_keep_prob: 1.0, M.keep_prob: 1.0, M.l1_decay: 0})
            except tf.errors.OutOfRangeError:
                break

        if i % 100 == 0:
            print("\nEpoch #{:^{}}".format(i, 3))
            new_loss, new_sd = validate_test(batch_size, good_index_valid, X_valid_good, X_valid_bad)
            #if new_sd < old_sd:
            #    break
            #else:
            #    old_sd = new_sd

## KOMBINACIJE DOBRIH S DOBRIMA I LOSIH S LOSIMA
comb_gg = np.array(list(combinations(list(X_train_good.index), 2)))
comb_bb = np.array(list(combinations(list(X_train_bad.index), 2)))
## SVE TROJKE OD 2 LOSA I JEDNOG DOBROG PRIMJERA
prva = []
for n in range(len(comb_bb)):
    print("%d / %d" % (n, len(comb_bb)))
    i = comb_bb[n, 0]
    j = comb_bb[n, 1]
    p3 = np.concatenate(
        (np.array(X_train_bad.loc[i, :]).reshape(1, -1), np.array(X_train_bad.loc[j, :]).reshape(1, -1)), axis=1)

    for k in range(len(X_train_good)):
        p1 = np.concatenate(
            (np.array(X_train_good.loc[k, :]).reshape(1, -1), np.array(X_train_bad.loc[i, :]).reshape(1, -1)), axis=1)
        p2 = np.concatenate(
            (np.array(X_train_good.loc[k, :]).reshape(1, -1), np.array(X_train_bad.loc[j, :]).reshape(1, -1)), axis=1)
        prva.append([p1, p2, p3])
## SVE TROJKE OD 2 DOBRA I JEDNOG LOSEG
druga = []
for m in range(len(comb_gg)):
    print("%d / %d" % (m, len(comb_gg)))
    i = comb_gg[m, 0]
    j = comb_gg[m, 1]
    p3 = np.concatenate(
        (np.array(X_train_good.loc[i, :]).reshape(1, -1), np.array(X_train_good.loc[j, :]).reshape(1, -1)), axis=1)

    for k in range(len(X_train_bad)):
        p1 = np.concatenate(
            (np.array(X_train_good.loc[i, :]).reshape(1, -1), np.array(X_train_bad.loc[k, :]).reshape(1, -1)),axis=1)
        p2 = np.concatenate(
            (np.array(X_train_good.loc[j, :]).reshape(1, -1), np.array(X_train_bad.loc[k, :]).reshape(1, -1)),axis=1)
        druga.append([p1, p2, p3])
##
sve = []
sve.extend(prva)
sve.extend(druga)
np.random.shuffle(sve)
sve = np.array(sve).reshape(-1, 20)
##
M.sess.run(M.iterator.initializer, feed_dict={M.input_l_placeholder: sve, M.batch_size: 30000})

while True:
    try:
        #M.sess.run(M.train_op, feed_dict={M.learning_rate: 1e-3, M.input_keep_prob: 1.0, M.keep_prob: 1.0, M.l1_decay: 0})
        reg,_ = M.sess.run([M.reg, M.train_op_1], feed_dict={M.A: A, M.cnt: cnt, M.learning_rate: 5*1e-6, M.input_keep_prob: 1.0, M.keep_prob: 1.0, M.l1_decay: 0})
    except tf.errors.OutOfRangeError:
        break

#if n % 100 == 0:
    #print("\nEpoch #{:^{}}".format(n, 3))
    #new_loss = validate_test(batch_size, good_index_valid, X_valid_good, X_valid_bad)

## TEST
validate_test(batch_size, good_index_test, X_test_good, X_test_bad)
