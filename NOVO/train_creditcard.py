##
import os
import numpy as np
import pandas as pd
import tensorflow as tf
from math import inf
from scipy.special import expit
from sklearn.utils import shuffle
from sklearn.preprocessing import Imputer
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
import matplotlib

matplotlib.rcParams["text.usetex"] = True

import model.model_dummy as M

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

np.random.seed(13377)

save_dir = "./save/"

##
data = pd.read_csv("creditcard_dataset/UCI_Credit_Card.csv", index_col=0)[:10000]
drop_features = ["default.payment.next.month"]
target = ["default.payment.next.month"]
X = data.drop(drop_features, axis=1)
y = data[target]

##
#   podjela dataseta na train, validate i test

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)
X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.1)

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


def run_dataset(left_pairs, summary_writer=None, epoch=0):
    if summary_writer:
        outputs, loss, summary = M.sess.run([M.logit, M.loss, M.summary], feed_dict={M.batch_left: left_pairs})
        summary_writer.add_summary(summary, epoch)
    else:
        outputs = M.sess.run(M.logit, feed_dict={M.batch_left: left_pairs})
    outputs = expit(np.mean(outputs, axis=1))
    return outputs, loss


def print_score(left_pairs, output_summary=None, epoch=0):
    outputs, loss = run_dataset(left_pairs, output_summary, epoch)
    y_score = np.sign(np.array(outputs) - 0.5)
    somers_d = sum(y_score) / len(y_score)
    return somers_d, loss


def prepare_permutations(x, y, epochs):
    good = x[y.squeeze() == 0]
    bad = x[y.squeeze() == 1]
    m = len(good)
    n = len(bad)
    perm = np.random.permutation(m * n)
    perm_size = len(perm) // epochs
    residue = len(perm) % epochs
    return good, bad, m, perm, perm_size, residue

##
# Podjela train seta na dobre i lose, a zatim priprema za njihove permutacije,.. isto i za valid set


epochs = 1000
X_train_good, X_train_bad, m, perm, perm_size, residue = prepare_permutations(X_train, y_train, epochs)
X_valid_good, X_valid_bad, m_v, perm_v, perm_v_size, residue_v = prepare_permutations(X_valid, y_valid, epochs)
X_test_good, X_test_bad, m_t, perm_t, perm_t_size, residue_t = prepare_permutations(X_test, y_test, epochs)
print(X_train_good.shape, X_train_bad.shape, X_valid_good.shape, X_valid_bad.shape, X_test_good.shape, X_test_bad.shape)
print(perm_size, residue, perm_v_size, residue_v, perm_t_size, residue_t)

##


def batch_pairs(i, perm_size, residue, epochs, perm, m, good, bad):
    p = i * perm_size
    r = (i + 1) * perm_size
    if residue != 0 and i + 1 == epochs:
        r = p + residue
    perm_batch = perm[p:r]
    perm_good = np.take(good, perm_batch % m, axis=0)
    perm_bad = np.take(bad, perm_batch // m, axis=0)
    return np.concatenate((perm_good, perm_bad), axis=1)


def validate_test(perm_size, residue, epochs, perm, m, good, bad):
    sum_somersd = 0
    sum_loss = 0
    for j in range(epochs):
        pairs = batch_pairs(j, perm_size, residue, epochs, perm, m, good, bad)
        somersd, loss = print_score(pairs, M.test_writer)
        sum_somersd += somersd*len(pairs)
        sum_loss += loss
    print("Loss:", sum_loss, "\nValid Somers' D ", sum_somersd/len(perm))
    return sum_loss


M.sess.run(tf.global_variables_initializer())
prev_loss = inf
for i in range(epochs):
    print("Epoch #{:^{}}".format(i, 3))

    concatenated_pairs = batch_pairs(i, perm_size, residue, epochs, perm, m, X_train_good, X_train_bad)

    M.sess.run(M.iterator.initializer, feed_dict={M.input_l_placeholder: concatenated_pairs})

    while True:
        try:
            M.sess.run(M.train_op, feed_dict={M.learning_rate: 1e-3, M.batch_size: 10000,
                                              M.input_keep_prob: 1.0, M.keep_prob: 1.0,
                                              M.l1_decay: 0.0001})
        except tf.errors.OutOfRangeError:
            break

    train_somersd, train_loss = print_score(concatenated_pairs, M.train_writer)
    print("Loss:", train_loss, "\nTrain Somers' D ",train_somersd)
    if i % 50 == 0:
        new_loss = validate_test(perm_v_size, residue_v, epochs, perm_v, m_v, X_valid_good, X_valid_bad)
        if new_loss > prev_loss:
            print("Early stopping - iteration ", i)
            break
        else:
            prev_loss = new_loss
    print()

print("\nTesting...............")
validate_test(perm_t_size, residue_t, epochs, perm_t, m_t, X_test_good, X_test_bad)
