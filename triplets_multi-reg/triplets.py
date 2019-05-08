##
import os, time
import numpy as np
import pandas as pd
import tensorflow as tf
from scipy.special import expit
from sklearn.preprocessing import Imputer
from sklearn.model_selection import train_test_split
from itertools import combinations
import matplotlib

import model.model_dummy as M
import train.util as util
matplotlib.rcParams["text.usetex"] = True
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
np.random.seed(13377)
save_dir = "./save/"

##
# PRIPREMA PODATAKA

data = pd.read_csv("gmscredit_dataset/cs-training.csv", index_col=0)[:200]
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

# IZDVAJANJE DOBRIH I LOSIH PRIMJERA (potrebno za kreiranje parova) I
# KREIRANJE LISTE INDEKSA DOBRIH PRIMJERA KOJI SE KORISTI PRI minibatchevima

X_train_good = pd.DataFrame(X_train[y_train.squeeze() == 0])
X_train_bad = pd.DataFrame(X_train[y_train.squeeze() == 1])
good_index = list(X_train_good.index)
np.random.shuffle(good_index)

X_valid_good = pd.DataFrame(X_valid[y_valid.squeeze() == 0])
X_valid_bad = pd.DataFrame(X_valid[y_valid.squeeze() == 1])
good_index_valid = list(X_valid_good.index)
np.random.shuffle(good_index_valid)

X_test_good = pd.DataFrame(X_test[y_test.squeeze() == 0])
X_test_bad = pd.DataFrame(X_test[y_test.squeeze() == 1])
good_index_test = list(X_test_good.index)
np.random.shuffle(good_index_test)

# KOMBINACIJE DOBRIH S DOBRIMA I LOSIH S LOSIMA (potrebno kod trojki)
comb_gg = np.array(list(combinations(list(X_train_good.index), 2)))
comb_bb = np.array(list(combinations(list(X_train_bad.index), 2)))


#pairs_index_train = np.array(list(product(X_train_good.index, X_train_bad.index)))
#pairs_index_valid = np.array(list(product(X_valid_good.index, X_valid_bad.index)))
#pairs_index_test = np.array(list(product(X_test_good.index, X_test_bad.index)))
##

# FUNKCIJA ZA DOBIVANJE KONACNOG SOMERS'D REZULTATA, GUBITKA I MJERE NEKONZISTENTNOSTI
def print_score(left_pairs, A, cnt):
    rem = np.arange(left_pairs.shape[0]).reshape(-1, 1) % 3
    outputs, loss, i_f, summary = M.sess.run([M.logit, M.loss, M.i_f, M.summary],
                                             feed_dict={M.batch_left: left_pairs, M.rem: rem.astype(float),
                                                        M.A: A, M.cnt: cnt})
    outputs = expit(np.mean(outputs, axis=1))
    y_score = np.sign(np.array(outputs) - 0.5)
    somers_d = sum(y_score) / len(y_score)
    return somers_d, loss, i_f


def validate_test(batch_size, good_index_valid, good, bad):
    sum_somers_d = 0
    sum_loss = 0
    num_of_batches = (len(good_index_valid)* len(bad)) // (batch_size * len(bad))
    residue = (len(good_index_valid) * len(bad)) % (batch_size * len(bad))
    i_f_list = []
    for i in range(num_of_batches):
        index_batch = util.get_index_batch(i, batch_size, num_of_batches, residue, good_index_valid)
        x_good, x_bad, gb_index, pairs = util.good_bad_pairs(good, bad, index_batch)
        A, cnt = util.get_a_cnt(gb_index)
        somers_d, loss, i_f = print_score(pairs, A, cnt)
        i_f_list.append(i_f)
        sum_somers_d += somers_d * len(gb_index)
        sum_loss += loss * len(gb_index)
        if batch_size == 0:
            break
    avg_loss = sum_loss / (num_of_batches * batch_size * len(bad) + residue)
    avg_sd = sum_somers_d / (num_of_batches * batch_size * len(bad) + residue)
    i_f_list = np.array(i_f_list)
    print("Loss:", avg_loss, "\nValid Somers' D ", avg_sd, "\nKonzistentnost min, max, mean: ", np.min(i_f_list), np.max(i_f_list), np.mean(i_f_list))
    return avg_loss, avg_sd


##
# SPREMANJE MODELA S INICIJALNIM PARAMTERIMA
saver = tf.train.Saver()
M.sess.run(tf.global_variables_initializer())
save_path = saver.save(M.sess, "/tmp/model_init.ckpt")


##
# UCITAVANJE INICIJALNOG MODELA I NJEGOVO TESTIRANJE TESTIRANJE
tf.reset_default_graph()
saver.restore(M.sess, "/tmp/model_init.ckpt")
validate_test(2, good_index_test, X_test_good, X_test_bad)


##
# UCENJE INICIJALNOG MODELA BEZ REGULARIZACIJE (samo nad raznovrsnim parovima)
batch_size = 2; print_batch = 30; i_f_list = []
num_of_batches = (len(good_index) * len(X_train_bad)) // (batch_size*len(X_train_bad))
residue = (len(good_index) * len(X_train_bad)) % (batch_size*len(X_train_bad))

print("Number of batches: ", num_of_batches)

for i in range(num_of_batches):
    index_batch = util.get_index_batch(i, batch_size, num_of_batches, residue, good_index)
    x_good, x_bad, gb_index, pairs = util.good_bad_pairs(X_train_good, X_train_bad, index_batch)
    A, cnt = util.get_a_cnt(gb_index)

    M.sess.run(M.iterator.initializer, feed_dict={M.input_l_placeholder: pairs, M.batch_size: len(gb_index)})

    while True:
        try:
            rem = np.arange(len(gb_index)).reshape(-1, 1) % 3
            i_f, _ = M.sess.run([M.i_f, M.train_op], feed_dict={M.learning_rate: 1e-4, M.lambda_: 0,
                                                                M.rem: rem.astype(float), M.A: A, M.cnt: cnt})
            i_f_list.append(i_f)
        except tf.errors.OutOfRangeError:
            break
    if i % print_batch == 0:
        print("\nEpoch #{:^{}}".format(i, 3))
        new_loss, new_sd = validate_test(batch_size, good_index_valid, X_valid_good, X_valid_bad)

# ISPIS REZULTATA NEKONZISTENTNOSTI TIJEKOM UČENJA
i_f_list = np.array(i_f_list)
print("Konzistentnost (min, max, mean): ", np.min(i_f_list), np.max(i_f_list), np.mean(i_f_list))

# SPREMANJE NAUČENOG MODELA I NJEGOVO TESTIRANJE
save_path = saver.save(M.sess, "/tmp/model.ckpt")
validate_test(3, good_index_test, X_test_good, X_test_bad)



##
# UCITAVANJE INICIJALNOG ili PRETHODNO NAUČENOG MODELA I NJEGOVO TESTIRANJE
tf.reset_default_graph()
saver.restore(M.sess, "/tmp/model.ckpt")
validate_test(3, good_index_test, X_test_good, X_test_bad)


##
# UCENJE UCITANOG MODELA S REGULARIZACIJOM (nad TROJKAMA)
a = 0; lambda_ = 0; i_f_list = []

#JEDAN MINIBATCH CINE TROJKE KOJE SE MOGU KREIRATI OD NEKA 2 DOBRA I SVIH LOSIH PRIMJERA
# za jedan par od parova good-good:
for i, j in comb_gg:

    # PAROVI UNUTARIH SVIH TROJKI BAD-BAD-GOOD (koji se mogu iskombinirati
    # s jedim od odabranih 2 dobra primjera i od para losih primjera iz svih losih

    prva, prva_index = util.triplets_bbg(comb_bb, X_train_good.loc[[i, j], :], X_train_bad)

    # PAROVI UNUTARIH SVIH TROJKI GOOD-GOOD-BAD (koji se mogu iskombinirati
    # s odabranim parom dobrih primjera (iz iteracije) i od jednog loseg primjera iz svih losih

    druga, druga_index = util.triplets_ggb(i, j, X_train_good, X_train_bad)

    sve = np.concatenate((prva, druga), axis=0)
    sve_index = np.concatenate((prva_index, druga_index), axis=0)
    a += sve.shape[0]

    A, cnt = util.get_a_cnt_triplets(sve_index, len(prva_index), [i, j] + X_train_bad.index.tolist(), len(X_train_bad))

    M.sess.run(M.iterator.initializer, feed_dict={M.input_l_placeholder: sve, M.batch_size: sve.shape[0]})

    while True:
        try:
            rem = np.arange(sve.shape[0]).reshape(-1, 1) % 3
            l, re, i_f, _ = M.sess.run([M.triplet_loss, M.reg, M.i_f, M.train_op_triplets],
                                       feed_dict={M.learning_rate: 1e-2, M.rem: rem.astype(float),
                                                  M.lambda_: lambda_, M.A: A, M.cnt: cnt})
            #print(l + lambda_ * re, l, re, i_f)
            i_f_list.append(i_f)
        except tf.errors.OutOfRangeError:
            break


# ISPIS REZULTATA NEKONZISTENTNOSTI TIJEKOM UČENJA
i_f_list = np.array(i_f_list)
print("Konzistentnost... min, max, mean: ", np.min(i_f_list), np.max(i_f_list), np.mean(i_f_list))

# SPREMANJE NAUČENOG MODELA I NJEGOVO TESTIRANJE
save_path = saver.save(M.sess, "/tmp/model_triplets.ckpt")
validate_test(3, good_index_test, X_test_good, X_test_bad)




