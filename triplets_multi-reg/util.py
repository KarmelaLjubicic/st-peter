import numpy as np
from scipy.special import expit


# FUNKCIJA ZA DOBIVANJE MINIBATCH PODSKUPA DOBRIH PRIMJERA (koji se kasnije kombiniraju sa svim losim)
def get_index_batch(i, batch_size, num_of_batches, residue, good_index):
    p = i * batch_size
    r = (i + 1) * batch_size
    if (residue != 0 and i + 1 == num_of_batches) or batch_size == 0:
        r = p + residue
    return good_index[p:r]


# FUNKCIJA ZA KOMBINIRANJE DOBRIH I LOSIH PRIMJERA U RAZNOVRSNE PAROVE
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


# FUNKCIJA ZA IZRACUN MATRICE INCIDENCIJE
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


# FUNKCIJA ZA SLAGANJE SVIH BAD-BAD-GOOD TROJKI OD DANIH DOBRIH I LOSIH PRIMJERA
def triplets_bbg(comb_bb, X_train_good, X_train_bad):
    prva = []
    prva_index = []
    app = prva.append
    app_1 = prva_index.append

    for i, j in comb_bb:
        x1 = np.array(X_train_bad.loc[i, :]).reshape(1, -1)
        x2 = np.array(X_train_bad.loc[j, :]).reshape(1, -1)
        p3 = np.concatenate((x1, x2), axis=1)
        p3_i = [i, j]
        ls = X_train_good.index.tolist()
        for k in ls:
            x3 = np.array(X_train_good.loc[k, :]).reshape(1, -1)
            p1 = np.concatenate((x3, x1), axis=1)
            p2 = np.concatenate((x3, x2), axis=1)
            app([p1, p2, p3])
            app_1([[k, i], [k, j], p3_i])
    return np.array(prva).reshape(-1, 20), np.array(prva_index).reshape(-1, 2)


# FUNKCIJA ZA SLAGANJE SVIH BAD-GOOD-GOOD TROJKI OD DANIH DOBRIH I LOSIH PRIMJERA
def triplets_ggb(i, j, X_train_good, X_train_bad):
    druga = []
    druga_index = []
    app = druga.append
    app_1 = druga_index.append

    #for m in range(len(comb_gg)):
    #    print("%d / %d" % (m, len(comb_gg)))
    x1 = np.array(X_train_good.loc[i, :]).reshape(1, -1)
    x2 = np.array(X_train_good.loc[j, :]).reshape(1, -1)
    p3 = np.concatenate((x1, x2), axis=1)
    p3_i = [i, j]
    for k in range(len(X_train_bad)):
        x3 = np.array(X_train_bad.loc[k, :]).reshape(1, -1)
        p1 = np.concatenate((x1, x3), axis=1)
        p2 = np.concatenate((x2, x3), axis=1)
        app([p1, p2, p3])
        app_1([[i, k], [j, k], p3_i])
    return np.array(druga).reshape(-1, 20), np.array(druga_index).reshape(-1, 2)


# FUNKCIJA ZA IZRACUN MATRICE INCIDENCIJE KAD RADIMO S TROJKAMA
def get_a_cnt_triplets(sve_index, len_prva, list1, len_bad):
    cnt = len(list1)
    A = np.zeros((len(sve_index), 2 + len_bad), dtype=float)
    n = 0
    for pair in sve_index:
        if n % 3 == 2:
            if n < len_prva:
                A[n, cnt - 1 - list1[::-1].index(pair[0])] = 1
                A[n, cnt - 1 - list1[::-1].index(pair[1])] = -1
            else:
                A[n, list1.index(pair[0])] = 1
                A[n, list1.index(pair[1])] = -1
        else:
            A[n, list1.index(pair[0])] = 1
            A[n, cnt - 1 - list1[::-1].index(pair[1])] = -1
        n += 1
    return A, cnt
