import pandas as pd
from sklearn.model_selection import train_test_split

def merge(dataset_good, dataset_bad, n):
    dataset_good['key'] = 0
    dataset_bad['key'] = 0
    df_1 = pd.merge(dataset_good, dataset_bad, on='key')
    del df_1['key']

    columns = df_1.columns.tolist()
    reversed_columns = columns[n:] + columns[:n]
    df_2 = df_1[reversed_columns]
    
    abba = pd.concat([df_1, df_2], axis=1, join_axes=[df_1.index])
    abba['first_better'] = 1

    baab = pd.concat([df_2, df_1], axis=1, join_axes=[df_2.index])
    baab['first_better'] = 0

    baab.columns = abba.columns
    both_sides = abba.append(baab, ignore_index=True)
    
    return both_sides

def train_test(both_sides, n, m):
    train, test = train_test_split(both_sides, test_size=0.2)
    left_train = train.iloc[:,:n] # 38 i 76 ako su indeksi ukljuceni
    left_test = test.iloc[:,:n]
    right_train = train.iloc[:,n:m] 
    right_test = test.iloc[:,n:m]
    y_train = train.iloc[:,m].reshape(-1,1)
    y_test = test.iloc[:,m].reshape(-1,1)
    
    return left_train, left_test, right_train, right_test, y_train, y_test