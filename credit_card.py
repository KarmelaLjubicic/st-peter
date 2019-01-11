import pandas as pd
from math import *
import util
from sklearn.preprocessing import MinMaxScaler, Imputer

def get_all():
    dataset = prepare_credit_data("data\\UCI_Credit_Card.csv")
    
    dataset_good = dataset[dataset['default.payment.next.month'] == 0]
    dataset_bad = dataset[dataset['default.payment.next.month'] == 1]
    del dataset_good['default.payment.next.month']
    del dataset_bad['default.payment.next.month']
    
    #print(dataset_good.columns.tolist())
    #print(len(dataset_good.columns.tolist()))
    both_sides = util.merge(dataset_good, dataset_bad, 23)
    
    return dataset, both_sides

def prepare_credit_data(s):
    dataset = pd.read_csv(s)

    del dataset['ID']
    dataset = dataset[:5000]
    
    need_scaling = ["LIMIT_BAL", "AGE", "PAY_0","PAY_2","PAY_3","PAY_4","PAY_5","PAY_6",
         "BILL_AMT1","BILL_AMT2","BILL_AMT3","BILL_AMT4","BILL_AMT5","BILL_AMT6",
         "PAY_AMT1","PAY_AMT2","PAY_AMT3","PAY_AMT4","PAY_AMT5","PAY_AMT6"]
    
    #for column in ['age', 'sibsp', 'parch', 'fare']:
        #imp = Imputer()
        #dataset[column] = imp.fit_transform(dataset[column].values.reshape(-1,1))
    
    scaler = MinMaxScaler()
    for feature in need_scaling:
        dataset[feature] = scaler.fit_transform(dataset[feature].values.reshape(-1,1))
    
    #print(dataset.columns.tolist())
    #print(len(dataset.columns.tolist()))
    
    return dataset