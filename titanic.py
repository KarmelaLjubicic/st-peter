import pandas as pd
from math import *
import util
from sklearn.preprocessing import MinMaxScaler, Imputer

def get_all():
    dataset = prepare_titanic_data("data\\Titanic.csv")
    
    dataset_good = dataset[dataset['survived'] == 1]
    dataset_bad = dataset[dataset['survived'] == 0]
    del dataset_good['survived']
    del dataset_bad['survived']

    both_sides = util.merge(dataset_good, dataset_bad, 9)
    #print(both_sides.columns.tolist())
    
    return dataset, both_sides

def prepare_titanic_data(s):
    dataset = pd.read_csv(s)

    #del dataset['PassengerId']
    del dataset['name']
    del dataset['cabin']
    del dataset['ticket']
    del dataset['home.dest']
    del dataset['body']
    del dataset['boat']
    
    for column in ['age', 'sibsp', 'parch', 'fare']:
        imp = Imputer()
        dataset[column] = imp.fit_transform(dataset[column].values.reshape(-1,1))
    
    dataset['sibsp'] =  dataset['sibsp'].astype(float)
    dataset = pd.get_dummies(dataset, columns=["sex", "embarked"], prefix=["sex", "embarked"])
    
    scaler = MinMaxScaler()
    dataset['age'] = scaler.fit_transform(dataset['age'].values.reshape(-1,1))
    dataset['fare'] = scaler.fit_transform(dataset['fare'].values.reshape(-1,1))
    
    #print(dataset.columns.tolist())
    
    return dataset