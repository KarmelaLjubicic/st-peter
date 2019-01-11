import pandas as pd
from math import *
import util

def get_all():
    dataset = pd.read_csv("data\Students_Performance.csv")
    dataset = pd.get_dummies(dataset, 
                   columns=["gender", "race/ethnicity", "parental level of education", "lunch", "test preparation course"],
                   prefix=["gender", "race/ethnicity", "parental", "lunch", "preparation"])
    #dataset['final_score'] = dataset['math score'] + dataset['reading score'] + dataset['writing score']
    #dataset['final_score'] = dataset.final_score.map(lambda x: ceil(x))
    #dataset['pass'] = dataset.final_score.map(lambda x: 1 if x/3 > 70 else 0)
    #dataset['index'] = range(0, len(dataset))

    #del dataset['final_score']
    #del dataset['math score']
    del dataset['reading score']
    del dataset['writing score']

    #dataset_good = dataset[dataset['pass'] == 1]
    #dataset_bad = dataset[dataset['pass'] == 0]
    dataset_good = dataset[dataset['math score'] > 80]
    dataset_bad = dataset[dataset['math score'] <= 80]
    #del dataset_good['math score']
    #del dataset_bad['math score']
    
    print(dataset_good.columns.tolist())
    print(len(dataset_good.columns.tolist()))
    both_sides = util.merge(dataset_good, dataset_bad, 18)

    return dataset, both_sides
