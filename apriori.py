import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv('Market_Basket_Optimisation.csv', header=None)
transaction = []
for i in range(0,7501):
    transaction.append([str(dataset.values[i,j]) for j in range(0, 20)])

from apyori import apriori
rules = apriori(transactions = transaction, 
min_support = 0.003, min_confidence = 0.20, min_lift = 3, min_length = 2, max_length = 2)

results = list(rules)

def inspect(results):
    lhs = [tuple(result [2][0][0])[0] for result in results]
    rhs = [tuple(result [2][0][1])[0] for result in results]
    supports = [result[1] for result in results]
    confidences = [result[2][0][2] for result in results]
    lift = [result[2][0][3] for result in results]
    return list(zip(lhs, rhs, supports, confidences, lift))

resultinDataframe = pd.DataFrame(inspect(results), columns = ['Left Hand side','Right Hand side','Support','Confidence','Lift'])
print(resultinDataframe.nlargest(n = 10, columns='Lift'))
print('done')