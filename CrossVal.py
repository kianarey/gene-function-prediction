#Resource: https://stackabuse.com/understanding-roc-curves-with-python/

#import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

model = LogisticRegression(solver='saga',l1_ratio=0.5,penalty='elasticnet')

xdata = pd.read_csv("x_data.csv")
ydata = pd.read_csv("y_data.csv")
XTEST = pd.read_csv("XTEST.csv")

print(xdata.iloc[:,1:])
print("---------")
print(ydata.iloc[:,1:])

allScores = []
count = 1
for i in range(1,len(ydata.columns)):
    scores = cross_val_score(model,xdata.iloc[:,1:],ydata.iloc[:,i],cv=5,scoring='roc_auc')
    print(scores)
    print(count)
    count +=1
    for i in scores:
        allScores.append(i)

total = 0
for i in allScores:
    total += i

mean = total / len(allScores)
print(mean)
