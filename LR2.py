#Resource: https://stackabuse.com/understanding-roc-curves-with-python/

import matplotlib.pyplot as plt
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


def plot_roc_curve(fpr, tpr):
    plt.plot(fpr, tpr, color='orange', label='ROC')
    plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend()
    plt.show()


xdata = pd.read_csv("x_data.csv")
print(xdata)
ydata = pd.read_csv("y_data.csv")
XTEST = pd.read_csv("XTEST.csv")

model = LogisticRegression() #l2

scores = cross_val_score(model,xdata,ydata,cv=5,scoring='roc_auc')
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

X = xdata.iloc[:, 1:]
Z = XTEST.iloc[:, 1:]

results = []

for i in range(1,len(ydata.columns)):
    Y = ydata.iloc[:, 1] #this should be i, then loop through everything

#Split the data into 80% training and 20% testing

    #x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)

#Logistic Regression
    #model = LogisticRegression() #l2
    #model = LogisticRegression(penalty='none')
    #model = LogisticRegression(solver='saga',penalty='l1') #l1
    #model = LogisticRegression(solver='saga',l1_ratio=0.5,penalty='elasticnet') #elasticnet (different l1 ratios)

#Random Forest Classifier
    #model = RandomForestClassifier(n_estimators=100, criterion='gini')

#kNN
    #model = KNeighborsClassifier(n_neighbors=100)
    model.fit(X, Y) #Training the model

#Test the model

#predictions = model.predict(x_test)
#print(predictions)

#Check precision, recall, f1-score

#print(classification_report(y_test, predictions) )
#print(accuracy_score(y_test, predictions))

    probs = model.predict_proba(Z)
    probs = probs[:, 1]
    results.append(probs)

flippedProbs = []
for i in results[0]:
    flippedProbs.append([i])
for row in results[1:]:
    for i in range(0,len(row)):
        flippedProbs[i].append(row[i])

df = pd.DataFrame(flippedProbs)
df.to_csv('Elastic.csv')

#auc = roc_auc_score(y_test, probs)
#print('AUC: %.2f' % auc)
#fpr, tpr, thresholds = roc_curve(y_test, probs)
#plot_roc_curve(fpr, tpr)