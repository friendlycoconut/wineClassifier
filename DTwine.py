import numpy
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import tree
import graphviz
from sklearn.datasets import load_wine
from sklearn.ensemble import RandomForestClassifier

wine_train = pd.read_csv("data/wine_X_train.csv")
wine_test = pd.read_csv("data/wine_X_test.csv")

xTrain = wine_train.drop('quality',axis=1)
yTrain = wine_train['quality']
xTest = wine_test

clfTre = tree.DecisionTreeClassifier(max_depth=7)
clfTre.fit(xTrain, yTrain)
clfTre = clfTre.predict(xTest)

print("DT (ID3) classifier results: \n", clfTre)

print ("-------------------------")

rfc = RandomForestClassifier(n_estimators=200)
rfc.fit(xTrain, yTrain)
pred_rfc = rfc.predict(xTest)
print("Random forest classifier results: \n", pred_rfc)

pred_rfc = np.where(pred_rfc >= 6, pred_rfc, 0 )
pred_rfc = np.where(pred_rfc == 0, pred_rfc, 1 )
print(pred_rfc)


numpy.savetxt("data/wine_X_result.csv", pred_rfc , delimiter=",", fmt='%d')

