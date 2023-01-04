import numpy as np
# A = np.zeros((2,3))
# a = ['1','2','3']
# A[0] = list(map(float, a))
# print(A)
# print(A[0][1:])
from collections import defaultdict
adj_lists = defaultdict(set)
# print(adj_lists)
adj_lists["11"].add("22")
adj_lists["12"].add("32")
adj_lists["12"].add("39")
# print(adj_lists)
# print(adj_lists)
import random
# print(random.sample)
# print(np.random.permutation(10))
# class A():
#     def forwa(x):
#         print(x+1)
# a = A()
# # print(a(3))
# from sklearn.metrics import f1_score,accuracy_score
# a = np.array([[1,0,1,1],[0,1,0,1]])
# #true
# b = np.array([[0,0,1,1],[1,1,0,1]]) #predict
# # print(f1_score(a,b,average='binary'))
# print(f1_score(a,b,average='micro'))
# print(f1_score(a,b,average='macro'))
# print(accuracy_score(a,b))
import random
# print(random.sample)
from sklearn.datasets import load_iris
# from sklearn.linear_model import LogisticRegression
# X, y = load_iris(return_X_y=True)
# print(y)
# clf = LogisticRegression(random_state=0, solver='lbfgs',multi_class='ovr').fit(X, y)
# print(clf.predict_proba(X[:200]))
from sklearn import model_selection
# from sklearn import datasets, linear_model
# from sklearn.model_selection import cross_val_predict
# diabetes = datasets.load_diabetes()
# X, y = load_iris(return_X_y=True)
# lasso = linear_model.Lasso()
# y_pred = cross_val_predict(lasso, X, y, cv=3)
# print(y_pred)
# print(y)
test = [i for i in range(2708) if i>=2000]
print(test)
a = [i for i in range(2708)]
# print(a[2000:])
a = [1,2,3]
b = []
for i in a:
    print(i)
    b.append(1)