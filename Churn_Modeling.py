# -*- coding: utf-8 -*-
"""
Created on Thu Jun 27 11:14:12 2019

@author: uzer
"""
"""
what's the challenge here?Well, the bank has been seeing unusual churn rates.
So churn is when people leave the company and they've seen customers leaving at
unusually high rates and they want to understand what the problem is; they want
to assess and address that problem.our goal is to create a demographic centric 
model to tell the bank which of the customers are at highest risk of leaving.
"""
#for training purposes I used different machine Learning methods


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset=pd.read_csv("Churn_Modelling.csv")
X=dataset.iloc[:,3:13].values
y=dataset.iloc[:,13].values

#Data Preprocessing
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
X[:, 1]=LabelEncoder().fit_transform(X[:, 1])
X[:, 2]=LabelEncoder().fit_transform(X[:, 2])
X=OneHotEncoder(categorical_features = [1]).fit_transform(X).toarray()
X=X[:,1:]

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)

from sklearn.preprocessing import StandardScaler
ss=StandardScaler()
X_train=ss.fit_transform(X_train)
X_test=ss.transform(X_test)



#LogisticRegression(LR)
from sklearn.linear_model import LogisticRegression
classifier_LR=LogisticRegression()
classifier_LR.fit(X_train,y_train)

y_pred_LR=classifier_LR.predict(X_test)

from sklearn.metrics import confusion_matrix,accuracy_score
CM_LR=confusion_matrix(y_test,y_pred_LR)
AcS_LR=accuracy_score(y_test,y_pred_LR)




#SVM 
from sklearn.svm import SVC
#SVM with poly kernel
classifier_SVM_poly=SVC(kernel="poly",random_state=0)
classifier_SVM_poly.fit(X_train,y_train)

y_pred_poly=classifier_SVM_poly.predict(X_test)

from sklearn.metrics import confusion_matrix,accuracy_score
CM_SVM_poly=confusion_matrix(y_test,y_pred_poly)
AcS_SVM_poly=accuracy_score(y_test,y_pred_poly)

#SVM with rbf kernel
classifier_SVM_rbf=SVC(kernel="rbf",random_state=0)
classifier_SVM_rbf.fit(X_train,y_train)

y_pred_rbf=classifier_SVM_rbf.predict(X_test)

from sklearn.metrics import confusion_matrix,accuracy_score
CM_SVM_rbf=confusion_matrix(y_test,y_pred_rbf)
AcS_SVM_rbf=accuracy_score(y_test,y_pred_rbf)




#K_NN
from sklearn.neighbors import KNeighborsClassifier
classifier_KNN=KNeighborsClassifier(n_neighbors=14,weights="distance")

classifier_KNN.fit(X_train,y_train)

y_pred_KNN=classifier_KNN.predict(X_test)

from sklearn.metrics import confusion_matrix,accuracy_score
CM_KNN=confusion_matrix(y_test,y_pred_KNN)
AcS_KNN=accuracy_score(y_test,y_pred_KNN)




#RandomForestClassifier
from sklearn.ensemble import RandomForestClassifier
classifier_RF=RandomForestClassifier(n_estimators=60,random_state=0)
classifier_RF.fit(X_train,y_train)

y_pred_RF=classifier_RF.predict(X_test)

from sklearn.metrics import confusion_matrix,accuracy_score
CM_RF=confusion_matrix(y_test,y_pred_RF)
AcS_RF=accuracy_score(y_test,y_pred_RF)




#Navis Bayes
from sklearn.naive_bayes import GaussianNB
classifier_NB=GaussianNB()
classifier_NB.fit(X_train,y_train)

y_pred_NB=classifier_NB.predict(X_test)

from sklearn.metrics import confusion_matrix,accuracy_score
CM_NB=confusion_matrix(y_test,y_pred_NB)
AcS_NB=accuracy_score(y_test,y_pred_NB)




#ANN
import keras
from keras.models import Sequential
from keras.layers import Dense
classifier=Sequential()
# Adding the input layer and two hidden layers
classifier_ANN.add(Dense(units=6,kernel_initializer="uniform",activation="relu",input_dim=11))
classifier_ANN.add(Dense(units=6,kernel_initializer="uniform",activation="relu"))

# Adding the output layer
classifier_ANN.add(Dense(units=1,kernel_initializer="uniform",activation="sigmoid"))

classifier_ANN.compile(optimizer="rmsprop" , loss="binary_crossentropy", metrics=["accuracy"])
classifier_ANN.fit(X_train,y_train,batch_size=10,epochs=100)


y_pred_ANN=classifier_ANN.predict(X_test)
y_pred_ANN=y_pred_ANN>0.5

from sklearn.metrics import confusion_matrix
CM_ANN=confusion_matrix(y_pred_ANN,y_test)
AcS_ANN=accuracy_score(y_test,y_pred_ANN)



# Improving the ANN
# Dropout Regularization to reduce overfitting if needed
# Tuning the ANN
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV 
from keras.models import Sequential
from keras.layers import Dense

def classifier(optimizer):
    classifier_ANN=Sequential()
    classifier_ANN.add(Dense(units=6,kernel_initializer="uniform",activation="relu",input_dim=11))
    classifier_ANN.add(Dense(units=6,kernel_initializer="uniform",activation="relu"))
    classifier_ANN.add(Dense(units=1,kernel_initializer="uniform",activation="sigmoid"))
    classifier_ANN.compile(optimizer=optimizer , loss="binary_crossentropy", metrics=["accuracy"])
    return classifier_ANN   

classifier_ANN=KerasClassifier(build_fn=classifier)
parameters={'batch_size':[25,27],'epochs':[200,500],'optimizer':['adam','rmsprop']}
grid_search=GridSearchCV(estimator=classifier_ANN,param_grid=parameters,cv=10,scoring = 'accuracy')

grid_search.fit(X_train,y_train)
best_parameters = grid_search.best_params_
best_accuracy = grid_search.best_score_


