###########################
##Student's name: Patrick##
##Student's  iD: 08470162##
###########################
from sklearn.compose import make_column_transformer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
import streamlit as st 
import pandas as pd 
import numpy as np 
import plotly.express as px 
import seaborn as sns 
import matplotlib.pyplot as plt 
import plotly.graph_objects as go 

#################Build Decision Tree Model##########################
#load dataset
df = pd.read_csv('./week6/winequality-white.csv', sep=";")
print(df.shape)
print(df.head(10))

#Prepare data 
X = df.iloc[:, :-1].values
y= df['quality'].apply(lambda y_value: 1 if y_value>=7 else 0)
#split data to train -- 80% and 20%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state=3)
"""Train Decision Tree"""
from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier()
clf = clf.fit(X_train,y_train)
y_pred = clf.predict(X_test)
print()
result2 = accuracy_score(y_test,y_pred)
print("Accuracy:{} %".format(result2*100))
import pickle 

#save model
pickle_out = open("decisiontreeclassifier.pkl", mode = "wb") 
pickle.dump(clf, pickle_out) 
pickle_out.close()

#print result for confusion matrix and classification report
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
result = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(result)
result1 = classification_report(y_test, y_pred)
print("Classification Report:",)
print (result1)