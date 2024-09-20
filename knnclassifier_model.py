
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
######################Build KNN Classifier model ########################
df = pd.read_csv('./week6/winequality-white.csv', sep=";")
print(df.shape)
# look at the top 10 data
print(df.head(10))

#Prepare data 
X = df.iloc[:, :-1].values
y= df['quality'].apply(lambda y_value: 1 if y_value>=7 else 0)
#split data to train -- 80% and 20%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state=3)

"""'''# 3. Train a KNN classifier'''"""

from sklearn.neighbors import KNeighborsClassifier
# we create a KNN classifier object with K=7
knn = KNeighborsClassifier(n_neighbors=7)
# we can train the knn model with fit() function
knn.fit(X_train, y_train)

# ##save the model
import pickle 
pickle_out = open("knnclassifier.pkl", mode = "wb") 
pickle.dump(knn, pickle_out) 
pickle_out.close()

#make prediction
# y_pred = knn.predict(X_test)

# #print result for confusion matrix, classification model, accuracy score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
result = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(result)
result1 = classification_report(y_test, y_pred)
print("Classification Report:",)
print (result1)
result2 = accuracy_score(y_test,y_pred)
print("Accuracy:",result2)

# # import sklearn metric modules
from sklearn import metrics
# model accuracy, how often is the knn classifier correct?
print("The KNN model accuracy, where k=5 is: {}.".format(metrics.accuracy_score(y_test, y_pred)))