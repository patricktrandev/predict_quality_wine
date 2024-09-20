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

#####################build random forest model##############
#load dataset
df = pd.read_csv('./week6/winequality-white.csv', sep=";")
#display dataset
print(df.shape)
print(df.head(10))

#Prepare data 
X = df.iloc[:, :-1].values
y= df['quality'].apply(lambda y_value: 1 if y_value>=7 else 0)
print(X)
print()
print(y)
#split data to train -- 80% and 20%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state=3)

#Train Random Forest Classifier
model_rfc= RandomForestClassifier()
model_rfc.fit(X_train, y_train)

# #load model
# pickle_in = open('classifier.pkl', 'rb') 
# classifier = pickle.load(pickle_in)
# X_test_prediction= classifier.predict(X_test)
# test_data_accuracy=accuracy_score(X_test_prediction,y_test)

#Get accuracy score
X_test_prediction= model_rfc.predict(X_test)
test_data_accuracy=accuracy_score(X_test_prediction,y_test)
print("Accuracy score {} ".format(test_data_accuracy))


#########Test data
input_data=(7.2,0.32,0.36,2,0.033,37,114,0.9906,3.1,0.71,12.3)
#change data to numpy array
input_data_asnumpy= np.asarray(input_data)
input_data_reshaped= input_data_asnumpy.reshape(1,-1)
prediction= model_rfc.predict(input_data_reshaped)
print(prediction)
if(prediction[0]==1):
  print("Good quality Wine")
else:
  print("Bad quality Wine")

# ##save the model
import pickle 
pickle_out = open("classifier.pkl", mode = "wb") 
pickle.dump(model_rfc, pickle_out) 
pickle_out.close()