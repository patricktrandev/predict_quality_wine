###########################
##Student's name: Patrick##
##Student's  iD: 08470162##
###########################
import streamlit as st 
import pandas as pd 
import numpy as np 
import plotly.express as px 
import seaborn as sns 
import matplotlib.pyplot as plt 
import plotly.graph_objects as go 

from sklearn.compose import make_column_transformer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split

# 1. let's create an APP title
st.title("White Wine Quality Forecast - Machine Learning project")
# let's load the dataset into a dataframe with pandas
df = pd.read_csv('./winequality-white.csv', sep=";")
feature_cols= ['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar',
       'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density',
       'pH', 'sulphates', 'alcohol', 'quality']
preprocessor= make_column_transformer((StandardScaler(),feature_cols))
# look at the top 10 data


print(df.head(10))

#streamlit run assigment2.py
# # let's create an APP checkbox
#show dataframe
if st.checkbox('Show Dataframe'):
  st.write(df)
#show statistics
print(df.describe())
if st.checkbox('Show statistics'):
  st.write(df.describe())
### Draw heatmap
res_heatmap= st.button("Show Heatmap")
if res_heatmap:
  st.subheader("Heatmap of Dataset")
  fig_ht, ax = plt.subplots()
  sns.heatmap(df.corr(),color = "honeydew", ax=ax)
  st.write(fig_ht)

#######Prediction of Wine Quality
st.header("White Wine Quality Prediction")
col1,col2,col3=st.columns(3)

fixed_acidity=np.float(col1.text_input("Fixed Acidity ",7.2))
volatile_acidity=np.float(col2.text_input("Volatile Acidity",0.32))
citric_acid=np.float(col3.text_input("Citric acid",0.36))
residual_sugar=np.float(col1.text_input("Residual sugar ",2.0))
chlorides=np.double(col2.text_input("Chlorides",0.033))
free_sulfur_dioxide=np.float(col3.text_input("Free sulfur dioxide",37.0))
total_sulfur_dioxide=np.float(col1.text_input("Total sulfur dioxide",114.0))
density=np.double(col2.text_input("Density ",0.9906))
pH=np.float(col3.text_input("pH ",3.1))
sulphates=np.float(col1.text_input("Sulphates",0.71))
alcohol=np.float(col2.text_input("Alcohol",12.3))
result= st.button("Predict White Wine Quality")

##load the model - Random Forest Classification
import pickle 
pickle_in = open('classifier.pkl', 'rb') 
classifier = pickle.load(pickle_in)
@st.cache()
#create function to get result by training model
def prediction(fixed_acidity,volatile_acidity,citric_acid,residual_sugar,chlorides,free_sulfur_dioxide,total_sulfur_dioxide,density,pH,sulphates,alcohol):   
  input_data=(fixed_acidity,volatile_acidity,citric_acid,residual_sugar,chlorides,free_sulfur_dioxide,total_sulfur_dioxide,density,pH,sulphates,alcohol)
  #convert dataset to array numpy
  input_data_asnumpy= np.asarray(input_data)
  input_data_reshaped= input_data_asnumpy.reshape(1,-1)
  prediction= classifier.predict(input_data_reshaped)
  if prediction == 1:
    pred="Good quality Wine"
  else:
    pred="Bad quality Wine"
  return pred
#output result
if result:
  res_prediction=prediction(fixed_acidity,volatile_acidity,citric_acid,residual_sugar,chlorides,free_sulfur_dioxide,total_sulfur_dioxide,density,pH,sulphates,alcohol)
  st.success("The predicted result is "+res_prediction)

#############################################
#visualization the dataset with chart
st.header("Visualization")
col1 = df.columns[11:]
col2 = st.selectbox("Which feature on the Y?", df.columns[:11])
print(df.columns[0:11])
#define columns to display button
col_1, col_2, col_3 = st.columns([1,1,1])
with col_1:
  res_bar= st.button('Bar chart')
with col_2:
  res_boxplot= st.button('Boxplot')
with col_3:
  res_dotplot= st.button('Dotplot')

# Draw barchart from user's input
if res_bar:
  st.subheader("Bar chart ")
# we can draw a figure inside the APP with plotlty express
  fig = px.bar(df, x='quality', y=col2)
  fig.update_traces(marker_color='rgb(248, 22, 73)', marker_line_color='rgb(181, 41, 73)',
                  marker_line_width=1.5)
  st.plotly_chart(fig)
## Draw boxplot from user's input
if res_boxplot:
  st.subheader("Boxplot ")
  fig1 = px.box(df, x='quality', y=col2)
  fig1.update_traces(marker_color='tomato', marker_line_color='limegreen',
                  marker_line_width=1.5)
  st.plotly_chart(fig1)

## Draw dot plot from user's input
if res_dotplot:  
  st.subheader("Dot plot ")
  fig2 = px.scatter(df, x=col2, y='quality')
  fig2.update_traces(marker_color='salmon', marker_line_color='midnightblue',
                  marker_line_width=1.5)
  st.plotly_chart(fig2)

#############################################################
###########MODEL EVALUATION###############################

st.header("Model Evaluation")
#define function to get X_train, y_train, X_test,y_test
@st.cache(persist = True)
def split(df):
  X = df.iloc[:, :-1].values
  y= df['quality'].apply(lambda y_value: 1 if y_value>=7 else 0)
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state=3)
  return X_train, X_test, y_train, y_test

##get split data
X_train, X_test, y_train, y_test= split(df)
#define function to get accuracy score , confusion matrix, classification report with randomForest
def getEvaluationModel_randomForest(X_train, X_test, y_train, y_test):
  pickle_in = open('classifier.pkl', 'rb') 
  classifier = pickle.load(pickle_in)
  X_test_prediction= classifier.predict(X_test)
  test_data_accuracy=accuracy_score(X_test_prediction,y_test)
  #st.write("Accuracy score : ",test_data_accuracy)
  result = confusion_matrix(y_test, X_test_prediction)
  #st.write("Confusion matrix : ",result)
  result2 = classification_report(y_test, X_test_prediction)
  #st.write("Classification report :")
  #st.write(result2)
  return test_data_accuracy,result,result2

test_data_accuracy1,confusion1,class_report1= getEvaluationModel_randomForest(X_train, X_test, y_train, y_test)
#define function to get accuracy score , confusion matrix, classification report with KNN classifier
def getEvaluationModel_KnnClassifier(X_train, X_test, y_train, y_test):
  pickle_in = open('knnclassifier.pkl', 'rb') 
  classifier = pickle.load(pickle_in)
  X_test_prediction= classifier.predict(X_test)
  test_data_accuracy=accuracy_score(X_test_prediction,y_test)
  #st.write("Accuracy score : ",test_data_accuracy)
  result = confusion_matrix(y_test, X_test_prediction)
  #st.write("Confusion matrix : ",result)
  result2 = classification_report(y_test, X_test_prediction)
  #st.write("Classification report :")
  #st.write(result2)
  return test_data_accuracy,result,result2

test_data_accuracy2,confusion2,class_report2= getEvaluationModel_KnnClassifier(X_train, X_test, y_train, y_test)
#define function to get accuracy score , confusion matrix, classification report with Decision Tree
def getEvaluationModel_DecisionTreeClassifier(X_train, X_test, y_train, y_test):
  pickle_in = open('decisiontreeclassifier.pkl', 'rb') 
  classifier = pickle.load(pickle_in)
  X_test_prediction= classifier.predict(X_test)
  test_data_accuracy=accuracy_score(X_test_prediction,y_test)
  result = confusion_matrix(y_test, X_test_prediction)
  result2 = classification_report(y_test, X_test_prediction)
  return test_data_accuracy,result,result2

test_data_accuracy3,confusion3,class_report3= getEvaluationModel_DecisionTreeClassifier(X_train, X_test, y_train, y_test)

types_model=['Random Forest Classification','Decision Tree Classification','KNN Classifier']
col_select = st.selectbox("Which algorithms?", types_model)
#display button to get results
coll_1, coll_2, coll_3 = st.columns([1,1,1])
with coll_1:
  accuracy_sscore= st.button('Accuracy Score')
with coll_2:
  confusion_mmatrix= st.button('Confusion Matrix')
with coll_3:
  classification_rreport= st.button('Classification Report')

#####declare accuracy score, confusion matrix, classification report by their following algorithms
arr_score=0
if col_select == 'Random Forest Classification':  
  if accuracy_sscore:
    st.info("Accuracy score : ")
    st.write(test_data_accuracy1)
  if confusion_mmatrix:
    st.info("Confusion matrix of the model: ")
    st.write(confusion1)
  if classification_rreport:
    st.info("Classification report of the model: ")
    st.write(class_report1)
elif col_select == 'KNN Classifier':
  if accuracy_sscore:
    st.info("Accuracy score : ")
    st.write("Accuracy score : ",test_data_accuracy2)
  if confusion_mmatrix:
    st.info("Confusion matrix of the model: ")
    st.write(confusion2)
  if classification_rreport:
    st.info("Classification report of the model: ")
    st.write(class_report2)
elif col_select == 'Decision Tree Classification':
  if accuracy_sscore:
    st.info("Accuracy score : ")
    st.write(test_data_accuracy3)
  if confusion_mmatrix:
    st.info("Confusion matrix of the model: ")
    st.write(confusion3)
  if classification_rreport:
    st.info("Classification report of the model: ")
    st.write(class_report3)





