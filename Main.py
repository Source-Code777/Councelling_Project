#LOADING THE DATA USING PANDAS WEB SCRAPPER
from Data_Files import Data
import pandas as pd
url_23 = "https://admissions.nic.in/wbjeeb/Applicant/report/orcrreport.aspx?enc=b6w3EPyuw0C4FADZ4v1XmYUz0XFq314fzLjkE3wbM2xr/DbsjpvUS9LBCKXjSeSL"
url_24 = "https://admissions.nic.in/wbjeeb/Applicant/report/orcrreport.aspx?enc=Nm7QwHILXclJQSv2YVS+7l8OpFY/O746kfneOXEneV50mv1B/txHsSKB11hFlsvw"

data = Data(url_23, url_24)
df = data.load_data()

#Performing Data_Cleaning With the help of Helper_Data Package
from Helper_data import Data_Mapping
pre = Data_Mapping()
df= pre.preprocess_dataframe(df)

#Adding College_type column using Pre_Processing Package
from Pre_Processing import ManualClassification
from Feature_Engineering import ManualClassification

feature=ManualClassification()
df_ccl = feature.classify_dataframe(df, "College_Name")
#Dropping Branch column as it is redundant and College_Name
#As we don't want the model to memorize college name
df_ccl=df_ccl.drop(["Branch","College_Name"],axis=1)

#Let's train the college_classifier_model Logistic_Regression now
from Models import Logistic_Model_class
lm=Logistic_Model_class()
Y_ccl=df_ccl["college_type"]
X_ccl=df_ccl.drop(["college_type"],axis=1)
lm.train(X_ccl,Y_ccl)

#Predict
y_pred_lm=lm.predict()
#Evaluate
lm.evaluate()
#The Model works well

#Lets train the College_Classifier Tree Model and repeat the steps
from Models import DecisionTree_Model

tree_clf=DecisionTree_Model()
tree_clf.train(X_ccl,Y_ccl)
y_pred_tree=tree_clf.predict()
tree_clf.predict()
metrics=tree_clf.evaluate()
#I have modified the print_metrics function so i need to pass metrics parameter now
tree_clf.print_metrics(metrics)


