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

#Splitting the data using sklearn
from sklearn.model_selection import train_test_split
Y_ccl=df_ccl["college_type"]
X_ccl=df_ccl.drop(["college_type"],axis=1)
X_train_ccl,X_test_ccl,Y_train_ccl,Y_test_ccl=train_test_split(X_ccl,Y_ccl,test_size=0.2,stratify=Y_ccl,random_state=42)

#Using Column_Transformer and Label encoding for target variable using Pre_Processing Package
from Pre_Processing import Transformers
trf=Transformers()
X_train_ccl,Y_train_ccl=trf.fit_transform(X_train_ccl,Y_train_ccl)
X_test_ccl,Y_test_ccl=trf.transform(X_test_ccl,Y_test_ccl)
