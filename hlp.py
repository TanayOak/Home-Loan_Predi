#The problem statment:
#In this notebook we look at the data we got via the Kaggle dataset.

#Banks want to automate and increase the rate of loan eligibility process 
#based on the customer information and identify the factors/customer segments 
#and who are eligible for taking the loan.

#We will explore the dataset given, check the various features we have and 
#we will make an algorithm that can predict whether or not the loan would 
#be approved in order to automate the process.

# Importing the important libraries / packages
import pandas as pd 			
import numpy as np 					
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

#load the data
data = pd.read_csv("loan_prediction.csv")
print(data)
print(data.shape)
print(data.head())
print(data.tail())

#understanding the data
print(data.isnull().sum())
print(data.select_dtypes("object").nunique())

#handling the null data
new_data=data.fillna({
	"Gender" : "Male",
	"Married" : "Yes",
	"Dependents" : 0,
	"Self Employed" : "No", 
	"LoanAmount": data["LoanAmount"].mean(),
	"Loan_Amount_Term" : data["Loan_Amount_Term"].mean(),
	"Credit_History" : 1
})
print(new_data.isnull().sum())

#handling the catagorical data:
cat_cols = ["Gender", "Married", "Dependents", "Education", "Self_Employed", "Property_Area"]
dummies = pd.get_dummies(data[cat_cols], drop_first=True)

final_data = pd.concat([new_data, dummies], axis="columns")
print(final_data.head())

final_data.drop(cat_cols, axis="columns", inplace=True)
print(final_data.head())
final_data.to_csv("l1.csv")

#features and target
features = final_data.drop(["Loan_ID", "Loan_Status"], axis="columns")
target = final_data["Loan_Status"]

print(features.head())
print(target.head())

# train and test
x_train, x_test, y_train, y_test = train_test_split(features, target, random_state=130)

# model
model=LogisticRegression()
model.fit(features, target)

#classification report
y_pred = model.predict(x_test)
cr = classification_report(y_test, y_pred)
print(cr)

score = model.score(x_test, y_test)
print("score= ", score)

# predict
data = [[4583, 1508, 128, 360, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0]]
data = [[12841, 10968, 34989, 360, 1, 1, 1, 1, 0, 0, 0, 0, 1, 0]]
res = model.predict(data)
print(res)


