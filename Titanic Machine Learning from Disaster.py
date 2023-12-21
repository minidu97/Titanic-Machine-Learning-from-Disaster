import pandas as pd
import os
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import BaggingClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt

#load the training data
train = pd.read_csv('train.csv')
print("First few rows of the training data:")
print(train.head())

#Encoding and filling the missing values
for column in ["Embarked", "Sex"]:
    train[column] = train[column].map({"S": 0, "C": 1, "Q": 2, "male": 0, "female": 1})
for column in ["Embarked", "Sex", "Pclass", "SibSp", "Parch", "Age"]:
    train[column].fillna(train.groupby("Survived")[column].transform("median"), inplace=True)

train["Relatives"] = train["SibSp"] + train["Parch"]
train["Relatives"].fillna(train.groupby("Survived")["Relatives"].transform("median"), inplace=True)

train["Age"] = pd.cut(train["Age"], bins=[0, 16, 60, np.inf], labels=[0, 1, 2], include_lowest=True)
train["Fare"] = (np.log10(train["Fare"]+1)).astype(int)

print("\n\nEncoded and preprocessed training data:")
print(train.head())

#Extracting and standardizing the features
features = train[["Pclass", "Sex", "Age", "Fare", "SibSp", "Parch", "Embarked", "Relatives"]]
target_values = train["Survived"]

scaler = preprocessing.StandardScaler().fit(features)
scaled_features = scaler.transform(features)
feature_table = pd.DataFrame(scaled_features, columns=features.columns)

print("\n\nStandardized features:")
print(feature_table.head())

#scatter plot visualization over each pair
columns = feature_table.columns
n = len(columns)
fig, axes = plt.subplots(n, n, figsize=(15, 15))
for i in range(n):
    for j in range(n):
        if i != j:
            axes[i, j].scatter(feature_table[columns[i]], feature_table[columns[j]], alpha=0.2, c=target_values)
            axes[i, j].set_xlabel(columns[i])
            axes[i, j].set_ylabel(columns[j])
plt.tight_layout()
plt.show()

#SVM Classifier and training
svm = SVC(C=1.0, kernel='rbf', tol=0.001)
svm.fit(feature_table, target_values)
#MLP classifier and training
mlp = MLPClassifier(activation='relu', alpha=1e-05, hidden_layer_sizes=(100, 50, 25), random_state=1, max_iter=5000)
mlp.fit(feature_table, target_values)
#DecisionTree classifier and training
dt = DecisionTreeClassifier(max_depth=None, min_samples_split=2, random_state=0)
dt.fit(feature_table, target_values)

#bagging classifier train to svm
bagging_svm = BaggingClassifier(estimator=svm, n_estimators=10)
bagging_svm.fit(feature_table, target_values)
#bagging classifier train to mlp
bagging_mlp = BaggingClassifier(estimator=mlp, n_estimators=10)
bagging_mlp.fit(feature_table, target_values)
#bagging classifier train to dt
bagging_dt = BaggingClassifier(estimator=dt, n_estimators=10)
bagging_dt.fit(feature_table, target_values)

#define voting classifier and training
voting_clf = VotingClassifier(estimators=[('mlp', mlp),  ('bsvm', bagging_svm), ('bdt', bagging_dt)], voting='hard')
voting_clf.fit(feature_table, target_values)

#Load testing data
test = pd.read_csv('test.csv')

#Encoding and filling the missing values for test data
for column in ["Embarked", "Sex"]:
    test[column] = test[column].fillna(train[column].mode()[0]).map({"S": 0, "C": 1, "Q": 2, "male": 0, "female": 1})
for column in ["Pclass", "SibSp", "Parch"]:
    test[column] = test[column].fillna(train[column].median())
test["Relatives"] = test["SibSp"] + test["Parch"]
#age and fare handled separately because of specific preprocessing steps
test['Age'].fillna(train['Age'].astype(float).median(), inplace=True)
test['Fare'].fillna(train['Fare'].median(), inplace=True)

#Perform transformations similar to training set
test["Age"] = pd.cut(test["Age"], bins=[0, 16, 60, np.inf], labels=[0, 1, 2], include_lowest=True)
test["Fare"] = (np.log10(test["Fare"]+1)).astype(int)

#measure the impact of each classifier
scores = cross_val_score(mlp, feature_table, target_values, cv=5)
print(f'MLPClassifier accuracy: {scores.mean()} (+/- {scores.std() * 2})')#standard deviation

scores = cross_val_score(bagging_svm, feature_table, target_values, cv=5)
print(f'Bagging SVM accuracy: {scores.mean()} (+/- {scores.std() * 2})')

scores = cross_val_score(bagging_dt, feature_table, target_values, cv=5)
print(f'Bagging Decision Tree accuracy: {scores.mean()} (+/- {scores.std() * 2})')

scores = cross_val_score(voting_clf, feature_table, target_values, cv=5)
print(f'Voting Classifier accuracy: {scores.mean()} (+/- {scores.std() * 2})')

#prediction on the test data
test_features = scaler.transform(test[["Pclass","Sex","Age","Fare","SibSp","Parch","Embarked","Relatives"]])
test_features = pd.DataFrame(test_features, columns=feature_table.columns)
predictions = voting_clf.predict(test_features)

#preperation of submission dataframe
submission = pd.DataFrame({'PassengerId': test["PassengerId"], 'Survived': predictions})

#taking the final output as a csv file
try:
    submission.to_csv("output.csv", index=False)
    print("File was created successfully!")
except Exception as e:
    print("An error occurred while trying to save the csv file: ", e)