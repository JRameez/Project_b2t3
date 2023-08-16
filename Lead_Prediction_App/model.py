import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import make_scorer, precision_score, f1_score
import pickle

file_id = '1nCbwq2PktId7IlJmlR5kGn7vvpgiE8vW'
file_url = f'https://drive.google.com/uc?id={file_id}'
data = pd.read_csv(file_url)

# Assuming 'data' is the DataFrame that contains both  lead_1_data and lead_0_data

# Fill missing values in 'Credit_Product' column with 'Yes' where Lead is 1
data.loc[data['Is_Lead']==1,'Credit_Product']=data.loc[data['Is_Lead'] == 1, 'Credit_Product'].fillna('Yes')

# Fill missing values in 'Credit_Product' column with 'No' where Lead is 0
data.loc[data['Is_Lead']==0,'Credit_Product']=data.loc[data['Is_Lead'] == 0, 'Credit_Product'].fillna('No')


data=data.drop('ID',axis=1)


data['Region_Code'] = data['Region_Code'].str.replace('RG', '').astype(int)

from sklearn.preprocessing import OneHotEncoder

# Select the categorical columns to encode
categorical_columns = ['Occupation', 'Channel_Code']

# Create a OneHotEncoder instance
encoder = OneHotEncoder(sparse_output=True)

# Fit and transform the encoder on the selected categorical columns
encoder.fit(data[categorical_columns])


encoded_column_names = ['Entrepreneur', 'Other', 'Salaried', 'Self_Employed','X1', 'X2', 'X3', 'X4']

encoded_columns = pd.DataFrame(encoder.transform(data[categorical_columns]).todense(), columns=encoded_column_names)

new_df = pd.concat([data, encoded_columns], axis=1, ignore_index=True)
new_df.columns = data.columns.to_list() + encoded_column_names
new_df.drop(columns=['Occupation', 'Channel_Code'], inplace=True)


new_df['Gender'] = ((new_df['Gender'] == 'Female')).astype(int)
new_df['Is_Active'] = ((new_df['Is_Active'] == 'Yes')).astype(int)
new_df['Credit_Product'] = ((new_df['Credit_Product'] == 'Yes')).astype(int)


#transformation technique
#log transformation:

# Log transformation on the "avg_account_balance" column
new_df["Avg_Account_Balance"] = new_df["Avg_Account_Balance"].map(lambda i: np.log(i) if i > 0 else 0)


x=new_df.drop('Is_Lead',axis=1)
y=new_df['Is_Lead']

from sklearn.preprocessing import MinMaxScaler
minmax = MinMaxScaler(feature_range=(0,1))

minmax.fit(x.iloc[:, [1,2,3]])

x.iloc[:, [1,2,3]] = minmax.transform(x.iloc[:, [1,2,3]])


from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import make_scorer, precision_score, f1_score
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import GradientBoostingClassifier


# Create a SMOTE instance
smote = SMOTE(random_state=42)

# Initialize StratifiedKFold for stratified cross-validation with 5 folds
stratified_kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

gb_params = {'subsample': 1.0,
             'random_state': None,
             'n_estimators': 140,
             'min_samples_split': 10,
             'min_samples_leaf': 4,
             'max_features': None,
             'max_depth': 3,
             'learning_rate': 0.2
             }

# Initialize lists to store precision and F1 scores
precision_scores_gb = []
f1_scores_gb = []

# Perform stratified cross-validation
for train_index, test_index in stratified_kfold.split(x, y):
    x_train, x_test = x.iloc[train_index], x.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    # Apply SMOTE to the training data
    x_train_smote, y_train_smote = smote.fit_resample(x_train, y_train)

    # Initialize the Gradient Boosting classifier
    gb_classifier = GradientBoostingClassifier(**gb_params)

    # Fit the classifier on the SMOTE-augmented training data
    gb_classifier.fit(x_train_smote, y_train_smote)

    # Predict on the test data
    y_pred = gb_classifier.predict(x_test)

    # Calculate precision and F1 scores
    precision = precision_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    precision_scores_gb.append(precision)
    f1_scores_gb.append(f1)

# Print cross-validation precision scores for Gradient Boosting
print("Cross-Validation Precision Scores (Gradient Boosting):", precision_scores_gb)
print("Mean Precision (Gradient Boosting):", sum(precision_scores_gb) / len(precision_scores_gb))
print("Standard Deviation (Precision) (Gradient Boosting):", np.std(precision_scores_gb))

# Print cross-validation F1 scores for Gradient Boosting
print("Cross-Validation F1 Scores (Gradient Boosting):", f1_scores_gb)
print("Mean F1 Score (Gradient Boosting):", sum(f1_scores_gb) / len(f1_scores_gb))
print("Standard Deviation (F1) (Gradient Boosting):", np.std(f1_scores_gb))






pickle.dump(gb_classifier, open('gb_model.pickle','wb'))
pickle.dump(encoder, open('encoder.pickle','wb'))
pickle.dump(minmax, open('minmax.pickle','wb'))

