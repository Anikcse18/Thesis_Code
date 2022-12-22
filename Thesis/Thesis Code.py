import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.model_selection import train_test_split
#import libraries
import numpy as np
import sklearn as sk
import pandas as pd
import os
from matplotlib import pyplot as plt
#plt.style.use('default')
import seaborn as sns
%matplotlib inline

#buliding model
import xgboost as xgb
# from lightgbm import LGBMClassifiera
from sklearn.metrics import classification_report , confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import CategoricalNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import cross_val_score
from sklearn.exceptions import FitFailedWarning
import warnings

from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
dataset = pd.read_csv('/Users/istiakjaved/Desktop/Thesis/Dataset/dataset.csv')
dataset.head()
dataset.shape
drop_element = ['ID','R_AB_2_n','R_AB_3_n','NA_R_2_n','NA_R_3_n','NOT_NA_2_n','NOT_NA_3_n']
dataset.drop(drop_element, axis=1, inplace=True)
dataset.shape

#KNN Imputer
impute_it = KNNImputer(n_neighbors=5, weights='uniform', metric='nan_euclidean')

After_imputation = impute_it.fit(dataset)

X_trans = impute_it.transform(dataset)

new_dataset = pd.DataFrame(X_trans)
new_dataset.columns = dataset.columns
new_dataset
new_dataset.isna().sum() # Again checking
new_dataset['REC_IM'] = new_dataset['REC_IM'].apply(np.int64)
new_dataset['REC_IM']

# Target Colomn data Distribution
plt.figure(figsize =(12, 10))
plt.rcParams.update({'font.size': 15})
plt.xticks(rotation=45)
plt.title("Class-wise Data Distribution")
sns.countplot(x=new_dataset['REC_IM'])
plt.savefig("/Users/istiakjaved/Desktop/Thesis/Image/REC_IM class.jpg", dpi=300, bbox_inches='tight')

target_column = new_dataset.REC_IM
Y_main = target_column
Y_main

#Drop target Colume
drop_element = ['REC_IM']
new_dataset.drop(drop_element, axis=1, inplace=True)
feature_data_set = new_dataset
feature_data_set

#Feature Scalling
standardization = StandardScaler()
feature_data_set.head(7)
#Smote
smote = SMOTE(random_state=0)
X_smote,Y_smote = smote.fit_resample(X_main,Y_main)

#Didn't Upload all code here

