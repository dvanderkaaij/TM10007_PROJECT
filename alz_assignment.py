'''
Code for loading the data and split into train and test set.
'''

# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
# Imports
from sklearn import model_selection
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from adni.load_data import load_data
from sklearn import metrics
from sklearn import preprocessing


import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.decomposition import PCA
from missingpy import KNNImputer
from sklearn.neighbors import KNeighborsClassifier

# %%
# Introduction (Eva)
# An introduction concerning the (clinical) problem to be solved.
# 200-300 words

# %%
# Load data
DATA_original = load_data()
DATA = load_data()

# %%
# Describe data (Jari)
AMOUNT_SAMPLES = len(DATA.index)
AMOUNT_FEATURES = len(DATA.columns)
print(f'The number of samples: {AMOUNT_SAMPLES}')
print(f'The number of columns: {AMOUNT_FEATURES}')

AMOUNT_AD = sum(DATA['label'] == 'AD')
AMOUNT_CN = sum(DATA['label'] == 'CN')
RATIO_AD = AMOUNT_AD/AMOUNT_SAMPLES
RATIO_CN = AMOUNT_CN/AMOUNT_SAMPLES
print(f'The number of AD samples: {AMOUNT_AD} ({round(RATIO_AD*100,2)}%)')
print(f'The number of CN samples: {AMOUNT_CN} ({round(RATIO_CN*100,2)}%)')

# %% 
# Extract labels
# Get dataframe X with all features and dataframe Y with labels
X = DATA
X = X.drop(['label'], axis=1)
Y = DATA['label']
# replaced the binarizing 

# Split dataset --> Trainset(4/5) en Testset(1/5)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size=0.8, random_state=None, stratify=Y)
lb = preprocessing.LabelBinarizer()
Y_test = lb.fit_transform(Y_test)

# %% Preprocessing

# Removal of duplicates (Daniek)

# remove 1 sample
X_train = X_train.drop_duplicates()
# remove same sample from labels
duplicate = X_train[X_train.duplicated(keep='first')]
duplicate_id = duplicate.index
Y_train = Y_train.drop(duplicate_id) 

lb = preprocessing.LabelBinarizer()
Y_train = lb.fit_transform(Y_train)

# remove 18 features
X_train = X_train.T.drop_duplicates().T 

# Removal of empty columns (Eva)
empty_cols = X_train.columns[(X_train == 0).sum() > 0.8*X_train.shape[0]]
print(X_train.columns)
print(f'empty: {empty_cols}') # missing: vf_Frangi_edge_energy_SR(1.0, 10.0)_SS2.0
X_train = X_train.drop(X_train[empty_cols], axis=1) 

# Removal of columns with same values
nunique = X_train.apply(pd.Series.nunique)
same_cols = nunique[nunique < 3].index
print(same_cols)
X_train = X_train.drop(X_train[same_cols], axis=1) # 4 colums removed

# %%
# Cross validation 10 Fold
# Trainset --> Trainset(4/5) en Validatieset(1/5) voor cross-validatie
X_train = X_train.to_numpy()
#Y_train = Y_train.to_numpy()

all_X_train_cv = []
all_X_validation_cv = []
all_Y_train_cv = []
all_Y_validation_cv = []

sss = model_selection.StratifiedShuffleSplit(n_splits=10, train_size=0.8, random_state=42)
for train_index, validation_index in sss.split(X_train, Y_train):
    X_train_cv, X_validation_cv = X_train[train_index], X_train[validation_index]
    Y_train_cv, Y_validation_cv = Y_train[train_index], Y_train[validation_index]

    # imputation of missing values with K-nn
    # imputer = KNNImputer(missing_values=0, n_neighbors=5, weights="uniform")
    # DATA_imputed = imputer.fit_transform(DATA)

    # Scaling: Robust range matching
    scaler = preprocessing.RobustScaler()
    # we hebben zoveel data dat het niet mogelijk is
    # om elke feature te plotten en te kijken of er outliers zijn. Daarom gaan we ervan uit
    # dat er outliers zijn en gebruiken we de RobustScaler()
    scaler.fit(X_train_cv)
    X_train_scaled      = scaler.transform(X_train_cv)
    X_validation_scaled = scaler.transform(X_validation_cv)


    all_X_train_cv.append(X_train_scaled)
    all_X_validation_cv.append(X_validation_scaled)
    all_Y_train_cv.append(Y_train_cv)
    all_Y_validation_cv.append(Y_validation_cv)


    # PCA (Jari)
    n_features = 50 # Meerder mogelijkheden, zoen nog optimaal aantal (hyperparameter)
    p = PCA(n_components=n_features)
    p = p.fit(X_train_cv)
    x = p.transform(X_train_cv)

    # K-NN (Jari)
    clf = KNeighborsClassifier(n_neighbors=5)
    clf.fit(X_train_cv, Y_train_cv)
    Y_pred_train      = clf.predict(X_train_cv)
    Y_pred_validation = clf.predict(X_validation_cv)
    
    # Metric
    auc = metrics.roc_auc_score(Y_train_cv, Y_pred_train)
    print(auc)
    auc2 = metrics.roc_auc_score(Y_validation_cv, Y_pred_validation)
    print(auc2)

# %%
# Classifiers

# 1. Support Vector Machine 

# Parameters:
# C -> regularization parameter, strength is inversely proportional to C
# kernel -> linear, poly rbf, sigmoid or precomputed
# degree -> integer
# coef0 -> independent term? Only significant in poly and sigmoid

clf = SVC(kernel='rbf', degree=1, coef0=0.5, C=0.5)
clf.fit(X_train_cv, y_train_cv)
y_pred = clf.predict(X_train_cv)

# 2. Random Forest (Eva)

# Parameters:
# n_estimators -> number of trees
# bootstrap -> False or True, determines if samples are drawn with replacement or not
# class_weight -> if not given, all classes have weight one
# max_features -> when looking at a split

n_trees = [1, 5, 10, 50, 100] # Moeten we beperken om overtraining te voorkomen
clf_rf = RandomForestClassifier(n_estimators=n_trees, criterion='gini') # Wati is die criterion?
clf_rf.fit(X_train_cv, y_train_cv)
y_pred_rf = clf_rf.predict(X_train_cv)

# Bootstrapping True/False?
# Bij voorkeur geen class weigh, denk ik?
# Zeker wel feature importance (ranglijst met belangrijkste features)
importances = clf_rf.feature_importances_
std = np.std([tree.feature_importances_ for tree in forest.estimators_],
             axis=0)
indices = np.argsort(importances)[::-1]

# 3. K-Nearest Neighbour



# %%
# Experimental and evaluation setup
# Beschrijven wat we doen en waarom

# %%
# Statistics
# Accuracy, AUC, F1score, Precision, Recall
   # auc=metrics.roc_auc_score(Y1, y_score)
   # accuracy=metrics.accuracy_score(Y1, y_pred)
   # F1=metrics.f1_score(Y1,y_pred)
   # precision=metrics.precision_score(Y1,y_pred)
   # recall=metrics.recall_score(Y1, y_pred)

   # print(type(clf))
   # print('Acc:' +str(accuracy))
   # print('AUC:' +str(auc))
   # print('F1:' +str(F1))
   # print('precision:' +str(precision))
   # print('recall:' +str(recall))

# %%
