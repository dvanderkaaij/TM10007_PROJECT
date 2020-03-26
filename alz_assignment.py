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
from adni.load_data import load_data

# %%
# Introduction (Eva)
# An introduction concerning the (clinical) problem to be solved.
# 200-300 words

# %%
# Load data
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
# Preprocessing (Daniek)

# get df X with all features and df y with labels
X = DATA
X = X.drop(['label'], axis=1)
y = DATA['label']

# 1. Split dataset --> Trainset(4/5) en Testset(1/5)
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=None, stratify=y)

# check if percentage of samples for each class is preserved
samples = len(y_train)
label1 = sum(y_train == 'AD')
proportion = label1/samples
print(f'Proportion: {round(proportion*100,2)}%')
print(f'The number of AD samples: {AMOUNT_AD} ({round(RATIO_AD*100,2)}%)')

# 2. Trainset --> Trainset(4/5) en Validatieset(1/5) voor cross-validatie

X_train = X_train.to_numpy()
y_train = y_train.to_numpy()

sss = model_selection.StratifiedShuffleSplit(n_splits=10, train_size=0.8, random_state=None)
for train_index, validation_index in sss.split(X_train, y_train):
    X_train_cv, X_validation = X_train[train_index], X_train[validation_index]
    y_train_cv, y_validation = y_train[train_index], y_train[validation_index]

# %%
# Classifiers

# 1. Support Vector Machine

# Parameters
# C -> regularization parameter, strength is inversely proportional to C
# kernel -> linear, poly rbf, sigmoid or precomputed
# degree -> integer
# coef0 -> independent term? Only significant in poly and sigmoid. 

clf = SVC(kernel='rbf', degree=1, coef0=0.5, C=0.5)
clf.fit(X_train_cv,y_train_cv)
y_pred = clf.predict(x_train_cv)

# 2. Random Forest


# %%
# Experimental and evaluation setup


# %%
# Statistics


# %%
