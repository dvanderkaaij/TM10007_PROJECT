# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
# Imports
from adni.load_data import load_data


# %%
# Introduction (Eva)
# An introduction concerning the (clinical) problem to be solved.
# 200-300 words


# %%
# Load data
data = load_data()


# %%
# Describe data (Jari)
amount_samples  = len(data.index)
amount_features = len(data.columns)
print(f'The number of samples: {amount_samples}')
print(f'The number of columns: {amount_features}')

amount_AD = sum(data['label']=='AD')
amount_CN = sum(data['label']=='CN')
ratio_AD = amount_AD/amount_samples
ratio_CN = amount_CN/amount_samples
print(f'The number of AD samples: {amount_AD} ({round(ratio_AD*100,2)}%)')
print(f'The number of CN samples: {amount_CN} ({round(ratio_CN*100,2)}%)')


# %%
# Preprocessing (Daniek)
y = data['label']
X = data
X = X.drop(['label'], axis=1)

# 1. Dataset --> Trainset(4/5) en Testset(1/5) verhouding label gelijk houden
from sklearn import model_selection
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=None, stratify=y) 
# checken of classes gelijk verdeeld zijn over train en test set
amount_samples = len(y_train)
amount_AD = sum(y_train=='AD')
amount_CN = sum(y_train=='CN')
ratio_AD = amount_AD/amount_samples

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
# 2. Random Forest


# %%
# Experimental and evaluation setup


# %%
# Statistics


# %%


