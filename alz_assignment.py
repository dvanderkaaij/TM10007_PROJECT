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
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from adni.load_data import load_data
from sklearn import metrics
from sklearn import preprocessing
from sklearn.model_selection import StratifiedKFold
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from missingpy import KNNImputer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_digits
from sklearn.model_selection import learning_curve
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline

# %%
# Load data
DATA_original = load_data()
DATA = load_data()

# %%
# Describe data
AMOUNT_SAMPLES = len(DATA_original.index)
AMOUNT_FEATURES = len(DATA_original.columns)
print(f'The number of samples: {AMOUNT_SAMPLES}')
print(f'The number of columns: {AMOUNT_FEATURES}')

AMOUNT_AD = sum(DATA_original['label'] == 'AD')
AMOUNT_CN = sum(DATA_original['label'] == 'CN')
RATIO_AD = AMOUNT_AD/AMOUNT_SAMPLES
RATIO_CN = AMOUNT_CN/AMOUNT_SAMPLES
print(f'The number of AD samples: {AMOUNT_AD} ({round(RATIO_AD*100,2)}%)')
print(f'The number of CN samples: {AMOUNT_CN} ({round(RATIO_CN*100,2)}%)')

# %%
# Extract information
# Get dataframe X with all features and dataframe Y with labels
X = DATA
X = X.drop(['label'], axis=1)
Y = DATA['label']

# Split dataset --> Trainset(4/5) en Testset(1/5)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size=0.8, stratify=Y)

# Binarize labels in testset
LB = preprocessing.LabelBinarizer()
Y_test = LB.fit_transform(Y_test)

# %% Preprocessing

# Removal of duplicates in X and corresponding Y (1 sample)
DUPLICATES = X_train[X_train.duplicated(keep='first')]
DUPLICATES_ID = DUPLICATES.index
X_train = X_train.drop(DUPLICATES_ID)
Y_train = Y_train.drop(DUPLICATES_ID)

# Binarize labels in trainset
LB = preprocessing.LabelBinarizer()
Y_train = LB.fit_transform(Y_train)

# remove 18 features
X_train = X_train.T.drop_duplicates().T 

# Removal of empty columns
empty_cols = X_train.columns[(X_train == 0).sum() > 0.8*X_train.shape[0]]
print(X_train.columns)
print(f'empty: {empty_cols}') # missing: vf_Frangi_edge_energy_SR(1.0, 10.0)_SS2.0
X_train = X_train.drop(X_train[empty_cols], axis=1) 

# Removal of columns with same values
nunique = X_train.apply(pd.Series.nunique)
same_cols = nunique[nunique < 3].index
print(same_cols)
X_train = X_train.drop(X_train[same_cols], axis=1) # 4 colums removed

# Scaling: Robust range matching
scaler = preprocessing.RobustScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)

# %% KNN

# Initialise lists
best_n_neighbors = []
AUC = []

# Amount of principal components (0 is without PCA)
components_list = [1, 2, 3, 4, 5, 10, 20, 50, 100, 0]

for components in components_list:
    print(f'PCA with {components} components')

    x = X_train
    if components > 0:
        p = PCA(n_components=components)
        p = p.fit(X_train)
        x = p.transform(X_train)

    knn = KNeighborsClassifier()
    parameters = {'n_neighbors': list(range(1, 99, 2)),
                  'weights' : ['uniform', 'distance']}

    cv_10fold = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

    grid_search = model_selection.GridSearchCV(knn, parameters, cv=cv_10fold, scoring='roc_auc')
    grid_search.fit(x, Y_train)

    clf_knn = grid_search.best_estimator_
    best_n_neighbors.append(clf_knn.n_neighbors)

    DF = pd.DataFrame(grid_search.cv_results_)
    AUC.append(max(DF['mean_test_score']))

    print("Best parameters set found on development set:")
    print(grid_search.best_params_)
    print(f'Amount of neighbors: {best_n_neighbors[-1]} with AUC: {AUC[-1]}')

# %% RANDOM FOREST
# Create pipeline a pipeline to search for the best
# hyperparameters for the combination of PCA and RandomForestClassifier

pipe_rf = Pipeline([('pca', PCA()),
    ('rf', RandomForestClassifier())])
score = {'accuracy': 'accuracy'}

# The set of hyperparameters to tune
parameters_rf = {'pca__n_components': [10, 50, 100, 150, 200],
               'rf__n_estimators': list(range(10, 200, 10)),
               'rf__max_features': ['auto', 'sqrt'],
               'rf__max_depth': list(range(10, 50,10)),
               'rf__min_samples_split': [2, 5, 10],
               'rf__min_samples_leaf': [1, 2, 4],
               'rf__bootstrap': [True, False]}     

clf_rf_pca = RandomizedSearchCV(pipe_rf, cv=3, n_jobs=-1, n_iter = 100, param_distributions=parameters_rf, scoring=score, refit='accuracy')

# Train RandomForest classifier
clf_rf_pca.fit(X_train, Y_train)

# DataFrame of the results with the different hyperparameters
df_results_rf_pca = pd.DataFrame(clf_rf_pca.cv_results_)

print("Best parameters set found on development set:")
print(clf_rf_pca.best_params_)

# %% 
# LEARNING CURVE, COMPLEXITY

# function
def plot_learning_curve(estimator, title, X, y, axes=None, ylim=None, cv=None,
                        n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5)):
    """
    Generate 2 plots: the test and training learning curve, 
    the fit times vs score curve.
    """
    if axes is None:
        _, axes = plt.subplots(2, 1, figsize=(10, 15))

    axes[0].set_title(title)
    if ylim is not None:
        axes[0].set_ylim(*ylim)
    axes[0].set_xlabel("Training examples")
    axes[0].set_ylabel("Score")

    train_sizes, train_scores, test_scores, fit_times, _ = \
        learning_curve(estimator, X, y, cv=cv, n_jobs=n_jobs,
                       train_sizes=train_sizes,
                       return_times=True)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    fit_times_mean = np.mean(fit_times, axis=1)
    fit_times_std = np.std(fit_times, axis=1)

    # Plot learning curve
    axes[0].grid()
    axes[0].fill_between(train_sizes, train_scores_mean - train_scores_std,
                         train_scores_mean + train_scores_std, alpha=0.1,
                         color="r")
    axes[0].fill_between(train_sizes, test_scores_mean - test_scores_std,
                         test_scores_mean + test_scores_std, alpha=0.1,
                         color="g")
    axes[0].plot(train_sizes, train_scores_mean, 'o-', color="r",
                 label="Training score")
    axes[0].plot(train_sizes, test_scores_mean, 'o-', color="g",
                 label="Cross-validation score")
    axes[0].legend(loc="best")

    # Plot fit_time vs score
    axes[1].grid()
    axes[1].plot(fit_times_mean, test_scores_mean, 'o-')
    axes[1].fill_between(fit_times_mean, test_scores_mean - test_scores_std,
                         test_scores_mean + test_scores_std, alpha=0.1)
    axes[1].set_xlabel("fit_times")
    axes[1].set_ylabel("Score")
    axes[1].set_title("Performance of the model")

    return plt

# plot learning curve 
X, y = load_digits(return_X_y=True)

title = "Learning Curves (KNN)"

estimator = clf
# can be changed:
# - clf_rf -> RandomForestClassifier(n_estimators=5, random_state=42)
# RandomForest performs much worse -> training size too small -> KNN better option

plot_learning_curve(estimator, title, X_train_cv, Y_train_cv, axes=None, ylim=None)

plt.show()


# %%
# Classifiers

# 1. Support Vector Machine 

# Parameters:
# C -> regularization parameter, strength is inversely proportional to C
# kernel -> linear, poly rbf, sigmoid or precomputed
# degree -> integer
# coef0 -> independent term? Only significant in poly and sigmoid

pipe_svc = Pipeline([('pca', PCA()),
    ('svc', SVC())])
score = {'accuracy': 'accuracy'}

hyperparameters = {'pca__n_components': [0, 1, 2, 4, 10, 50, 100, 150, 200],
                   'svc__C': [0.01, 0.1, 0.5, 1, 10, 100], 
                   'svc__gamma': [ 0.1, 0.01, 0.001, 0.0001, 0.00001], 
                   'svc__kernel': ['rbf', 'poly', 'sigmoid'],
                   'svc__degree': [2,4,6],
                   'svc__max_iter': [100000]}

clf_svc_pca = RandomizedSearchCV(pipe_svc, cv=10, n_jobs=-1, n_iter= 100, param_distributions=hyperparameters, scoring=score, refit='accuracy')

clf_svc_pca.fit(X_train, Y_train)

# DataFrame of the results with the different hyperparameters
df_results_svc_pca = pd.DataFrame(clf_svc_pca.cv_results_)

print("Best parameters set found on development set:")
parameters_svc = clf_svc_pca.best_params_
print(parameters_svc)
print('accuracy:')
print(clf_svc_pca.best_score_)

# %%
