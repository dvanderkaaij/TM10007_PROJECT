'''
This code is a machine learning pipeline for an Alzheimer's disease database
'''

# %%
# Imports

# from sklearn.model_selection import GridSearchCV
# from missingpy import KNNImputer
# from sklearn import model_selection
from sklearn import preprocessing
from sklearn.svm import SVC
from sklearn.datasets import load_digits
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import learning_curve
from sklearn.model_selection import RandomizedSearchCV
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from adni.load_data import load_data

# %%
# Load data
DATA = load_data()

# %%
# Describe data
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
# Extract information
# Get dataframe X with all features and dataframe Y with labels
X = DATA
X = X.drop(['label'], axis=1)
Y = DATA['label']

# Split dataset --> Trainset(4/5) en Testset(1/5)
CV_5FOLD = StratifiedKFold(n_splits=5)

# Loop over the folds
for train_index, test_index in CV_5FOLD.split(X, Y):
    
    # Split the data properly
    X_TRAIN = X[train_index]
    Y_TRAIN = Y[train_index]

    X_TEST = X[test_index]
    Y_TEST = Y[test_index]

    # X_TRAIN, X_TEST, Y_TRAIN, Y_TEST = train_test_split(X, Y, train_size=0.8, stratify=Y)

    # Binarize labels in testset
    LB = preprocessing.LabelBinarizer()
    Y_TEST = LB.fit_transform(Y_TEST)

    # %% Preprocessing

    # Remove duplicates in X and corresponding Y 
    DUPLICATES = X_TRAIN[X_TRAIN.duplicated(keep='first')]
    DUPLICATES_ID = DUPLICATES.index
    X_TRAIN = X_TRAIN.drop(DUPLICATES_ID)
    Y_TRAIN = Y_TRAIN.drop(DUPLICATES_ID)

    # Binarize labels in trainset
    LB = preprocessing.LabelBinarizer()
    Y_TRAIN = LB.fit_transform(Y_TRAIN)

    # Remove duplicate features
    X_TRAIN = X_TRAIN.T.drop_duplicates().T

    # Remove empty columns
    EMPTY_COLS = X_TRAIN.columns[(X_TRAIN == 0).sum() > 0.8*X_TRAIN.shape[0]]
    X_TRAIN = X_TRAIN.drop(X_TRAIN[EMPTY_COLS], axis=1)

    # Removal of columns with same values
    NUNIQUE = X_TRAIN.apply(pd.Series.nunique)
    SAME_COLS = NUNIQUE[NUNIQUE < 3].index
    X_TRAIN = X_TRAIN.drop(X_TRAIN[SAME_COLS], axis=1)

    # Scaling: Robust range matching
    SCALER = preprocessing.RobustScaler()
    SCALER.fit(X_TRAIN)
    X_TRAIN = SCALER.transform(X_TRAIN)

    # Classifiers
    SCORE = {'accuracy': 'accuracy',
             'roc_auc': 'roc_auc'}
    REFIT =  'roc_auc'

    CV_10 = StratifiedShuffleSplit(n_splits=10, test_size=0.10, train_size=0.90)

    # %% K-Nearest Neighbors (KNN)

    # Create pipeline a pipeline to search for the best
    # hyperparameters for the combination of PCA and Nearest Neighbors
    PIPE_KNN = Pipeline([('pca', PCA()),
                         ('knn', KNeighborsClassifier())])

    # The set of hyperparameters to tune
    PARAMETERS_KNN = {'pca__n_components': [1, 2, 3, 4, 5, 10, 20, 50, 100, 150, 200],
                      'knn__n_neighbors': list(range(1, 99, 2)),
                      'knn__weights': ['uniform', 'distance']}

    CLF_KNN = RandomizedSearchCV(PIPE_KNN, cv=CV_10, n_jobs=-1, n_iter=100,
                                 param_distributions=PARAMETERS_KNN,
                                 scoring=SCORE, refit=REFIT)

    # Train
    CLF_KNN.fit(X_TRAIN, Y_TRAIN)

    # DataFrame of the results with the different hyperparameters
    DF_KNN = pd.DataFrame(CLF_KNN.cv_results_)

    print("NN Best parameters set found on development set:")
    print(CLF_KNN.best_params_)
    print(f'Outcome {REFIT}: {CLF_KNN.best_score_}')

    CLF_KNN_BEST = CLF_KNN.best_estimator_

    # %% Random Forrest (RF)
    # Create pipeline a pipeline to search for the best
    # hyperparameters for the combination of PCA and RandomForestClassifier
    PIPE_RF = Pipeline([('pca', PCA()),
                        ('rf', RandomForestClassifier())])

    # The set of hyperparameters to tune
    PARAMETERS_RF = {'pca__n_components': [1, 2, 3, 4, 5, 10, 20, 50, 100, 150, 200],
                     'rf__n_estimators': list(range(10, 200, 10)),
                     'rf__max_features': ['auto', 'sqrt'],
                     'rf__max_depth': list(range(10, 50, 10)),
                     'rf__min_samples_split': [2, 5, 10],
                     'rf__min_samples_leaf': [1, 2, 4],
                     'rf__bootstrap': [True, False]}

    CLF_RF = RandomizedSearchCV(PIPE_RF, cv=CV_10, n_jobs=-1, n_iter=100,
                                param_distributions=PARAMETERS_RF,
                                scoring=SCORE, refit=REFIT)

    # Train
    CLF_RF.fit(X_TRAIN, Y_TRAIN)

    # DataFrame of the results with the different hyperparameters
    DF_RF = pd.DataFrame(CLF_RF.cv_results_)

    print("RF Best parameters set found on development set:")
    print(CLF_RF.best_params_)
    print(f'Outcome {REFIT}: {CLF_RF.best_score_}')

    CLF_RF_BEST = CLF_RF.best_estimator_
    # The set of hyperparameters to tune
    PARAMETERS_SVM = [{'svc__kernel': ['rbf'], 'svc__gamma': [0.1, 0.01, 0.001, 0.0001],
                    'svc__C': [0.01, 0.1, 0.5, 1, 10, 100], 'svc__max_iter': [1000],
                    'pca__n_components': [1, 2, 3, 4, 5, 10, 20, 50, 100, 150, 200]},
                   {'svc__kernel': ['sigmoid'], 'svc__gamma': [0.1, 0.01, 0.001, 0.0001],
                    'svc__C': [0.01, 0.1, 0.5, 1, 10, 100], 'svc__max_iter': [1000],
                    'pca__n_components': [1, 2, 3, 4, 5, 10, 20, 50, 100, 150, 200]},
                   {'svc__kernel': ['poly'], 'svc__degree': [2, 3, 4, 5],
                    'svc__C': [0.01, 0.1, 0.5, 1, 10, 100], 'svc__max_iter': [1000],
                    'pca__n_components': [1, 2, 3, 4, 5, 10, 20, 50, 100, 150, 200]}]

    # %% Support Vector Machine (SVM)

    # Create pipeline a pipeline to search for the best
    # hyperparameters for the combination of PCA and RandomForestClassifier
    PIPE_SVM = Pipeline([('pca', PCA()),
                        ('svc', SVC())])

    # The set of hyperparameters to tune
    PARAMETERS_SVM = {'pca__n_components': [1, 2, 3, 4, 5, 10, 20, 50, 100, 150, 200],  # Good
                      'svc__C': [0.01, 0.1, 0.5, 1, 10, 100],
                      # 'svc__gamma': [0.1, 0.01, 0.001, 0.0001, 0.00001],
                      # 'svc__gamma': ['scale', 'auto'],
                      'svc__kernel': ['rbf', 'poly', 'sigmoid'],  # Good
                      'svc__degree': [2, 4, 6],
                      'svc__max_iter': [100000]}  # Good

    CLF_SVM = RandomizedSearchCV(PIPE_SVM, cv=CV_10, n_jobs=-1, n_iter=100,
                                 param_distributions=PARAMETERS_SVM,
                                 scoring=SCORE, refit=REFIT)

    # Train
    CLF_SVM.fit(X_TRAIN, Y_TRAIN)

    # DataFrame of the results with the different hyperparameters
    DF_SVM = pd.DataFrame(CLF_SVM.cv_results_)

    print("SVM Best parameters set found on development set:")
    print(CLF_SVM.best_params_)
    print(f'Outcome {REFIT}: {CLF_SVM.best_score_}')

    CLF_SVM_BEST = CLF_SVM.best_estimator_

# %% 
# LEARNING CURVES FOR COMPLEXITY

# function
def plot_learning_curve(estimator, title, X, y, axes=None, ylim=None, cv=None,
                        n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5)):
    """
    Generate 2 plots: the test and training learning curve, 
    the fit times vs score curve.
    """
    if axes is None:
        _, axes = plt.subplots(1, 2, figsize=(10, 15))

    axes[0].set_title(title)
    if ylim is not None:
        axes.set_ylim(*ylim)
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


# plot
X, y = load_digits(return_X_y=True)
CLFS = [CLF_KNN, CLF_RF, CLF_SVM]
TITLE_CLF = ['KNN', 'RF', 'SVM']

fig, axes = plt.subplots(2, 3, figsize=(15, 10))
num = 0
for CLF, TITLE_CLF in zip(CLFS, TITLE_CLF):
    title = f'Learning Curve {TITLE_CLF}'
    plot_learning_curve(CLF, title, X_TRAIN, Y_TRAIN, axes=axes[:, num], ylim=None)
    num += 1
plt.show()
plt.savefig('learning_curves.png')
# %%
