'''
This code is a machine learning pipeline for an Alzheimer's disease database
'''

# %%
# Imports

from sklearn import preprocessing
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import learning_curve
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from adni.load_data import load_data

# %%
# Function for plotting learning curves


def plot_learning_curve(estimator, title, X, y, axes=None,
                        n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5)):
    """
    Generates 1 plot: the test and training learning curve.
    """
    axes.set_title(title)
    # if ylim is not None:
    #    axes.set_ylim(*ylim)
    axes.set_xlabel("Training examples")
    axes.set_ylabel("Score")

    train_sizes, train_scores, test_scores, _, _ = \
        learning_curve(estimator, X, y, cv=StratifiedKFold(n_splits=10), n_jobs=n_jobs,
                       train_sizes=train_sizes,
                       return_times=True)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    # Plot learning curve
    axes.grid()
    axes.fill_between(train_sizes, train_scores_mean - train_scores_std,
                      train_scores_mean + train_scores_std, alpha=0.1,
                      color="r")
    axes.fill_between(train_sizes, test_scores_mean - test_scores_std,
                      test_scores_mean + test_scores_std, alpha=0.1,
                      color="g")
    axes.plot(train_sizes, train_scores_mean, 'o-', color="r",
              label="Training score")
    axes.plot(train_sizes, test_scores_mean, 'o-', color="g",
              label="Cross-validation score")
    axes.legend(loc="best")

    return plt


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

# Function for outer cross validation
CV_10FOLD = StratifiedKFold(n_splits=10)

# Lists for AUC scores
AUC_KNN = []
AUC_RF = []
AUC_SVM = []

# Lists for sensitivity scores
SENS_KNN = []
SENS_RF = []
SENS_SVM = []

# Lists for specificity scores
SPEC_KNN = []
SPEC_RF = []
SPEC_SVM = []

# lists for different hyperparameter sets
PAR_KNN = []
PAR_RF = []
PAR_SVM = []

# Loop over the folds
for train_index, test_index in CV_10FOLD.split(X, Y):
    # Split the data
    X_TRAIN = X.iloc[train_index]
    Y_TRAIN = Y.iloc[train_index]

    X_TEST = X.iloc[test_index]
    Y_TEST = Y.iloc[test_index]

    # PREPROCESSING

    # Remove duplicates in X and corresponding Y
    DUPLICATES = X_TRAIN[X_TRAIN.duplicated(keep='first')]
    DUPLICATES_ID = DUPLICATES.index
    X_TRAIN = X_TRAIN.drop(DUPLICATES_ID)
    Y_TRAIN = Y_TRAIN.drop(DUPLICATES_ID)

    DUPLICATES_TEST = X_TEST[X_TEST.duplicated(keep='first')]
    DUPLICATES_ID_TEST = DUPLICATES_TEST.index
    X_TEST = X_TEST.drop(DUPLICATES_ID_TEST)
    Y_TEST = Y_TEST.drop(DUPLICATES_ID_TEST)

    # Binarize labels
    LB = preprocessing.LabelBinarizer()
    Y_TRAIN = LB.fit_transform(Y_TRAIN)
    Y_TEST = LB.fit_transform(Y_TEST)

    # Remove duplicate features
    X_TRAIN = X_TRAIN.T.drop_duplicates().T
    X_TEST = X_TEST.T.drop_duplicates().T

    # Remove empty columns
    EMPTY_COLS = X_TRAIN.columns[(X_TRAIN == 0).sum() > 0.8*X_TRAIN.shape[0]]
    X_TRAIN = X_TRAIN.drop(X_TRAIN[EMPTY_COLS], axis=1)

    EMPTY_COLS_TEST = X_TEST.columns[(X_TEST == 0).sum() > 0.8*X_TEST.shape[0]]
    X_TEST = X_TEST.drop(X_TEST[EMPTY_COLS_TEST], axis=1)

    # Removal of columns with same values
    NUNIQUE = X_TRAIN.apply(pd.Series.nunique)
    SAME_COLS = NUNIQUE[NUNIQUE < 2].index
    X_TRAIN = X_TRAIN.drop(X_TRAIN[SAME_COLS], axis=1)

    NUNIQUE_TEST = X_TEST.apply(pd.Series.nunique)
    SAME_COLS_TEST = NUNIQUE_TEST[NUNIQUE < 2].index
    X_TEST = X_TEST.drop(X_TEST[SAME_COLS_TEST], axis=1)

    # Scaling: Robust range matching
    SCALER = preprocessing.RobustScaler()
    SCALER.fit(X_TRAIN)
    X_TRAIN = SCALER.transform(X_TRAIN)
    X_TEST = SCALER.transform(X_TEST)

    # CLASSIFIERS
    SCORE = {'accuracy': 'accuracy',
             'roc_auc': 'roc_auc'}
    REFIT = 'roc_auc'

    # Function for outer cross validation
    CV_5 = StratifiedShuffleSplit(n_splits=5, test_size=0.10, train_size=0.90)

    # K-Nearest Neighbors (KNN)

    # Search for best hyperparameters for the combination of PCA and Nearest Neighbors
    PIPE_KNN = Pipeline([('pca', PCA()),
                         ('knn', KNeighborsClassifier())])

    # The set of hyperparameters to tune
    PARAMETERS_KNN = {'pca__n_components': [1, 2, 3, 4, 5, 10, 20, 50, 100, 150, 200],
                      'knn__n_neighbors': list(range(1, 99, 2)),
                      'knn__weights': ['uniform', 'distance']}

    CLF_KNN = RandomizedSearchCV(PIPE_KNN, cv=CV_5, n_jobs=-1, n_iter=100,
                                 param_distributions=PARAMETERS_KNN,
                                 scoring=SCORE, refit=REFIT)

    # Train
    CLF_KNN.fit(X_TRAIN, Y_TRAIN)

    # DataFrame of the results with the different hyperparameters
    DF_KNN = pd.DataFrame(CLF_KNN.cv_results_)

    CLF_KNN_BEST = CLF_KNN.best_estimator_
    PAR_KNN.append(CLF_KNN.best_params_)

    # Random Forest (RF)
    # Search for best hyperparameters for the combination of PCA and RandomForestClassifier
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

    CLF_RF = RandomizedSearchCV(PIPE_RF, cv=CV_5, n_jobs=-1, n_iter=100,
                                param_distributions=PARAMETERS_RF,
                                scoring=SCORE, refit=REFIT)

    # Train
    CLF_RF.fit(X_TRAIN, Y_TRAIN)

    # DataFrame of the results with the different hyperparameters
    DF_RF = pd.DataFrame(CLF_RF.cv_results_)

    CLF_RF_BEST = CLF_RF.best_estimator_
    PAR_SVM.append(CLF_RF.best_params_)

    # Support Vector Machine (SVM)

    # Search for best hyperparameters for the combination of PCA and SVM
    PIPE_SVM = Pipeline([('pca', PCA()),
                         ('svc', SVC())])

    # The set of hyperparameters to tune
    PARAMETERS_SVM = [{'svc__kernel': ['rbf'], 'svc__gamma': [1, 0.1, 0.01, 0.001, 0.0001],
                       'svc__C': [0.01, 0.1, 0.5, 1, 10, 50, 100], 'svc__max_iter': [1000],
                       'pca__n_components': [1, 2, 3, 4, 5, 10, 20, 50, 100, 150, 200]},
                      {'svc__kernel': ['sigmoid'], 'svc__gamma': [1, 0.1, 0.01, 0.001, 0.0001],
                       'svc__C': [0.01, 0.1, 0.5, 1, 10, 50, 100], 'svc__max_iter': [1000], 
                       'svc__coef0' : list(np.arange(0,1,0.1)),
                       'pca__n_components': [1, 2, 3, 4, 5, 10, 20, 50, 100, 150, 200]},
                      {'svc__kernel': ['poly'], 'svc__degree': [2, 3, 4, 5],
                       'svc__C': [0.01, 0.1, 0.5, 1, 10, 50, 100], 'svc__max_iter': [1000],
                       'svc__coef0' : list(np.arange(0,1,0.1)),
                       'pca__n_components': [1, 2, 3, 4, 5, 10, 20, 50, 100, 150, 200]}]

    CLF_SVM = RandomizedSearchCV(PIPE_SVM, cv=CV_5, n_jobs=-1, n_iter=100,
                                 param_distributions=PARAMETERS_SVM,
                                 scoring=SCORE, refit=REFIT)

    # Train
    CLF_SVM.fit(X_TRAIN, Y_TRAIN)

    # DataFrame of the results with the different hyperparameters
    DF_SVM = pd.DataFrame(CLF_SVM.cv_results_)

    CLF_SVM_BEST = CLF_SVM.best_estimator_
    PAR_RF.append(CLF_RF.best_params_)

    # Train with outer cross-validation
    CLF_KNN_BEST.fit(X_TRAIN, Y_TRAIN)
    CLF_RF_BEST.fit(X_TRAIN, Y_TRAIN)
    CLF_SVM_BEST.fit(X_TRAIN, Y_TRAIN)

    # get predictions
    KNN_prediction = CLF_KNN_BEST.predict(X_TEST)
    RF_prediction = CLF_RF_BEST.predict(X_TEST)
    SVM_prediction = CLF_SVM_BEST.predict(X_TEST)

    AUC_KNN.append(roc_auc_score(Y_TEST, KNN_prediction))
    AUC_RF.append(roc_auc_score(Y_TEST, RF_prediction))
    AUC_SVM.append(roc_auc_score(Y_TEST, SVM_prediction))

    TN_KNN, FP_KNN, FN_KNN, TP_KNN = confusion_matrix(Y_TEST, KNN_prediction).ravel()
    SPEC_KNN.append(TN_KNN / (TN_KNN+FP_KNN))
    SENS_KNN.append(TP_KNN / (TP_KNN+FN_KNN))

    TN_RF, FP_RF, FN_RF, TP_RF = confusion_matrix(Y_TEST, RF_prediction).ravel()
    SPEC_RF.append(TN_RF / (TN_RF+FP_RF))
    SENS_RF.append(TP_RF / (TP_RF+FN_RF))

    TN_SVM, FP_SVM, FN_SVM, TP_SVM = confusion_matrix(Y_TEST, SVM_prediction).ravel()
    SPEC_SVM.append(TN_SVM / (TN_SVM+FP_SVM))
    SENS_SVM.append(TP_SVM / (TP_SVM+FN_SVM))

# %%
# learning curves

# PREPROCESSING X AND Y

# Remove duplicates in X and corresponding Y
DUPLICATES = X[X.duplicated(keep='first')]
DUPLICATES_ID = DUPLICATES.index
X = X.drop(DUPLICATES_ID)
Y = Y.drop(DUPLICATES_ID)

# Binarize labels
LB = preprocessing.LabelBinarizer()
Y = LB.fit_transform(Y)

# Remove duplicate features
X = X.T.drop_duplicates().T

# Remove empty columns
EMPTY_COLS = X.columns[(X == 0).sum() > 0.8*X.shape[0]]
X = X.drop(X[EMPTY_COLS], axis=1)

# Removal of columns with same values
NUNIQUE = X.apply(pd.Series.nunique)
SAME_COLS = NUNIQUE[NUNIQUE < 2].index
X = X.drop(X[SAME_COLS], axis=1)

# Scaling: Robust range matching
SCALER = preprocessing.RobustScaler()
SCALER.fit(X)
X = SCALER.transform(X)

# %%
# plot learning curves
CLFS = [CLF_KNN_BEST, CLF_RF_BEST, CLF_SVM_BEST]
TITLE_CLF = ['KNN', 'RF', 'SVM']

fig, AXES = plt.subplots(1, 3, figsize=(20, 10))
NUM = 0
for CLF, TITLE_CLF in zip(CLFS, TITLE_CLF):
    title = f'Learning Curve {TITLE_CLF}'
    plot_learning_curve(CLF, title, X, Y, axes=AXES[NUM])
    NUM += 1
fig.savefig(f'learning_curves 3 classifiers.png')
plt.show()
# %%
# get results

# Create df with AUC-values of the tree classifiers over the 10-fold cross validation
DF_AUC_RESULTS = pd.DataFrame({'KNN': list(AUC_KNN),
                               'RF': list(AUC_RF),
                               'SVM': list(AUC_SVM)})
DF_AUC_RESULTS.loc['mean'] = DF_AUC_RESULTS.mean()
DF_AUC_RESULTS['best_clf'] = DF_AUC_RESULTS.idxmax(axis=1)
DF_AUC_RESULTS.loc['std'] = DF_AUC_RESULTS.std()
print(DF_AUC_RESULTS)
DF_AUC_RESULTS.to_csv('df_AUC_results.csv')

# Create df with specificities
DF_SPEC_RESULTS = pd.DataFrame({'KNN': list(SPEC_KNN),
                                'RF': list(SPEC_RF),
                                'SVM': list(SPEC_SVM)})
DF_SPEC_RESULTS.loc['mean'] = DF_SPEC_RESULTS.mean()
DF_SPEC_RESULTS.loc['std'] = DF_SPEC_RESULTS.std()
print(DF_SPEC_RESULTS)
DF_SPEC_RESULTS.to_csv('df_SPEC_results.csv')

# Create df with sensitivities
DF_SENS_RESULTS = pd.DataFrame({'KNN': list(SENS_KNN),
                                'RF': list(SENS_RF),
                                'SVM': list(SENS_SVM)})
DF_SENS_RESULTS.loc['mean'] = DF_SENS_RESULTS.mean()
DF_SENS_RESULTS.loc['std'] = DF_SENS_RESULTS.std()
print(DF_SENS_RESULTS)
DF_SENS_RESULTS.to_csv('df_SENS_results.csv')
# %%
