
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
from sklearn import preprocessing
from sklearn.decomposition import PCA
from missingpy import KNNImputer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_digits
from sklearn.model_selection import learning_curve

# Load data
DATA_original = load_data()
DATA = load_data()

X = DATA
X = X.drop(['label'], axis=1)
Y = DATA['label']

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size=0.8, stratify=Y)
lb = preprocessing.LabelBinarizer()
Y_test = lb.fit_transform(Y_test)

duplicate = X_train[X_train.duplicated(keep='first')]
duplicate_id = duplicate.index
X_train = X_train.drop(duplicate_id)
Y_train = Y_train.drop(duplicate_id) 

lb = preprocessing.LabelBinarizer()
Y_train = lb.fit_transform(Y_train)

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




from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline

#eigenlijk hiervoor nog PCA, maar dat moet dan voor splitten train-val
pipe = Pipeline([('pca', PCA()),
    ('rf', RandomForestClassifier())])
score = {'f1': 'f1', 'accuracy': 'accuracy'}

random_grid_rf = {'pca__n_components': [10, 20, 30, 40, 50],
               'rf__n_estimators': list(range(10,200,10)),
               'rf__max_features': ['auto', 'sqrt'],
               'rf__max_depth': list(range(10,50,10)),
               'rf__min_samples_split': [2, 5, 10],
               'rf__min_samples_leaf': [1, 2, 4],
               'rf__bootstrap': [True, False]}     

rf_random_pca = RandomizedSearchCV(pipe, cv=3, n_jobs=-1, n_iter = 100, param_distributions=random_grid_rf, scoring=score, refit='accuracy')

rf_random_pca.fit(X_train, Y_train)

df_results_rf_pca = pd.DataFrame(rf_random_pca.cv_results_)


print("Best parameters set found on development set:")
print(rf_random_pca.best_params_)


# %%
print(pipeline.get_params().keys())

# %% 
# Parameters of pipelines can be set using ‘__’ separated parameter names:

clf_rf = RandomForestClassifier()
rf_random = RandomizedSearchCV(estimator = clf_rf, param_distributions = random_grid_rf, n_iter = 100, cv = 3, verbose=2, random_state=42, n_jobs = -1)
# wat is n_jobs en random_state?

rf_random.fit(X_train, Y_train)

df_results_rf_pca = pd.DataFrame(rf_random.cv_results_)
Y_pred_train      = rf_random.predict(X_train)

print("Best parameters set found on development set:")
print(rf_random.best_params_)

# %%
# %% RANDOM FOREST
# Cross validation 10 Fold
# belangrijkste hyperparameters n_components (trees) and max_features (the numer of features
# considered for splitting at eacht leaf node)
# max_depth = max number of levels in each decision tree
# min_samples_split 
# min_samples_leaf
# bootstrap - method for sampling datapoints (with or without replacement)
# Aantal trees bepalen aan de hand van een grafiek --> nog goeie van maken

import matplotlib.pyplot as plt
import numpy as np
import seaborn

kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
auc_train_values = []
auc_val_values = []
auc_n_trees = []

for train_index, validation_index in kfold.split(X_train, Y_train):
    X_train_cv, X_validation_cv = X_train[train_index], X_train[validation_index]
    Y_train_cv, Y_validation_cv = Y_train[train_index], Y_train[validation_index]

    # Scaling: Robust range matching
    scaler = preprocessing.RobustScaler()

    scaler.fit(X_train_cv)
    X_train_scaled      = scaler.transform(X_train_cv)
    X_validation_scaled = scaler.transform(X_validation_cv)

    n_features = 50 # Meerder mogelijkheden, zoen nog optimaal aantal (hyperparameter)
    p = PCA(n_components=n_features)
    p = p.fit(X_train_cv)
    x = p.transform(X_train_cv)

    n_trees = [1, 5, 10, 20, 50, 100] # Moeten we beperken om overtraining te voorkomen
    #moeten we nog iets met random_state?
    # Wat is die criterion?
    
    for index, tree in enumerate(n_trees): 
        clf_rf = RandomForestClassifier(n_estimators=tree)
        
        clf_rf.fit(X_train_cv, Y_train_cv)
        Y_pred_train      = clf_rf.predict(X_train_cv)
        Y_pred_validation = clf_rf.predict(X_validation_cv)
       
        # Metric
        auc_train = metrics.roc_auc_score(Y_train_cv, Y_pred_train)
        auc_val = metrics.roc_auc_score(Y_validation_cv, Y_pred_validation)
        print(tree)
        print(auc_val)
        auc_train_values.append(auc_train) 
        auc_n_trees.append(tree)
        auc_val_values.append(auc_val)

plt.plot(auc_n_trees, auc_train_values)
plt.plot(auc_n_trees, auc_val_values) 
plt.show()

seaborn.scatterplot(x=auc_n_trees, y=auc_train_values)
seaborn.scatterplot(x=auc_n_trees, y=auc_val_values)

