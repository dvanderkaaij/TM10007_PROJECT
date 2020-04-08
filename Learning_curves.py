# LEARNING CURVES FOR COMPLEXITY

# function
def plot_learning_curve(estimator, title, X, y, axes=None, xlim=None, ylim=None, cv=None,
                        n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5)):
    """
    Generates 1 plot: the test and training learning curve.
    """
    axes.set_title(title)
    if ylim is not None:
        axes.set_ylim(*ylim)
    axes.set_xlabel("Training examples")
    axes.set_ylabel("Score")

    train_sizes, train_scores, test_scores, _ , _ = \
        learning_curve(estimator, X, y, cv=cv, n_jobs=n_jobs,
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

# plot
X, y = load_digits(return_X_y=True)
CLFS = [CLF_KNN, CLF_RF, CLF_SVM]
TITLE_CLF = ['KNN', 'RF', 'SVM']

fig, axes = plt.subplots(1, 3, figsize=(10, 15))
num = 0
for CLF, TITLE_CLF in zip(CLFS, TITLE_CLF):
    title = f'Learning Curve {TITLE_CLF}'
    plot_learning_curve(CLF, title, X_TRAIN, Y_TRAIN, axes=axes[num], xlim=(50, 550), ylim=(0.7, 1.05))
    num += 1
plt.show()
fig.savefig('learning_curves.png')