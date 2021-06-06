from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import LinearSVC, SVC
from sklearn.ensemble import AdaBoostClassifier, BaggingClassifier, GradientBoostingClassifier, RandomForestClassifier,\
    StackingClassifier, VotingClassifier
from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier
import numpy as np
# import plotly.graph_objects as go
from plotly.subplots import make_subplots
import dill
from numpy.random import default_rng

rng = default_rng(42)

results_file = "/mnt/c/Users/jbortiz/GoogleRoot/School/Clemson/Thesis/Submissions/Journal_May2021/code/Ansys Data/" \
    "ANSYS_Results_2021Mar11.csv"
X = np.loadtxt(results_file, skiprows=1, delimiter=',')

# Cols: 0-Wall, 1-Width, 2-Height, 3-Floor, 4-Length, 5-Dfrmtn, 6-Mass

# Classifying based on deformation, mass ignored
n, d = X[:, :-2].shape

y = X[:, -2] < 0.001
X = X[:, :-2]

split = 0.8
X_train, X_test = X[:int(split*n), :], X[int(split*n):, :]
y_train, y_test = y[:int(split*n)], y[int(split*n):]


X_mins = X_train.min(0)
X_maxs = X_train.max(0)
X_train = (X_train - X_mins) / (X_maxs - X_mins)
X_test = (X_test - X_mins) / (X_maxs - X_mins)
X = (X - X_mins) / (X_maxs - X_mins)

show = True
# TODO: Side-by-side comparisons of results from different classifiers
'''
classifiers = [
    SVC(),
    LinearSVC(),
    GaussianNB(),
    RandomForestClassifier(),
    LogisticRegression(),
    DecisionTreeClassifier(),
    ExtraTreeClassifier(),
    AdaBoostClassifier(),
    BaggingClassifier(),
    GradientBoostingClassifier(),
    StackingClassifier(),
    VotingClassifier(),
    HistGradientBoostingClassifier()
]

for classifier in classifiers:
    clf_name = classifier.__doc__[:classifier.__doc__.find(".")]
    # clf = classifier()
    clf = classifier
    clf.fit(X_train, y_train)

    y_hat = clf.predict(X_test)
    errs = sum(np.logical_xor(y_test, y_hat))
    print(clf_name)
    print("\t# Samples:", len(y_test))
    print("\t# Errors:", errs)
'''

Wall = np.array([0.050, 0.125])
Width = np.array([0.36, 1.5])
Height = np.array([0.06, 0.75])
Floor = np.array([0.050, 0.2])
Length = np.array([5.0, 30.00])

# Wall = np.array([0.050, 1])
# Width = np.array([0.5, 3])
# Height = np.array([0.36, 1.5])
# Floor = np.array([0.050, 1.5])
# Length = np.array([10.0, 30.00])

N = 1000
X_new = np.vstack((
    rng.random(N)*np.diff(Wall) + Wall.min(),
    rng.random(N)*np.diff(Width) + Width.min(),
    rng.random(N)*np.diff(Height) + Height.min(),
    rng.random(N)*np.diff(Floor) + Floor.min(),
    rng.random(N)*np.diff(Length) + Length.min()
)).T

X_new = (X_new - X_mins) / (X_maxs - X_mins)

# classifier_file = "/mnt/c/Users/jbortiz/GoogleRoot/School/Clemson/Thesis/Submissions/Journal_May2021/code/Old Code/" \
#     "gantry_clf.pkl"
# with open(classifier_file, "wb") as f:
#     dill.dump(clf, f)

'''
classifiers = [
    SVC,
    LinearSVC,
    GaussianNB,
    RandomForestClassifier,
    LogisticRegression,
    DecisionTreeClassifier,
    ExtraTreeClassifier,
    AdaBoostClassifier,
    BaggingClassifier,
    GradientBoostingClassifier,
    StackingClassifier,
    VotingClassifier,
    HistGradientBoostingClassifier
]
'''

estimators = [
    ('clf', SVC()),
    ('clf1', LinearSVC()),
    ('clf2', GaussianNB()),
    ('clf3', RandomForestClassifier()),
    ('clf4', LogisticRegression()),
    ('clf5', DecisionTreeClassifier()),
    ('clf6', ExtraTreeClassifier()),
    ('clf7', AdaBoostClassifier()),
    ('clf8', BaggingClassifier()),
    ('clf9', GradientBoostingClassifier())
]
clf = StackingClassifier(estimators)
# clf = SVC()
clf.fit(X_train, y_train)

y_new = clf.predict(X_new)
print("num fails: ", len(y_new) - y_new.sum())

classifier_file = "gantry_clf.pkl"
with open(classifier_file, "wb") as f:
    dill.dump((clf, X_mins, X_maxs), f)

if show:
    fig = make_subplots(rows=d-1, cols=d-1)
    feature_labels = ['Wall', 'Width', 'Height', 'Floor', 'Length']
    for i in range(d-1):
        R = i+1
        for ii in range(i+1, d):
            fig.add_scatter(y=X_new[y_new, i], x=X_new[y_new, ii],
                            row=R, col=ii,
                            mode='markers', marker=dict(color="PaleGreen"),
                            opacity=0.8
                            )
            fig.add_scatter(y=X_new[~y_new, i], x=X_new[~y_new, ii],
                            row=R, col=ii,
                            mode='markers', marker=dict(color="PaleVioletRed"),
                            opacity=0.5
                            )
            fig.add_scatter(y=X[y, i], x=X[y, ii],
                            row=R, col=ii,
                            mode='markers', marker=dict(color="Green"))
            fig.add_scatter(y=X[~y, i], x=X[~y, ii],
                            row=R, col=ii,
                            mode='markers', marker=dict(color="Red"))

            if R == ii:
                fig.update_yaxes(title_text=feature_labels[i], row=R, col=ii)
                fig.update_xaxes(title_text=feature_labels[ii], row=R, col=ii)

    fig.update_layout(showlegend=False)
    fig.show()

# TODO: Try regression model to predict deflection values and use threshhold
# TODO: Regression model for mass to use in other requirement calculations
