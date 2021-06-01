import numpy as np
from plotly.subplots import make_subplots
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
import dill
from numpy.random import default_rng
RANDOM_STATE = 42

rng = default_rng(RANDOM_STATE)

results_file = "/mnt/c/Users/jbortiz/GoogleRoot/School/Clemson/Thesis/Submissions/Journal_May2021/code/Ansys Data/" \
    "ANSYS_Results_2021Mar11.csv"
X = np.loadtxt(results_file, skiprows=1, delimiter=',')
# Cols: 0-Wall, 1-Width, 2-Height, 3-Floor, 4-Length, 5-Dfrmtn, 6-Mass

# Classifying based on deformation, mass ignored
n, d = X[:, :-2].shape
y = X[:, -2]

# split = 0.8
# X_train, X_test = X[:int(split*n), :-2], X[int(split*n):, :-2]
# y_train, y_test = y[:int(split*n)], y[int(split*n):]
X_train, X_test, y_train, y_test = train_test_split(X[:, :-2], y, test_size=0.33, random_state=42)
Xscaler = MinMaxScaler()
Yscaler = MinMaxScaler()
X_train = Xscaler.fit_transform(X_train)
y_train = Yscaler.fit_transform(y_train.reshape(-1, 1)).flatten()
X_test = Xscaler.transform(X_test)
y_test = Yscaler.transform(y_test.reshape(-1, 1)).flatten()

from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.gaussian_process.kernels import RBF

show = True
# TODO: Side-by-side comparisons of results from different classifiers
regressors = [
    SVR(C=1.0, epsilon=0.2),
    KNeighborsRegressor(n_neighbors=5, weights='distance'),
    GaussianProcessRegressor(kernel=RBF(), random_state=RANDOM_STATE),
    MLPRegressor(random_state=RANDOM_STATE, max_iter=1000),
    SGDRegressor(max_iter=1000, tol=1e-3)
]

for regressor in regressors:
    regr_name = regressor.__doc__[:regressor.__doc__.find(".")]
    regressor.fit(X_train, y_train)

    y_hat = regressor.predict(X_test)
    # y_hat = Yscaler.inverse_transform(y_hat.reshape(-1, 1)).flatten()
    MSE = mean_squared_error(y_test, y_hat)
    MAE = mean_absolute_error(y_test, y_hat)
    R2 = r2_score(y_test, y_hat)
    print(regr_name)
    print("\t# Samples:", len(y_test))
    print("\tMSE:", MSE)
    print("\tMSE:", MAE)
    print("\tR^2:", R2)

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

# classifier_file = "/mnt/c/Users/jbortiz/GoogleRoot/School/Clemson/Thesis/Submissions/Journal_May2021/code/Old Code/" \
#     "gantry_regr.pkl"
# with open(classifier_file, "wb") as f:
#     dill.dump(regr, f)

# regr = GaussianProcessRegressor(kernel=RBF(), random_state=RANDOM_STATE)
# regr = KNeighborsRegressor(n_neighbors=5, weights='distance')
# regr = MLPRegressor(random_state=RANDOM_STATE, max_iter=1000)
# regr = SGDRegressor(max_iter=1000, tol=1e-3)
regr = SVR(C=1.0, epsilon=0.2)
regr.fit(X_train, y_train)
y_hat = regr.predict(X_new)
y_hat = Yscaler.inverse_transform(y_hat.reshape(-1, 1)).flatten()
y_new = y_hat < 0.001
y = y < 0.001

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
