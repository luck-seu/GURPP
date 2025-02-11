from sklearn import linear_model
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold
import numpy as np

crime_counts = np.load('data/task/crime_counts.npy', allow_pickle=True)
check_counts = np.load('data/task/check_counts.npy', allow_pickle=True)
carbon_counts = np.load('data/task/carbon_counts.npy', allow_pickle=True)
income_counts = np.load('data/task/income_counts.npy', allow_pickle=True)

def regression(X_train, y_train, X_test, alpha):
    reg = linear_model.Ridge(alpha=alpha)
    reg.fit(X_train, y_train)
    y_pred = reg.predict(X_test)
    return y_pred

def kf_predict(X, Y):
    kf = KFold(n_splits=5)
    y_preds = []
    y_truths = []
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = Y[train_index], Y[test_index]
        y_pred = regression(X_train, y_train, X_test, 1)
        y_preds.append(y_pred)
        y_truths.append(y_test)

    return np.concatenate(y_preds), np.concatenate(y_truths)


def compute_metrics(y_pred, y_test):
    y_pred[y_pred < 0] = 0
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    return mae, np.sqrt(mse), r2


def predict_crime(emb):
    y_pred, y_test = kf_predict(emb, crime_counts)
    mae, rmse, r2 = compute_metrics(y_pred, y_test)
    return mae, rmse, r2


def predict_check(emb):
    y_pred, y_test = kf_predict(emb, check_counts)
    mae, rmse, r2 = compute_metrics(y_pred, y_test)
    return mae, rmse, r2

def predict_carbon(emb):
    y_pred, y_test = kf_predict(emb, carbon_counts)
    mae, rmse, r2 = compute_metrics(y_pred, y_test)
    return mae, rmse, r2

def predict_income(emb):
    y_pred, y_test = kf_predict(emb, income_counts)
    mae, rmse, r2 = compute_metrics(y_pred, y_test)
    return mae, rmse, r2