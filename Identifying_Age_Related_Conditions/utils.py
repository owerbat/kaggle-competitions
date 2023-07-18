from typing import Any
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, \
    log_loss
from sklearn.model_selection import train_test_split


def balanced_log_loss(y_true, y_pred):
    nc = np.bincount(y_true)
    return log_loss(y_true, y_pred, sample_weight=1 / nc[y_true], eps=1e-15)


def read_data():
    train = pd.read_csv('./data/train.csv')
    test = pd.read_csv('./data/test.csv')
    greeks = pd.read_csv('./data/greeks.csv')
    sample_submission = pd.read_csv('./data/sample_submission.csv')

    return train, test, greeks, sample_submission


def split_x_y(train: pd.DataFrame, test: pd.DataFrame = None):
    y_train = train.loc[:, 'Class']
    x_train = train.drop(['Id', 'Class'], axis=1)

    if test is None:
        return x_train, y_train

    x_test = test.drop(['Id'], axis=1)
    return x_train, y_train, x_test


def train_and_score(model: Any,
                    train: pd.DataFrame,
                    eval_set: bool = False,
                    **train_args):
    x_train, y_train = split_x_y(train)
    x_train, x_val, y_train, y_val = train_test_split(
        x_train, y_train, test_size=.2, random_state=0)

    if eval_set:
        model.fit(x_train, y_train, **train_args, eval_set=(x_val, y_val))
    else:
        model.fit(x_train, y_train, **train_args)

    preds_train = model.predict(x_train)
    preds_val = model.predict(x_val)

    print(pd.DataFrame({
        data_type: [accuracy_score(labels, preds),
                    f1_score(labels, preds),
                    precision_score(labels, preds),
                    recall_score(labels, preds)]
        for data_type, preds, labels in zip(['train', 'val'],
                                            [preds_train, preds_val],
                                            [y_train, y_val])
    }, index=['accuracy', 'f1', 'precision', 'recall']))


def submit_result(model: Any,
                  test: pd.DataFrame,
                  sample_submission: pd.DataFrame,
                  name: str = 'submission'):
    x_test = test.drop(['Id'], axis=1) if 'Id' in test.columns else test
    preds = model.predict_proba(x_test)

    sample_submission['class_0'] = preds[:, 0]
    sample_submission['class_1'] = preds[:, 1]
    sample_submission.to_csv(f'submissions/{name}.csv', index=False)
