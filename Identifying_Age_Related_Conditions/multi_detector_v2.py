import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from autosklearn.classification import AutoSklearnClassifier

from utils import read_data, submit_result


class SumProbaModel:
    def __init__(self, threshold: float = .5) -> None:
        self.threshold = threshold

    def predict_proba(self, x: np.ndarray):
        if isinstance(x, pd.DataFrame):
            x = x.to_numpy()

        p1 = x[:, 0]
        p2 = x[:, 1]
        p3 = x[:, 2]

        p = p1 + p2 + p3 - p1 * p2 - p2 * p3 - p1 * p3 + p1 * p2 * p3
        p = np.clip(p, 0, 1)

        return np.vstack([1 - p, p]).T

    def predict(self, x: np.ndarray):
        proba = self.predict_proba(x)[:, 1]
        return np.array([1 if p > self.threshold else 0 for p in proba])


if __name__ == '__main__':
    train, test, greeks, sample_submission = read_data()
    train = train.merge(greeks.loc[:, 'Alpha'], 'inner', left_index=True, right_index=True)

    train['EJ'] = train['EJ'].replace({'A': 0, 'B': 1})
    test['EJ'] = test['EJ'].replace({'A': 0, 'B': 1})

    condition_trains, condition_vals = {}, {}
    for condition in ['A', 'B', 'D', 'G']:
        condition_trains[condition], condition_vals[condition] = train_test_split(
            train.loc[train.Alpha.isin([condition])], test_size=.2, random_state=0)

    models = {}
    for condition in ['B', 'D', 'G']:
        print(f'Condition {condition}')
        x_train = pd.concat([condition_trains['A'], condition_trains[condition]]) \
            .drop(['Id', 'Alpha', 'Class'], axis=1) \
            .reset_index(drop=True)
        x_val = pd.concat([condition_vals['A'], condition_vals[condition]]) \
            .drop(['Id', 'Alpha', 'Class'], axis=1) \
            .reset_index(drop=True)
        y_train = np.array([0] * condition_trains['A'].shape[0] +
                           [1] * condition_trains[condition].shape[0])
        y_val = np.array([0] * condition_vals['A'].shape[0] +
                         [1] * condition_vals[condition].shape[0])

        train_idxs = list(range(len(y_train)))
        val_idxs = list(range(len(y_val)))
        rng = np.random.default_rng(0)
        rng.shuffle(train_idxs)
        rng.shuffle(val_idxs)

        x_train = x_train.loc[train_idxs]
        y_train = y_train[train_idxs]
        x_val = x_val.loc[val_idxs]
        y_val = y_val[val_idxs]

        # model = CatBoostClassifier(iterations=50000,
        #                            learning_rate=1e-3,
        #                            depth=4,
        #                            loss_function=None,
        #                            random_seed=None,
        #                            class_weights=None,
        #                            auto_class_weights='Balanced',
        #                            early_stopping_rounds=20,
        #                            use_best_model=True)
        # model.fit(x_train,
        #           y_train,
        #           cat_features=['EJ'],
        #           eval_set=(x_val, y_val),
        #           verbose=False)
        model = AutoSklearnClassifier()
        feat_type = ["Categorical" if col == 'EJ' else "Numerical" for col in x_train.columns]
        model.fit(x_train, y_train, x_val, y_val)
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

        models[condition] = model

    common_train = pd.concat(
        [condition_trains[condition] for condition in ['A', 'B', 'D', 'G']],
        ignore_index=True
    )
    common_val = pd.concat(
        [condition_vals[condition] for condition in ['A', 'B', 'D', 'G']],
        ignore_index=True
    )

    y_train = common_train.Class
    y_val = common_val.Class

    inputs_train = common_train.drop(['Id', 'Alpha', 'Class'], axis=1)
    inputs_val = common_val.drop(['Id', 'Alpha', 'Class'], axis=1)

    x_train = pd.DataFrame({
        condition: model.predict_proba(inputs_train)[:, 1]
        for condition, model in models.items()
    })
    x_val = pd.DataFrame({
        condition: model.predict_proba(inputs_val)[:, 1]
        for condition, model in models.items()
    })

    train_idxs = list(range(len(y_train)))
    val_idxs = list(range(len(y_val)))
    rng = np.random.default_rng(0)
    rng.shuffle(train_idxs)
    rng.shuffle(val_idxs)

    x_train = x_train.loc[train_idxs]
    y_train = y_train.loc[train_idxs]
    x_val = x_val.loc[val_idxs]
    y_val = y_val.loc[val_idxs]

    common_model = LogisticRegression()
    common_model.fit(x_train, y_train)
    # common_model = SumProbaModel()
    preds_train = common_model.predict(x_train)
    preds_val = common_model.predict(x_val)

    print('Final model')
    print(pd.DataFrame({
        data_type: [accuracy_score(labels, preds),
                    f1_score(labels, preds),
                    precision_score(labels, preds),
                    recall_score(labels, preds)]
        for data_type, preds, labels in zip(['train', 'val'],
                                            [preds_train, preds_val],
                                            [y_train, y_val])
    }, index=['accuracy', 'f1', 'precision', 'recall']))

    new_test = pd.DataFrame({
        condition: model.predict_proba(test.drop('Id', axis=1))[:, 1]
        for condition, model in models.items()
    })
    print(new_test)
    submit_result(common_model, new_test, sample_submission, 'multi_detector_v2')
