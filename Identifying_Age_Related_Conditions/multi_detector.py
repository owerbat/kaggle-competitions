import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.linear_model import LogisticRegression

from utils import read_data, train_and_score, submit_result


if __name__ == '__main__':
    train, test, greeks, sample_submission = read_data()
    original_labels = train['Class']
    train.drop(['Class'], axis=1)

    trains = {condition: train.loc[greeks.loc[greeks.Alpha.isin(['A', condition])].index]
              for condition in ['B', 'D', 'G']}
    models = {}

    for condition, train_data in trains.items():
        print(f'Condition {condition}')
        model = CatBoostClassifier(iterations=50000,
                                   learning_rate=1e-3,
                                   depth=4,
                                   loss_function=None,
                                   random_seed=None,
                                   class_weights=None,
                                   auto_class_weights='Balanced',
                                   early_stopping_rounds=20,
                                   use_best_model=True)

        train_data['Class'] = np.zeros(train_data.shape[0])
        train_data['Class'].loc[greeks.loc[greeks.Alpha.isin([condition])].index] = 1

        train_and_score(model, train_data, eval_set=True, cat_features=['EJ'], verbose=False)
        models[condition] = model

    # new_train = np.vstack([
    #     model.predict_proba(train.drop('Id', axis=1))[:, 1]
    #     for _, model in models.items()
    # ]).T
    # new_train = np.hstack([new_train, original_labels.to_numpy().reshape((-1, 1))])
    new_train = pd.DataFrame({
        condition: model.predict_proba(train.drop('Id', axis=1))[:, 1]
        for condition, model in models.items()
    })
    new_train['Id'] = train.Id
    new_train['Class'] = original_labels

    common_model = LogisticRegression()
    train_and_score(common_model, new_train, eval_set=False)
    # common_model = CatBoostClassifier(iterations=50000,
    #                                   learning_rate=1e-3,
    #                                   depth=4,
    #                                   loss_function=None,
    #                                   random_seed=None,
    #                                   class_weights=None,
    #                                   auto_class_weights='Balanced',
    #                                   early_stopping_rounds=20,
    #                                   use_best_model=True)
    # train_and_score(common_model, new_train, eval_set=True)

    new_test = pd.DataFrame({
        condition: model.predict_proba(test.drop('Id', axis=1))[:, 1]
        for condition, model in models.items()
    })
    new_test['Id'] = test.Id
    print(new_test)
    submit_result(common_model, new_test, sample_submission, 'multi_detector')
