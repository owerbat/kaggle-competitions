from catboost import CatBoostClassifier

from utils import read_data, train_and_score, submit_result


if __name__ == '__main__':
    train, test, _, sample_submission = read_data()
    model = CatBoostClassifier(iterations=50000,
                               learning_rate=1e-3,
                               depth=4,
                               loss_function=None,
                               random_seed=None,
                               class_weights=None,
                               auto_class_weights='Balanced',
                               early_stopping_rounds=20,
                               use_best_model=True)
    # train_and_score(model, train, cat_features=['EJ'])
    train_and_score(model, train, eval_set=True, cat_features=['EJ'])
    submit_result(model, test, sample_submission, 'catboost')
