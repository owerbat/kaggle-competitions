from catboost import CatBoostClassifier

from utils import read_data, train_and_score, submit_result


if __name__ == '__main__':
    train, test, _, sample_submission = read_data()
    model = CatBoostClassifier()
    train_and_score(model, train, cat_features=['EJ'])
    submit_result(model, test, sample_submission, 'baseline')
