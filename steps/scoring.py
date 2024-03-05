from sklearn.linear_model import LinearRegression
from steps.training import load_model_data
from config import ConfigLoader
from sklearn import metrics
from pathlib import Path
import pandas as pd
import logging
import pickle

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
scoring_logger = logging.getLogger()


def load_model(logger, model_path: Path, model_name='trainedmodel.pkl') -> LinearRegression:
    model_file = model_path / model_name
    logger.info(f"Loading model from {model_file}")
    if not model_file.is_file():
        raise FileNotFoundError(f"Model file {model_file} not Found")
    return pickle.load(open(model_file, "rb"))


def score_model(logger, model: LinearRegression, x_df: pd.DataFrame, y: pd.Series):
    logger.info(f"Scoring model {model.__class__}")
    y_pred = model.predict(x_df)
    f1_score = metrics.f1_score(y, y_pred)
    logger.info(f"F1 score is {f1_score}")
    return f1_score


def save_model_score(logger, score: float, path: Path, file_name='latestscore.txt') -> None:
    file_path = path / file_name
    logger.info(f"Saving score to {file_path}")
    with open(file_path, 'w') as file:
        file.write(str(score) + '\n')
    logger.info(f"Score saved successfully to {file_path}")


def run():
    config = ConfigLoader.init_from_json_file(scoring_logger)
    model = load_model(scoring_logger, config.output_model_path)
    x_df, y = load_model_data(scoring_logger, config.test_data_path, file_name='testdata.csv')
    score = score_model(scoring_logger, model, x_df, y)
    save_model_score(scoring_logger, score, config.output_model_path)


if __name__ == '__main__':
    run()
