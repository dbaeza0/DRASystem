from pathlib import Path

from sklearn.linear_model import LogisticRegression
from config import ConfigLoader
from steps.ingestion import read_csv
import pandas as pd
import logging
import pickle
import os

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
training_logger = logging.getLogger()


def load_model_data(logger: logging.Logger, data_path: Path, file_name='finaldata.csv') -> (pd.DataFrame, pd.Series):
    file_path = data_path / file_name
    df = read_csv(logger, file_path)
    target_variable = 'exited'
    logger.info(f"Splitting data, target variable: {target_variable}")
    y = df[target_variable]
    df.drop(columns=['exited'], inplace=True)
    return df.iloc[:, 1:], y


def train_model(logger, x_df: pd.DataFrame, y: pd.Series) -> LogisticRegression:
    """
    Function for training the model
    """
    logger.info('Training the model')
    # use this logistic regression for training
    model = LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
                               intercept_scaling=1, l1_ratio=None, max_iter=100,
                               multi_class='auto', n_jobs=None, penalty='l2',
                               random_state=0, solver='liblinear', tol=0.0001, verbose=0,
                               warm_start=False)

    # fit the logistic regression to the data
    model.fit(x_df, y)
    return model


def save_model(logger, model: LogisticRegression, model_path: Path, model_name='trainedmodel.pkl') -> None:
    model_file = model_path / model_name
    logger.info(f'Saving the model to {model_file}')
    try:
        pickle.dump(model, open(model_file, 'wb'))
    except OSError:
        os.mkdir(model_path)
        pickle.dump(model, open(model_file, 'wb'))


def run():
    config = ConfigLoader.init_from_json_file(training_logger)
    train_df, target = load_model_data(training_logger, config.output_folder_path)
    model = train_model(training_logger, train_df, target)
    save_model(training_logger, model, config.output_model_path)


if __name__ == '__main__':
    run()
