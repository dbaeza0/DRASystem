from sklearn.linear_model import LogisticRegression
from steps.diagnostics import model_predictions
from steps.training import load_model_data
from steps.scoring import load_model
from matplotlib import pyplot as plt
from config import ConfigLoader
from sklearn import metrics
from pathlib import Path
import seaborn as sns
import pandas as pd
import logging


logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
reporting_logger = logging.getLogger()


def score_model(logger, data_x: pd.DataFrame, data_y: pd.Series, model: LogisticRegression, save_path: Path,
                filename='confusionmatrix.png'):

    logger.info('Starting score model')
    y_pred = model_predictions(logger, model, data_x)
    confusion_matrix = metrics.confusion_matrix(data_y, y_pred)
    plt.figure(figsize=(6, 6))

    fig = sns.heatmap(confusion_matrix, annot=True, cmap='Blues')

    fig.set_title('Confusion matrix Logistic Regression')
    fig.set_xlabel('\nPredicted Values')
    fig.set_ylabel('Actual Values')
    fig.xaxis.set_ticklabels(['False', 'True'])
    fig.yaxis.set_ticklabels(['False', 'True'])
    file_path = save_path / filename
    plt.savefig(file_path)
    logger.info(f"Confusion matrix score saved as {file_path}")


def run():
    config = ConfigLoader.init_from_json_file(reporting_logger)
    test_x, test_y = load_model_data(reporting_logger, config.test_data_path, file_name='testdata.csv')
    model = load_model(reporting_logger, config.output_model_path)
    score_model(reporting_logger, test_x, test_y, model, config.output_model_path)


if __name__ == '__main__':
    run()
