import logging
import timeit
from pathlib import Path
from typing import Iterable
from automation import AutomatedPipeline, DependencyChecker
import pandas as pd
from sklearn.linear_model import LogisticRegression
from steps.training import load_model_data
from steps.scoring import load_model
from config import ConfigLoader
from dataclasses import dataclass

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
diagnostics_logger = logging.getLogger()


@dataclass
class ColumnDiagnosticsSummary:
    stddev: float
    col_name: str
    media: float
    mean: float


def model_predictions(logger, model: LogisticRegression, x_df: pd.DataFrame) -> pd.Series:
    logger.info("Calculating predictions for deployed model.")
    predictions = model.predict(x_df)
    return predictions


def dataframe_summary(logger, x_df: pd.DataFrame) -> Iterable[ColumnDiagnosticsSummary]:
    logger.info("Calculating diagnostic summary of the dataset.")
    stddev = x_df.std()
    media = x_df.median()
    mean = x_df.mean()
    diagnostics = []
    for col_name in x_df.columns:
        diagnostics.append(ColumnDiagnosticsSummary(col_name=col_name,
                                                    stddev=stddev[col_name],
                                                    media=media[col_name],
                                                    mean=mean[col_name]))
    return diagnostics


def missing_data(logger, x_df: pd.DataFrame) -> dict:
    logger.info('Recording missing data.')
    missing = x_df.isna().sum()
    n_data = x_df.shape[0]
    missing = missing / n_data
    return missing.to_dict()


def execution_time(logger):
    logger.info('Timing ingestion and training pipeline steps')
    start_time = timeit.default_timer()
    AutomatedPipeline.run_task("ingestion")
    ingestion_timing = timeit.default_timer() - start_time

    start_time = timeit.default_timer()
    AutomatedPipeline.run_task("training")
    training_timing = timeit.default_timer() - start_time

    return [ingestion_timing, training_timing]


def outdated_packages_list(logger, directory: Path):
    logger.info('Generating dependencies report')
    file_full_path = directory
    try:
        file_full_path = DependencyChecker.write_dependency_report(directory)
    except Exception as e:
        logger.error(f"Failed to write dependency report to {file_full_path}: \n"
                     f"{e}")
        raise e
    logger.info(f'Report written successfully at {file_full_path}')
    return True


def run():
    config = ConfigLoader.init_from_json_file(diagnostics_logger)
    test_x, test_y = load_model_data(diagnostics_logger, config.test_data_path, file_name='testdata.csv')
    model = load_model(diagnostics_logger, config.output_model_path)
    preds = model_predictions(diagnostics_logger, model, test_x)
    stats_x, stats_y = load_model_data(diagnostics_logger, config.output_folder_path)
    diagnositcs_summ = dataframe_summary(diagnostics_logger, stats_x)
    miss_data = missing_data(diagnostics_logger, stats_x)
    times = execution_time(diagnostics_logger)
    outdated_packages_list(diagnostics_logger, config.prod_deployment_path)
    print(preds, diagnositcs_summ, miss_data, times)


if __name__ == '__main__':
    run()
