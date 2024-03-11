import logging
import os
from datetime import datetime
from io import StringIO
from pathlib import Path
from automation import TestAPI
from config import ConfigLoader, Config
from automation import AutomatedPipeline
from steps.training import load_model_data
from steps.scoring import load_model, score_model


logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
process_logger = logging.getLogger()


def log_job_execution(msg: str):
    log_buffer = StringIO()
    stream_handler = logging.StreamHandler(log_buffer)
    process_logger.addHandler(stream_handler)
    process_logger.info(f"{datetime.now().strftime('%m/%d/%Y, %H:%M:%S')} {msg}")
    with open('cron_log.txt', 'a') as f:
        f.write(log_buffer.getvalue())
    process_logger.removeHandler(stream_handler)


def is_there_new_data(config: Config, new_files: list) -> bool:
    process_logger.info('Checking for new data')
    ingested_data_ledger = 'ingestedfiles.txt'
    try:
        with open(os.path.join(config.output_folder_path, ingested_data_ledger)) as ledger:
            ingested_files = [Path(ingested_file).name for ingested_file in ledger.read().split('\n')]
            if set(new_files).issubset(ingested_files):
                process_logger.info('No new data was found')
                return False
            else:
                process_logger.info('There is new data to be ingested')
                return True
    except Exception as e:
        process_logger.error(f'Failed to read from {ingested_data_ledger}\n'
                             f'error: {e}')
        process_logger.info("Assuming this is the first run for the pipeline")
        return True if new_files is not None else False


def collect_files_from_input_folder(config: Config) -> list:
    file_list = []
    for root, directories, files in os.walk(config.input_folder_path):
        for file in files:
            file_list.append(Path(os.path.join(root, file)).name)
    return file_list


def read_current_score(config: Config, score_file='latestscore.txt') -> float:
    try:
        with open(os.path.join(config.prod_deployment_path, score_file), 'r') as f:
            return float(f.read())
    except Exception as e:
        process_logger.error(f'Failed to read from {score_file}\n'
                             f'error: {e}')


def model_drifted(config: Config):
    current_score = read_current_score(config)
    if current_score is None:
        current_score = 0
    train_df, target = load_model_data(process_logger, config.output_folder_path)
    model = load_model(process_logger, config.prod_deployment_path)
    new_score = score_model(process_logger, model, train_df, target)
    if new_score > current_score:
        process_logger.info("Found drift in model")
        return True
    else:
        process_logger.info("No drift found in model")
        return False


def run():
    log_job_execution("Starting model evaluation")
    config = ConfigLoader.init_from_json_file(process_logger)
    input_files = collect_files_from_input_folder(config)

    if is_there_new_data(config, input_files):
        log_job_execution("New data found")
        AutomatedPipeline.run_task("ingestion")
    else:
        log_job_execution("No new data found")

    if model_drifted(config):
        log_job_execution("Found drift in model")
        AutomatedPipeline.run_task("training")
        AutomatedPipeline.run_task("scoring")
        AutomatedPipeline.run_task("deployment")
    else:
        log_job_execution("No drift found in model")
    log_job_execution("Running diagnostics")
    TestAPI.hit_all_api_endpoints()
    log_job_execution("Job finished correctly.")
    exit(0)


if __name__ == '__main__':
    run()
