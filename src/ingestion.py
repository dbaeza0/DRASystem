from config import ConfigLoader
from pathlib import Path
import pandas as pd
import logging
import os


logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
ingestion_logger = logging.getLogger()


def list_csv_files_in_dir(dir_path: Path):
    return list(dir_path.glob("*.csv"))


def read_csv(logger, csv_path: Path) -> pd.DataFrame:
    logger.info(f"Reading CSV {csv_path}")
    if not csv_path.is_file():
        raise FileNotFoundError(f"csv file {csv_path} not found")
    return pd.read_csv(csv_path)


def merge_multiple_dataframe(logger, dir_path: Path) -> [pd.DataFrame, list]:
    logger.info(f"Reading CSVs from dir: {dir_path}")
    if not dir_path.is_dir():
        raise FileNotFoundError(f"csv directory {dir_path} not found")
    found_csvs = list_csv_files_in_dir(dir_path)
    logger.info(f"Reading found files {found_csvs}")
    result_df = pd.DataFrame()
    for csv_found in found_csvs:
        df = read_csv(logger, dir_path / csv_found)
        result_df = pd.concat([result_df, df], ignore_index=True)
    logger.info(f"Resulting dataframe info: {result_df.info()}")
    return result_df, found_csvs


def save_csv(logger, df, dir_path: Path, filename='finaldata.csv') -> None:
    full_path = dir_path / filename
    logger.info(f"Saving result dataset to: {full_path}")
    try:
        df.to_csv(full_path, index=False)
    except FileNotFoundError:
        os.mkdir(dir_path)
        df.to_csv(full_path, index=False)
    logger.info("Successfully saved")


def record_processed_datasets(logger, dir_path: Path, files: list, filename='ingestedfiles.txt') -> None:
    full_path = dir_path / filename
    logger.info(f"Recording ingested files to: {full_path}")
    with open(full_path, 'w') as f:
        for file in files:
            f.write(file + '\n')
    logger.info("Record successfully saved")


def basic_df_preprocessing(logger, df: pd.DataFrame) -> pd.DataFrame:
    logger.info("Dropping duplicate rows")
    return df.drop_duplicates(ignore_index=True)


if __name__ == "__main__":
    config = ConfigLoader.init_from_json_file(ingestion_logger)
    res_df, csv_proccessed = merge_multiple_dataframe(ingestion_logger, config.input_folder_path)
    res_df = basic_df_preprocessing(ingestion_logger, res_df)
    save_csv(ingestion_logger, res_df, config.output_folder_path)
    record_processed_datasets(ingestion_logger, config, csv_proccessed)
