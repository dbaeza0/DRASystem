from dataclasses import dataclass, fields
from pathlib import Path
import json


@dataclass
class Config:
    input_folder_path: Path
    output_folder_path: Path
    test_data_path: Path
    output_model_path: Path
    prod_deployment_path: Path


class ConfigLoader:
    CONFIG_FILE_NAME = Path.cwd() / "config.json"

    @classmethod
    def init_from_json_file(cls, logger, json_file_path=CONFIG_FILE_NAME):
        logger.info("Reading config from {}".format(json_file_path))
        config = cls._load_json(json_file_path)
        if not set(config.keys()) == set([field.name for field in fields(Config)]):
            raise ValueError(f"Provided json config {json_file_path}"
                             f"does not match expected config {Config.__dict__.keys()}")
        return Config(**{k: Path(Path.cwd() / v) for k, v in config.items()})

    @staticmethod
    def _load_json(json_file: Path) -> dict:
        if not json_file.is_file():
            raise FileNotFoundError(f"File {json_file} not found")

        with open(json_file, "r") as conf_file:
            return json.load(conf_file)
