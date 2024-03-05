from config import Config, ConfigLoader
from dataclasses import dataclass
from typing import Iterable
from pathlib import Path
import logging
import shutil
import os


logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
deployment_logger = logging.getLogger()


@dataclass
class DeploymentComponent:
    target_path: Path
    source_path: Path
    name: str


def compile_deployment_components(logger, config: Config) -> Iterable[DeploymentComponent]:
    logger.info("Compiling deployment components")
    model = DeploymentComponent(name='model', source_path=config.output_model_path / 'trainedmodel.pkl',
                                target_path=config.prod_deployment_path)

    ingestedfiles_record = DeploymentComponent(name='ingestedfiles_record',
                                               source_path=config.output_folder_path / 'ingestedfiles.txt',
                                               target_path=config.prod_deployment_path)

    score_record = DeploymentComponent(name='score_record', source_path=config.output_model_path / 'latestscore.txt',
                                       target_path=config.prod_deployment_path)

    components = [model, ingestedfiles_record, score_record]

    for component in components:
        if not component.source_path.exists():
            logger.error(f"Source path {component.source_path} for component {component.name} does not exist")
            raise FileNotFoundError(f"{component.source_path} not found.")
        logger.info(f"Compiled {component.name} at {component.source_path}")
    return components


def copy_deployment_component_to_target(logger, component: DeploymentComponent) -> bool:
    logger.info(f"Copying deployment component {component.name} to {component.target_path}")
    if not os.path.isdir(component.target_path):
        logger.warn(f"Target directory {component.target_path} not found. Attempting to create directory")
        os.mkdir(component.target_path)
    try:
        shutil.copy(component.source_path, component.target_path)
    except Exception as e:
        logger.error(f"Cannot copy {component.name} from {component.source_path} to {component.target_path}\n"
                     f"Error message: {e}")
        return False
    return True


def deploy(logger, components: Iterable[DeploymentComponent]) -> bool:
    logger.info(f"Starting deployment of {len(list(components))} components")
    for component in components:
        copy_deployment_component_to_target(logger, component)
    return True


def run():
    config = ConfigLoader.init_from_json_file(deployment_logger)
    deploy_components = compile_deployment_components(deployment_logger, config)
    success = deploy(deployment_logger, deploy_components)
    deployment_logger.info(f"Deployment completed successfully: {success}")


if __name__ == '__main__':
    run()
