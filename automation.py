from apicalls import InvokeRestMethod, URL
from typing import Iterator, Coroutine
import importlib.metadata as metadata
from dataclasses import dataclass
from pathlib import Path
from enum import Enum
import pandas as pd
import subprocess
import asyncio
import sys
import os


class CommandLineRunner(Enum):
    PYTHON = 'python3'
    EXECUTABLE = ''
    SHELL = 'sh'

    @classmethod
    def get_runner_for_extension(cls, extension: str) -> Enum:
        if extension == '.py':
            return cls.PYTHON
        elif extension == '.sh':
            return cls.SHELL
        return cls.EXECUTABLE


@dataclass
class PipelineTask:
    runner: Enum
    file_path: Path
    name: str


class AutomatedPipeline:

    @classmethod
    def run_task(cls, task_name: str):
        try:
            task = cls.load_task(task_name)
            os.system(f"{task.runner.value} {task.file_path}")
        except Exception as e:
            raise Exception(f"Falied to run Task {task_name}: {e}")

    @classmethod
    def load_task(cls, task_name: str) -> PipelineTask:
        task_file, extension = cls.find_task_file(task_name)

        return PipelineTask(name=task_name, file_path=task_file,
                            runner=CommandLineRunner.get_runner_for_extension(extension))

    @staticmethod
    def find_task_file(task_name: str, pipeline_dir='steps') -> (Path, str):
        steps_dir = Path(os.getcwd()) / pipeline_dir
        for root, dirs, files in os.walk(steps_dir):
            for file in files:
                file_name, extension = os.path.splitext(file)
                if file_name == task_name:
                    return steps_dir / file, extension


class DependencyChecker:

    @staticmethod
    async def _get_module_versions():
        print("Getting module versions...")
        installed_packages = metadata.distributions()
        package_names = []
        package_versions = []

        for package in installed_packages:
            package_names.append(package.metadata['Name'])
            package_versions.append(package.version)
            print(f"{package.metadata['Name']}=={package.version}")

        return pd.DataFrame({'Name': package_names, 'Version': package_versions})

    @staticmethod
    async def _check_for_updates():
        print('Checking for updates...')
        try:
            result = subprocess.run([sys.executable, '-m', 'pip', 'list', '--outdated'],
                                    capture_output=True, text=True, check=True)
            outdated_packages = result.stdout.strip().split('\n')[2:]
            package_names = []
            package_latest_versions = []
            if outdated_packages:
                print("Outdated packages:")
                for package_info in outdated_packages:
                    package_name, current_version, latest_version, *_ = package_info.split()
                    package_names.append(package_name)
                    package_latest_versions.append(latest_version)
                    print(f"{package_name}: {current_version} -> {latest_version}")
            else:
                print("All packages are up to date.")
            return pd.DataFrame({'Name': package_names, 'Latest_Version': package_latest_versions})
        except subprocess.CalledProcessError as e:
            print("Error:", e)

    @classmethod
    async def get_dependency_check(cls):
        current_df, latest_df = await asyncio.gather(cls._get_module_versions(), cls._check_for_updates())

        merged_df = pd.merge(current_df, latest_df, on='Name', how='left')
        merged_df['Latest_Version'] = merged_df['Latest_Version'].fillna(merged_df['Version'])
        return merged_df

    @classmethod
    def write_dependency_report(cls, directory: Path) -> Path:
        dependencies = asyncio.run(cls.get_dependency_check())
        file_full_path = directory / 'dependencyreport.txt'
        dependencies.to_csv(file_full_path, sep='\t', index=False)
        return file_full_path


class TestAPI:

    no_parameter_endpoints = [["summarystats", "scoring"], "write_dependency_report", "diagnostics"]

    parameter_endpoints = {
        "async": [{
            "prediction": {
                "dataset_location": "testdata",
                "dataset_name": "testdata.csv"
            }
        }]

    }

    @classmethod
    def hit_all_api_endpoints(cls):
        results = asyncio.run(cls.hit_non_parameters_endpoints(cls.no_parameter_endpoints))
        results += asyncio.run(cls.hit_parameters_endpoints(cls.parameter_endpoints))
        InvokeRestMethod.write_responses_to_file(results)

    @classmethod
    async def hit_non_parameters_endpoints(cls, endpoints: list) -> list:
        results = []
        for task in endpoints:
            if isinstance(task, list):
                async_calls = [InvokeRestMethod.get(f"{URL}/{_}") for _ in task]
                result = await asyncio.gather(*async_calls, return_exceptions=True)
                results += result
            else:
                result = await InvokeRestMethod.get(f"{URL}/{task}")
                results.append(result)
        return results

    @classmethod
    async def hit_parameters_endpoints(cls, endpoints: dict) -> list:
        results = []
        for endpoint, values in endpoints.items():
            if endpoint == "async":
                async_calls = cls.collect_calls_from_dict(values)
                result = await asyncio.gather(*async_calls)
                results += result
            else:
                result = await InvokeRestMethod.post(f"{URL}/{endpoint}", body=values)
                results.append(result)
        return results

    @staticmethod
    def collect_calls_from_dict(calls: Iterator[dict]) -> list[Coroutine]:
        results = []
        for call in calls:
            endpoint = list(call.keys())[0]
            body = call[endpoint]
            request = InvokeRestMethod.post(f"{URL}/{endpoint}", body=body)
            results.append(request)
        return results
