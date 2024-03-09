import pandas as pd

from steps.scoring import load_model, score_model
from flask import Flask, jsonify, request, abort
from steps.diagnostics import model_predictions, dataframe_summary, missing_data, execution_time, outdated_packages_list
from steps.training import load_model_data
from config import ConfigLoader
from dataclasses import asdict
import logging

app = Flask(__name__)
app.secret_key = '1652d576-484a-49fd-913a-6879acfa6ba4'

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
app_logger = logging.getLogger()

config = ConfigLoader.init_from_json_file(app_logger)

prediction_model = load_model(app_logger, config.output_model_path)


@app.route("/prediction", methods=['POST', 'OPTIONS'])
def predict():
    dataset_path = request.get_json()['dataset_location']
    dataset_name = request.get_json()['dataset_name']

    try:
        data_x, data_y = load_model_data(app_logger, dataset_path, file_name=dataset_name)
    except Exception as e:
        app_logger.error(F"dataset {dataset_name} not found at {dataset_path}")
        return abort(404, f"Dataset {dataset_name} not found at {dataset_path}\n"
                          f"error message: {e}")

    y_pred = model_predictions(app_logger, prediction_model, data_x)
    return jsonify({"dataset_name": dataset_name, "model": prediction_model.__class__, "y_pred": y_pred})


@app.route("/scoring", methods=['GET', 'OPTIONS'])
def scoring():
    data_x, data_y = load_model_data(app_logger, config.test_data_path, file_name='testdata.csv')
    score = score_model(app_logger, prediction_model, data_x, data_y)
    return jsonify({"model": str(prediction_model.__class__), "f1score": score})


@app.route("/summarystats", methods=['GET', 'OPTIONS'])
def summarystats():
    stats_x, _ = load_model_data(app_logger, config.output_folder_path)
    data_summaries = dataframe_summary(app_logger, stats_x)
    return jsonify({"ingested_data_summaries": [asdict(datasum) for datasum in data_summaries]})


@app.route("/diagnostics", methods=['GET', 'OPTIONS'])
def diagnostics():
    stats_x, _ = load_model_data(app_logger, config.output_folder_path)
    miss_data = missing_data(app_logger, stats_x)
    times = execution_time(app_logger)
    dep_report = config.prod_deployment_path / 'dependencyreport.txt'
    if not dep_report.exists():
        abort(404, f"Dependency report not found at {dep_report}\n"
                   f"Because this is a lengthy operation, the collection of "
                   f"dependencies does not run in realtime, please hit /write_dependency_report"
                   f"endpoint to generate (or update) a dependency report.")
    else:
        dependency_report = pd.read_csv(config.prod_deployment_path / 'dependencyreport.txt', sep='\t')
        return jsonify({'missing': miss_data, 'time_check': times,
                        'dependency_report': dependency_report.to_json(orient='records')})


@app.route("/write_dependency_report", methods=['GET', 'OPTIONS'])
def write_dependency_report():
    if outdated_packages_list(app_logger, config.prod_deployment_path):
        return jsonify('Dependency report written correctly')
    else:
        abort(500, "Could not write dependency report.")


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8000, debug=True, threaded=True)
