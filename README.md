[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-380/)
[![Scikit-learn 1.4.0](https://img.shields.io/badge/scikit_learn-1.4.0-blue.svg)](https://scikit-learn.org/0.24/whats_new/v0.24.html#version-0-24-1)
[![Flask 3.0.2](https://img.shields.io/badge/flask-3.0.2-blue.svg)](https://pypi.org/project/Flask/)

# Dynamic Risk Assessment System

Udacity Machine Learning DevOps Engineer project.

REPO:  https://github.com/dbaeza0/DRASystem/tree/master

## Project Overview

### Background

Case of a company concerned about attrition risk: the risk that some of their clients will exit their contracts and 
decrease the company's revenue. If the client management team is small they're not able to stay in close 
contact with all their clients.

The company could find helpful to create, deploy, and monitor a risk assessment ML model that will estimate the 
attrition risk of each of the company's clients. If the model created and deployed is accurate, it will enable the 
client managers to contact the clients with the highest risk and avoid losing clients and revenue.

Creating and deploying the model isn't the end of our work, though. Most industries are dynamic and constantly changing, 
and a model that was created a year or a month ago might not still be accurate today. Because of this, we need to 
set up regular monitoring of the model to ensure that it remains accurate and up-to-date. We need to set up processes 
and scripts to re-train, re-deploy, monitor, and report on th ML model, so that the company can get risk assessments 
that are as accurate as possible and minimize client attrition.

## Running

## Requirements

- Python 3.11
- Poetry

## Installation

To install Poetry, you can follow the instructions on the official website: [Poetry Installation Guide](https://python-poetry.org/docs/#installation).

Once Poetry is installed, navigate to your project directory and run the following command to initialize a new Poetry project:

```bash
poetry install
```

- We need to define the folder that will be used for storing new data and saving the model. For that purpose, we need 
to define in the **config.json** file:
  - `input_folder_path` entry for storing new data
  - `output_model_path` entry for storing production models
  - Executing **wsgi.py** will start the API:
  ```bash
  python3 app.py
  ```
- We then run **fullprocess.py**:
  ```bash
  python3 fullprocess.py
  ```

