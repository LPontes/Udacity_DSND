# Disaster Response Pipeline Project

#edit by LMPontes
### Table of Contents

1. [Project Overview](#overview)
  - [ETL Pipeline](#etl_pipeline)
  - [ML Pipeline](#ml_pipeline)
  - [Flask Web App](#flask)
  - [Instructions](#instructions)
2. [Installation](#installation)
3. [File Descriptions](#files)
4. [Results](#results)
5. [Licensing, Authors, and Acknowledgements](#licensing)


## Project Overview<a name="overview"></a>

For this project we will analyze data from disaster data from Figure Eight to build a model for an API that classifies disaster messages. The project include a web app where an emergency worker can input a new message and get classification results in several categories. The web app also displays visualizations of the data. 

## ETL Pipeline<a name="etl_pipeline"></a>

- Loads the `messages` and `categories` dataset
- Merges the two datasets
- Cleans the data
- Stores it in a **SQLite database**

## ML Pipeline<a name="ml_pipeline"></a>

- Loads data from the **SQLite database**
- Splits the data into training and testing sets
- Builds a text processing and machine learning pipeline
- Trains and tunes a Random Forest Classifier
- Exports the fitted model

## Flask Web App<a name="flask"></a>

Running python run.py command from app directory will start the web app where one can enter their message related to a natural disaster event.

### Instructions<a name="instructions"></a>

1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/

## Installation <a name="installation"></a>

This project uses Python 3.7 and the requirements includes the following packages: Flask, nltk, numpy, pandas, plotly, scikit-learn and SQLAlchemy.

## File Descriptions <a name="files"></a>


<pre>
.
├── app
│   ├── run.py------------------------# FLASK FILE THAT RUNS APP
│   ├── static
│   │   └── favicon.ico---------------# FAVICON FOR THE WEB APP
│   └── templates
│       ├── go.html-------------------# CLASSIFICATION RESULT PAGE OF WEB APP
│       └── master.html---------------# MAIN PAGE OF WEB APP
├── data
│   ├── DisasterResponse.db-----------# DATABASE TO SAVE CLEANED DATA TO
│   ├── disaster_categories.csv-------# DATA TO PROCESS
│   ├── disaster_messages.csv---------# DATA TO PROCESS
│   └── process_data.py---------------# PERFORMS ETL PROCESS
├── img-------------------------------# PLOTS FOR USE IN README AND THE WEB APP
├── models
│   └── train_classifier.py-----------# PERFORMS CLASSIFICATION TASK

</pre>

<a id='sw'></a>

## Results<a name="results"></a>

The Random Forest Classifier achieved a good precision. However, due to the imbalanced data, some of the categories presented low recall score.

## Licensing, Authors, Acknowledgements<a name="licensing"></a>

Must give credit to Figure Eight for the opportunity to work with such interesting and relevant data


