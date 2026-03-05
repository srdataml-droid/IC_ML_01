Weather Prediction with Machine Learning & LSTM

A machine learning project that explores historical weather data to perform classification, regression, and time-series forecasting using classical ML models and deep learning.

The project demonstrates how weather data can be processed, analyzed, and modeled to predict temperature and precipitation patterns using Scikit-Learn and TensorFlow.

Project Overview

This project investigates a historical weather dataset and applies multiple machine learning techniques to extract patterns and build predictive models.

Three modeling tasks are explored:

1. Classification

Predict the type of precipitation (rain or snow).

2. Regression

Predict the temperature (°C) using traditional regression models.

3. Time Series Forecasting

Predict future temperature values using an LSTM neural network designed for sequential data.

Dataset

The dataset contains hourly weather observations including atmospheric and environmental variables.

Key Features
Feature	Description
Formatted Date	Timestamp of weather observation
Temperature (C)	Recorded temperature
Apparent Temperature (C)	Feels-like temperature
Humidity	Relative humidity
Wind Speed (km/h)	Wind speed
Wind Bearing (degrees)	Wind direction
Visibility (km)	Visibility distance
Pressure (millibars)	Atmospheric pressure
Precip Type	Type of precipitation (rain or snow)
Summary	Short weather description

The dataset used:

weatherHistory.csv
Project Structure
weather-ml-project/
|
|__ archive.zip           # zip files
├── IC_ML_01.ipynb        # Main machine learning notebook
├── weatherHistory.csv    # Weather dataset
└── README.md             # Project documentation
Workflow

The notebook follows a standard machine learning pipeline.

1. Data Loading

The dataset is loaded using Pandas.

pd.read_csv("weatherHistory.csv")
2. Data Cleaning

The following preprocessing steps were applied:

Converted Formatted Date into datetime format

Extracted time features:

year

month

day

hour

Removed irrelevant columns:

Loud Cover

Daily Summary

Handled missing values in Precip Type

Encoded precipitation labels:

rain → 1

snow → 0

3. Exploratory Data Analysis

Basic EDA was performed to understand:

feature distributions

outliers

correlations between weather variables

A boxplot was used to inspect outliers in Wind Speed.

Machine Learning Models
Classification Models

Goal: Predict precipitation type.

Models used:

Logistic Regression

Random Forest Classifier

Evaluation metrics:

Accuracy

Precision

Recall

F1 Score

Classification Report

Regression Models

Goal: Predict temperature (°C).

Models used:

Linear Regression

Random Forest Regressor

Evaluation metric:

Mean Squared Error (MSE)

Deep Learning Model

For time-series prediction, an LSTM (Long Short-Term Memory) network was implemented using TensorFlow / Keras.

Model Architecture
LSTM
 ↓
Dense Layer
 ↓
Temperature Prediction

The model was trained using sequences generated with:

timeseries_dataset_from_array()

This allows the model to learn patterns from sequential weather observations.

Technologies Used
Tool	Purpose
Python	Core programming language
Pandas	Data manipulation
NumPy	Numerical operations
Matplotlib	Data visualization
Seaborn	Statistical visualization
Scikit-Learn	Machine learning models
TensorFlow / Keras	Deep learning (LSTM)
Installation

Install required dependencies:

pip install numpy pandas matplotlib seaborn scikit-learn tensorflow
Running the Project

Clone the repository

git clone <repo-url>

Navigate to the project folder

cd weather-ml-project

Launch Jupyter Notebook

jupyter notebook

Open

IC_ML_01.ipynb

Run all cells to reproduce the results.

Future Improvements

Possible extensions to the project include:

Hyperparameter tuning for models

Feature scaling and advanced preprocessing pipelines

Incorporating additional weather datasets

Deploying the model as an API

Building a dashboard for temperature forecasting

Author

Machine Learning experiment exploring classification, regression, and time-series forecasting using weather data.
