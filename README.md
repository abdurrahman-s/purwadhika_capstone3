# California Housing Price Prediction - Machine Learning Project
## Section 1: Introduction

- Problem Statement: As a property company, our mission is to strike a balance between maximizing profit and maintaining high customer satisfaction for both buyers and sellers. To that end, we aim to build a predictive model that computes the optimal house price based on various factors such as location, house age, and more. This not only ensures profitable returns but also enhances customer satisfaction—a win-win scenario for all stakeholders involved.

- Goals: The objective of this project is to develop a machine learning model that predicts the median_house_value, our target variable, using various features provided in the dataset.

- Metric Evaluation: The performance of the model is evaluated using the following metrics: Mean Squared Error (MSE), Mean Absolute Error (MAE), Mean Absolute Percentage Error (MAPE), and Mean Cross-Validation Score.

## Section 2: Data Preparation
Data preparation includes loading the dataset, checking for missing values and duplicate rows, and handling any discovered inconsistencies. Missing values are dealt with using an unsupervised learning approach based on the k-Nearest Neighbors (kNN) model.

## Section 3: Exploratory Data Analysis
To better understand the dataset, exploratory data analysis is performed using techniques such as histograms, a correlation matrix, and geospatial plots. These tools provide insights into the data distribution, outliers, and correlations between variables.

## Section 4: Feature Engineering
Feature engineering is performed to create new features that could potentially enhance the model's performance. Some of the new features include diag_coord (diagonal coordinate), bed_per_room (ratio of bedrooms to total rooms), room_per_household (ratio of total rooms to households), and population_per_household (ratio of population to households).

## Section 5: Outlier Handling
Outliers are handled in this stage to ensure they do not adversely affect the performance of the predictive model.

## Section 6: Modelling

The modelling stage involves several steps:

- Encoding: The categorical variables are encoded to prepare them for the machine learning algorithms.
- Feature Selection: Feature importance is evaluated to select the most impactful features for model training.
- Model Selection: Various algorithms are tried, including Linear Regression, RandomForest Regressor, XGBoost Regressor, Lasso, and Ridge.
- Hyperparameter Tuning: RandomizedSearchCV is used for tuning the hyperparameters to optimize the model's performance.
- Feature Importance: The importance of different features in predicting the target variable is evaluated.

## Section 7: Conclusion and Recommendations

The model's performance, feature importance, and potential improvements are discussed in the conclusion. The findings offer valuable insights that could be used to enhance future iterations of the model. Recommendations for further improvement are also suggested.

## Prerequisites
To execute the project successfully, ensure that you have the following packages installed in your Python environment:
### Data Manipulation

- pandas: A software library for data manipulation and analysis. You can install it with pip install pandas.
- numpy: A package for scientific computing with Python. You can install it with pip install numpy.
- pickle: A module for serializing and de-serializing Python object structures. It's included in standard Python distribution.

### Visualization

- matplotlib: A Python 2D plotting library. You can install it with pip install matplotlib.
- seaborn: A statistical data visualization library based on matplotlib. You can install it with pip install seaborn.
- geopandas: A library for geospatial data operations. You can install it with pip install geopandas.
- geoplot: A high-level geospatial data visualization library for Python. You can install it with pip install geoplot.

### Preprocessing and Model Selection

- scikit-learn: A machine learning library for Python. It includes various algorithms for model training and utilities for preprocessing, model selection, etc. You can install it with pip install scikit-learn.

### Models

- xgboost: An optimized distributed gradient boosting library. You can install it with pip install xgboost.

### Suppress Warnings

- warnings: An inbuilt Python module for warning control. It's included in standard Python distribution.

Make sure that you have these packages installed and imported at the beginning of your notebook to ensure the smooth running of your project.
