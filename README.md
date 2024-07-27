# Car Price Regressor
This project aims to predict the selling price of cars based on various features such as the car's name, year of manufacture, kilometers driven, fuel type, seller type, transmission, owner, mileage, engine, max power, and seats. 
The project involves data preprocessing, exploratory data analysis, feature engineering, model training, and deployment using FastAPI.

![Image about the final project](<Car Price Prediction.png>)

## Prerequisites
To run this project, you will need the following libraries and tools:

- Python 3.7+
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- joblib
- FastAPI
- uvicorn
- Jinja2

## Overview of the Code
The project consists of several parts:

1. Data Loading and Preprocessing:
- Load the dataset and perform initial data exploration.
- Handle missing values and invalid data.
- Convert categorical features to numerical using Label Encoding.
- Handle skewness and outliers in the data.

2. Exploratory Data Analysis (EDA):
- Visualize the distribution of features.
- Analyze correlations between features and the target variable.

3. Model Training:
- Split the data into training and testing sets.
- Train multiple regression models (Linear Regression, Ridge, Lasso, ElasticNet, Decision Tree, Gradient Boosting, K-Nearest Neighbors, Random Forest).
- Evaluate model performance using Mean Squared Error (MSE) and select the best model.

4. Model Saving:
- Save the best trained model and label encoders using joblib.

5. FastAPI Integration:
- Create a FastAPI application to serve the model.
- Define endpoints for home and prediction.
- Load the saved model and label encoders.
- Preprocess input data and make predictions using the model.


## Model Accuracy
The Random Forest model achieved the best performance with a test accuracy of 97.6%. 
The features with the highest importance in the model were max_power, engine, and year.


## Contributions
Contributions to this project are welcome! If you have any suggestions, bug fixes, or improvements, please feel free to open an issue or submit a pull request.

