# House Price Prediction Project

This project aims to predict house prices using a dataset containing various features related to housing. The project uses Python for data manipulation, visualization, and machine learning to build a predictive model.

## Table of Contents

1. [Introduction](#introduction)
2. [Dataset](#dataset)
3. [Installation](#installation)
4. [Project Workflow](#project-workflow)
5. [Modeling and Evaluation](#modeling-and-evaluation)
6. [Results](#results)
7. [Conclusion](#conclusion)
8. [License](#license)

## Introduction

This project demonstrates the process of building a machine learning model to predict house prices based on various features such as the number of rooms, location, and other housing characteristics. The model development follows the standard data science workflow, including data exploration, preprocessing, feature selection, model training, and evaluation.

## Dataset

The dataset used in this project is stored in the `Dataset` folder. It contains several features that are used to predict house prices.

### Data Features

- **Numerical Features**: Examples include square footage, number of bedrooms, number of bathrooms, etc.
- **Categorical Features**: Examples include the type of property, location, etc.

## Installation

To run this project, you need Python installed along with the necessary libraries. You can install the required libraries using the following command:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

## Project Workflow

The project workflow consists of the following steps:

1. **Importing Libraries**: 
   ```python
   import pandas as pd
   import numpy as np
   import matplotlib.pyplot as plt
   import seaborn as sns
   from sklearn.model_selection import train_test_split
   from sklearn.linear_model import LinearRegression
   from sklearn.metrics import mean_squared_error
   ```
The dataset is loaded using `pandas.read_csv()` from the `Dataset` folder. This data contains various attributes related to housing.

3. **Data Exploration**:
   - **Data Visualization**: Visualize the distribution of numerical and categorical features using histograms and bar charts.
     ```python
     df.hist(bins=50, figsize=(20, 15))
     plt.show()
     ```
   - **Correlation Analysis**: Explore correlations between different features and the target variable (house prices) using a heatmap.
     ```python
     corr_matrix = df.corr()
     sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
     plt.show()
     ```

4. **Data Preprocessing**:
   - **Handling Missing Values**: Handle missing values by filling them with median values.
     ```python
     df.fillna(df.median(), inplace=True)
     ```
   - **Encoding Categorical Variables**: Convert categorical variables to numeric using one-hot encoding.
     ```python
     df = pd.get_dummies(df, drop_first=True)
     ```

5. **Feature Selection**:
   - Identify and select the most relevant features for the model.
   - This step involves removing irrelevant or redundant features based on domain knowledge or statistical tests.

6. **Model Training**:
   - **Split the Data**: Split the dataset into training and testing sets using `train_test_split`.
     ```python
     X = df.drop('price', axis=1)
     y = df['price']
     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
     ```
   - **Train a Linear Regression Model**: Train a Linear Regression model using the training data.
     ```python
     model = LinearRegression()
     model.fit(X_train, y_train)
     ```
