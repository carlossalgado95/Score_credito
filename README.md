## CREDIT SCORE PREDICTION ML MODELS


Let's create a machine learning prediction model to predict customer credit score
Project Overview

In this project, we aim to predict the credit score of customers using machine learning models. The workflow involves various stages of data preprocessing, model building, testing, and evaluation. This includes handling missing data, dealing with outliers, and performing feature engineering, data standardization, and machine learning model evaluation.
#### Key Steps Involved
<ul>
    <li><strong>Exploratory Data Analysis (EDA) and Graphs</strong>: Visualizing the data to understand distributions, relationships, and potential issues such as missing data and outliers.</li>
    <li><strong>Missing Data Handling</strong>: Identifying and managing missing values in the dataset.</li>
    <li><strong>Outlier Handling</strong>: Detecting and addressing outliers that may distort model training.</li>
    <li><strong>OneHotEncoding</strong>: Converting categorical variables into numerical format for model compatibility.</li>
    <li><strong>Attribute Engineering</strong>: Creating new features to improve model performance.</li>
    <li><strong>Data Handling</strong>: Cleaning and organizing data to ensure it's ready for machine learning models.</li>
    <li><strong>Data Standardization</strong>: Scaling the data to bring all features into a common range, especially important for algorithms like Linear Regression.</li>
    <li><strong>Machine Learning Model Building, Testing, and Validation</strong>: Implementing and validating multiple models, including Linear Regression, Random Forest, and XGBoost.</li>
</ul>
#### Libraries and Dependencies

This project utilizes the following libraries:

    -<strong>Pandas</strong>: Data manipulation and analysis.
    -<strong>Matplotlib & Seaborn</strong>: Data visualization.
    -<strong>NumPy</strong>: Numerical computing.
    -<strong>Scikit-learn</strong>: Machine learning model building and evaluation.
    -<strong>XGBoost</strong>: Gradient boosting model.

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.pipeline import Pipeline

#### Model Performance

Here are the performance results of the models used:

    Linear Regression:
        R2 Score (Training): 0.7957
        R2 Score (Test): 0.7941
        R2 Score (Cross-validation): 0.7938 ± 0.0197

    Random Forest:
        R2 Score (Training): 1.0000
        R2 Score (Test): 1.0000
        R2 Score (Cross-validation): 1.0000 ± 0.0000

    XGBoost:
        R2 Score (Training): 1.0000
        R2 Score (Test): 1.0000
        R2 Score (Cross-validation): 1.0000 ± 0.0000

#### Conclusion

The Random Forest and XGBoost models performed exceptionally well with perfect R2 scores on both training and testing datasets. These models also showed great consistency during cross-validation. Linear Regression showed good performance as well, with slightly lower but reasonable R2 scores.
How to Run the Project

#### Clone the repository:

git clone https://github.com/carlossalgado95/Score_credito.git

Install the required dependencies:

    pip install -r requirements.txt

    Run the Jupyter notebook or Python scripts to reproduce the analysis and model training.

#### License

This project is licensed under the MIT License - see the LICENSE file for details.
