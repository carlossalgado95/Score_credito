# CREDIT SCORE PREDICTION ML MODELS

### Let's create a machine learning prediction model to predict customer credit score

## Project Overview
<p style="font-size: 18px;">
    In this project, we aim to predict the credit score of customers using machine learning models. The workflow involves various stages of data preprocessing, model building, testing, and evaluation. This includes handling missing data, dealing with outliers, and performing feature engineering, data standardization, and machine learning model evaluation.
</p>

### Key Steps Involved
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

### Libraries and Dependencies
This project utilizes the following libraries:

- **Pandas**: Data manipulation and analysis.
- **Matplotlib & Seaborn**: Data visualization.
- **NumPy**: Numerical computing.
- **Scikit-learn**: Machine learning model building and evaluation.
- **XGBoost**: Gradient boosting model.

```python
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

Model Performance
<p style="font-size: 20px; font-weight: bold;"> Here are the performance results of the models used: </p> <ul> <li><strong>Linear Regression</strong>: <ul> <li>R2 Score (Training): <strong>0.7957</strong></li> <li>R2 Score (Test): <strong>0.7941</strong></li> <li>R2 Score (Cross-validation): <strong>0.7938 ± 0.0197</strong></li> </ul> </li> <li><strong>Random Forest</strong>: <ul> <li>R2 Score (Training): <strong>1.0000</strong></li> <li>R2 Score (Test): <strong>1.0000</strong></li> <li>R2 Score (Cross-validation): <strong>1.0000 ± 0.0000</strong></li> </ul> </li> <li><strong>XGBoost</strong>: <ul> <li>R2 Score (Training): <strong>1.0000</strong></li> <li>R2 Score (Test): <strong>1.0000</strong></li> <li>R2 Score (Cross-validation): <strong>1.0000 ± 0.0000</strong></li> </ul> </li> </ul>
Conclusion
<p style="font-size: 18px;"> The Random Forest and XGBoost models performed exceptionally well with perfect R2 scores on both training and testing datasets. These models also showed great consistency during cross-validation. Linear Regression showed good performance as well, with slightly lower but reasonable R2 scores. </p>
