# Asteroid MOID Prediction Using Regression

## 1. Project Overview

This project uses machine learning regression to predict the
Minimum Orbit Intersection Distance (MOID) of an asteroid with
Earth using physical and orbital characteristics.

The main research question is:

> Can multiple physical and orbital characteristics of an asteroid
> be used to predict its Minimum Orbit Intersection Distance with Earth?

The project uses multiple input variables to predict one continuous
numeric target.

---

## 2. Target Variable

The target variable is:

### MOID

MOID stands for Minimum Orbit Intersection Distance.

It represents the minimum distance between an asteroid's orbit and
Earth's orbit.

The target is measured in:

**Astronomical Units (AU)**

The web application also converts the predicted value to kilometers.

Important:

MOID is not the same as impact probability.

A small MOID does not automatically mean that an asteroid will
collide with Earth.

---

## 3. Input Variables

The model uses eight input variables.

| Variable | Description | Unit |
|---|---|---|
| H | Absolute Magnitude | magnitude |
| albedo | Surface reflectivity | ratio |
| e | Orbital eccentricity | dimensionless |
| a | Semi-major axis | AU |
| i | Orbital inclination | degrees |
| om | Longitude of ascending node | degrees |
| w | Argument of periapsis | degrees |
| ma | Mean anomaly | degrees |

The model therefore uses:

8 input variables

↓

1 continuous target

↓

MOID

---

## 4. Removed Variables

Several variables were excluded from the final feature set.

### moid_ld

`moid_ld` represents MOID in lunar-distance units.

It contains the same information as the target variable and would
therefore cause target leakage.

### q

Perihelion distance is mathematically related to semi-major axis
and eccentricity:

q = a(1-e)

Therefore it is redundant when both `a` and `e` are already used.

### ad

Aphelion distance is related to semi-major axis and eccentricity:

ad = a(1+e)

Therefore it is also redundant.

### n

Mean motion has a strong mathematical relationship with the orbital
semi-major axis.

### per / per_y

Orbital period is strongly related to the semi-major axis.

### ID variables

Identifiers such as object IDs, names, and orbit IDs were not used
as predictive variables because they identify observations rather
than represent physical characteristics.

---

## 5. Data Preprocessing

The dataset was processed using the following steps:

1. Load the CSV dataset.
2. Verify required columns.
3. Convert required variables to numeric format.
4. Handle invalid target values.
5. Check duplicate rows.
6. Check missing values.
7. Use median imputation for missing feature values.
8. Split the data into training and testing sets.
9. Fit preprocessing only using the training data.
10. Evaluate the final model on the unseen test set.

The train/test split uses:

80% training data

20% testing data

---

## 6. Exploratory Data Analysis

The project performs exploratory data analysis including:

- Descriptive statistics
- Missing-value analysis
- Target distribution
- Feature distributions
- Correlation matrix
- Feature versus MOID plots
- Outlier inspection

The generated EDA plots are stored in:

`outputs/`

---

## 7. Regression Models

Several regression algorithms are compared:

### Linear Regression

Used as a baseline model.

It provides a simple reference for determining whether nonlinear
models provide an improvement.

### Random Forest Regressor

An ensemble of decision trees that can model nonlinear relationships.

### Gradient Boosting Regressor

Builds an ensemble of weak learners sequentially and focuses on
reducing prediction errors.

### HistGradientBoostingRegressor

A histogram-based gradient boosting model designed to be efficient
on large datasets.

---

## 8. Evaluation Metrics

The models are evaluated using:

### MAE

Mean Absolute Error.

Lower values are better.

MAE represents the average absolute prediction error.

---

### RMSE

Root Mean Squared Error.

Lower values are better.

RMSE gives greater weight to large prediction errors.

---

### R²

Coefficient of determination.

Higher values are better.

R² measures how much of the variation in the target variable is
explained by the model.

---

## 9. Model Selection

The models are compared using the test set.

The model with the best overall test performance is selected as the
final model.

The comparison is saved to:

`outputs/model_comparison.csv`

---

## 10. Error Analysis

The project performs error analysis using:

- Actual versus predicted MOID
- Residual plots
- Absolute prediction error
- Largest prediction errors

The largest prediction errors are saved to:

`outputs/prediction_errors.csv`

---

## 11. Feature Importance

For models that support feature importance, the project calculates
the importance of each input feature.

The result is saved to:

`outputs/feature_importance.csv`

and:

`outputs/feature_importance.png`

---

## 12. Model Saving

The final trained model is saved as:

`model/moid_model.joblib`

The preprocessing steps are included in the saved pipeline.

The web application loads this saved model rather than retraining
the model every time a user makes a prediction.

---

## 13. Web Application

The web application is built using Gradio.

The user enters:

- H
- Albedo
- Eccentricity
- Semi-major Axis
- Inclination
- Longitude of Ascending Node
- Argument of Periapsis
- Mean Anomaly

The application outputs:

- Predicted MOID in AU
- Predicted MOID in kilometers

---

## 14. How to Run

Install dependencies:

```bash
pip install -r requirements.txt