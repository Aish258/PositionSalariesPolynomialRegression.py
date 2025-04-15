💼 Polynomial Regression on Position Salaries Dataset
This project demonstrates the use of Polynomial Regression to model non-linear relationships between job position levels and corresponding salaries.

🎯 Objective
To build a regression model that accurately predicts Salary based on the Position Level using Polynomial Regression, and compare it with Simple Linear Regression.

🗂️ Dataset
The dataset used (Position_Salaries.csv) includes:

Position – Job title (e.g., Business Analyst, Manager, etc.)

Level – Numerical value representing the hierarchy of the position

Salary – Salary corresponding to the position level

This dataset is commonly used to demonstrate how Polynomial Regression can model non-linear trends more effectively than Linear Regression.

🛠️ Technologies Used
Python 🐍

Pandas

NumPy

Matplotlib & Seaborn (Visualization)

Scikit-learn (Modeling)

📝 Workflow
Data Loading & Exploration

Visualizing the Data

Applying Linear Regression (for comparison)

Applying Polynomial Regression

Choosing the optimal degree for the polynomial

Fitting the model and visualizing results

Prediction

Predict salary for a given position level

Model Evaluation

Comparing Linear vs Polynomial fit visually

📈 Results
Linear Regression failed to capture the salary curve properly.

Polynomial Regression (with an appropriate degree) provided a better fit for the observed non-linear relationship in the data.

