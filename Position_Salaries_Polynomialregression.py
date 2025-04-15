#POLYNOMIAL REGRESSION
#Equation m1*x+m2*x square+m3*x cube+m4*x power four+..........+c
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
link="C:\\Users\\Aishwarya\\Downloads\\4_Position_Salaries.csv"
df=pd.read_csv(link)
print(df)
print("Shape:",df.shape)
print("columns:",df.columns)

x=df.iloc[:,1:2].values
y=df.iloc[:,2].values
print("Values of x:",x)
print("Values of y:",y)

#Performing EDA(Explotatory data analysis) to understand the data
#Scatter plot
plt.scatter(df['Level'],df['Salary'])
plt.xlabel('Level')
plt.ylabel('Salary')
plt.title('Scatter plot of Level vs Salary')
plt.show()
#There is Positive corelation between X(R&D Spend) and y(Profit) so this is an example of Linear Regression Model

#Model building
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
outcome=[]
for i in range (1,6):
    parameters = [('Polynomial', PolynomialFeatures(degree=i)), ('model', LinearRegression())]
    regressor = Pipeline(parameters)

    # train the data
    regressor.fit(x, y)

    # predict x
    y_pred=regressor.predict(x)
    outcome.append(y_pred)
    out_df = pd.DataFrame({'Actual': y, 'Predicted': y_pred})
    print('Actual vs Predicted:\n', out_df)

    # Model Evaluation
    from sklearn import metrics

    mse = metrics.mean_squared_error(y, y_pred)
    rmse = mse ** 0.5
    mae = metrics.mean_absolute_error(y, y_pred)
    print("==========================")
    print("Degree:",i)
    print("Mean Squared Error=", mse)
    print("Root Mean Squared Error=", rmse)
    print("Mean Absolute error=", mae)


    # Calculating R-Squared value
    # This value can be maximum 1(Good fit) and minimum 0 (Bad fit) [R-squared=1-(MSE REGRESSION-MSE AVERAGE)]-Formula
    R_Squared = metrics.r2_score(y, y_pred)
    print("R-squared value=", R_Squared)

plt.figure(figsize=(10,6))
plt.scatter(x,y,s=15)
plt.plot(x,outcome[0],color="red",label="Degree 1")
plt.plot(x,outcome[2],color="blue",label="Degree 3")
plt.plot(x,outcome[4],color="green",label="Degree 5")
plt.show()

