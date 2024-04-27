import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import r_regression
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
def lin_regres():
    #task 1.1
    df = pd.read_csv("lab3_lin4.csv")
    print(df.head())
    #task 1.2
    x1_split = np.array(df["x1"]).reshape(-1, 1)
    x2_split = np.array(df["x2"]).reshape(-1, 1)
    y_split = np.array(df["y"]).reshape(-1, 1)
    x1_train, x1_test, x2_train, x2_test, y_train, y_test = (
        train_test_split(x1_split,x2_split, y_split, test_size=0.3, shuffle=False))
    plt.scatter(x1_train, y_train, c="b", marker = ".")
    plt.scatter(x1_test, y_test, c="r", marker=".")
    plt.show()
    plt.scatter(x2_train, y_train, c="b", marker=".")
    plt.scatter(x2_test, y_test, c="r", marker=".")
    plt.show()
    #task 1.3
    lin_reg2 = LinearRegression()
    lin_reg2.fit(x1_train, y_train)
    yp_train_pred = lin_reg2.predict(x1_train)
    yp_test_pred = lin_reg2.predict(x1_test)
    print(lin_reg2.coef_, lin_reg2.intercept_)



# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    lin_regres()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
