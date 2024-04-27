import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import r_regression
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error,  mean_absolute_percentage_error
def lin_regres():
    #task 1.1
    df = pd.read_csv("lab3_lin4.csv")
    print(df.head())
    #task 1.2
    x1_split = np.array(df["x1"]).reshape(-1, 1)
    x2_split = np.array(df["x2"]).reshape(-1, 1)
    y_split = np.array(df["y"]).reshape(-1, 1)

    x1_train, x1_test, x2_train,x2_test, y_train, y_test = (
        train_test_split(x1_split,x2_split, y_split, test_size=0.3))
    plt.scatter(x1_train, y_train, c="b", marker = ".")
    plt.scatter(x1_test, y_test, c="r", marker=".")
    plt.show()
    plt.scatter(x2_train, y_train, c="b", marker=".")
    plt.scatter(x2_test, y_test, c="r", marker=".")
    plt.show()

    #task 1.3
    lin_reg = LinearRegression()
    lin_reg.fit(x1_train, y_train)
    y_train_pred = lin_reg.predict(x1_train)
    y_test_pred = lin_reg.predict(x1_test)
    print(lin_reg.coef_, lin_reg.intercept_)
    lin_reg2 = LinearRegression()
    lin_reg2.fit(x2_train, y_train)
    y_train_pred2 = lin_reg2.predict(x2_train)
    y_test_pred2 = lin_reg2.predict(x2_test)
    print(lin_reg2.coef_, lin_reg2.intercept_)
    #
    # #task 1.4

    # xp_m = np.mean(df["x1"])
    # yp_m = np.mean(df["y"])
    # ap = (df["x1"] - xp_m) @ (df["y"] - yp_m) / ((df["x1"] - xp_m) @ (df["x1"] - xp_m))
    # bp = yp_m - ap * xp_m

    xp_line = np.array([np.min(np.array(df["x1"])), np.max(np.array(df["x1"]))]).reshape(-1,1)
    yp_line = lin_reg.predict(xp_line)
    plt.scatter(df["x1"], df["y"])
    plt.plot(xp_line, yp_line, 'r-')
    plt.grid()
    plt.show()

    print(r2_score(y_train, y_train_pred))
    print(r2_score(y_test, y_test_pred))
    #print(r2_score(df["y"], ap * df["x1"] + bp))


    print("mae y_train: "+str(mean_absolute_error(y_train, y_train_pred)))
    print("mae y_test: "+str(mean_absolute_error(y_test, y_test_pred)))
    print("mape y_train: "+str(mean_absolute_percentage_error(y_train, y_train_pred) * 100))
    print("mape y_test: "+str(mean_absolute_percentage_error(y_test, y_test_pred) * 100))

    print(r2_score(y_train, y_train_pred2))
    print(r2_score(y_test, y_test_pred2))
    print("mae y_train2: "+str(mean_absolute_error(y_train, y_train_pred2)))
    print("mae y_test2: "+str(mean_absolute_error(y_test, y_test_pred2)))
    print("mape y_train2: "+str(mean_absolute_percentage_error(y_train, y_train_pred2) * 100))
    print("mape y_test2: "+str(mean_absolute_percentage_error(y_test, y_test_pred2) * 100))




# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    lin_regres()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
