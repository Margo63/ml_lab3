import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import r_regression
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error,  mean_absolute_percentage_error
from sklearn.preprocessing import PolynomialFeatures
import tensorflow as tf
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



def not_lin_regression():
    #task 2.1
    df = pd.read_csv("lab3_poly1.csv")
    print(df.head())
    plt.scatter(df["x"],df["y"], c="b", marker = ".")
    plt.show()

    #task 2.2
    x_split = np.array(df["x"]).reshape(-1, 1)
    y_split = np.array(df["y"]).reshape(-1, 1)

    x_train, x_test,  y_train, y_test = (
        train_test_split(x_split,  y_split, test_size=0.3))
    plt.scatter(x_train, y_train, c="b", marker=".")
    plt.scatter(x_test, y_test, c="r", marker=".")
    plt.show()

    #task 2.3
    lin = LinearRegression()
    lin.fit(x_train, y_train)
    y_train_pred = lin.predict(x_train)
    y_test_pred = lin.predict(x_test)
    print(lin.coef_, lin.intercept_)

    plt.plot(x_split, y_split, 'k.')
    plt.plot(np.array(x_train), np.array(y_train_pred), 'r-')
    plt.grid()
    plt.title('degree = 1')
    plt.show()

    #task 2.4
    # 2 -> 4 -> 8
    x2 = PolynomialFeatures(8).fit_transform(x_train)
    lin2 = LinearRegression(fit_intercept=False)
    lin2.fit(x2, y_train)
    print(lin2.coef_, lin2.intercept_)
    c2 = lin2.coef_[0]
    y2_pred = (c2[0] + c2[1] * x_train + c2[2] * x_train * x_train +c2[3] * x_train * x_train* x_train
               + c2[4] * x_train * x_train * x_train * x_train + c2[5] * x_train * x_train * x_train * x_train* x_train
               + c2[6] * x_train * x_train * x_train * x_train * x_train * x_train
               + c2[7] * x_train * x_train * x_train * x_train * x_train * x_train * x_train
               + c2[8] * x_train * x_train * x_train * x_train * x_train * x_train * x_train * x_train)

    y2_pred_test = (c2[0] + c2[1] * x_test + c2[2] * x_test * x_test + c2[3] * x_test * x_test * x_test
               + c2[4] * x_test * x_test * x_test * x_test + c2[5] * x_test * x_test * x_test * x_test * x_test
               + c2[6] * x_test * x_test * x_test * x_test * x_test * x_test
               + c2[7] * x_test * x_test * x_test * x_test * x_test * x_test * x_test
               + c2[8] * x_test * x_test * x_test * x_test * x_test * x_test * x_test * x_test)



    #task 2.5
    #>1/2 значит все хорошо
    print(r2_score(y_train, y2_pred))
    print(r2_score(y_test, y2_pred_test))
    print("mae y_train: "+str(mean_absolute_error(y_train, y2_pred)))
    print("mae y_test: "+str(mean_absolute_error(y_test, y2_pred_test)))
    print("mape y_train: "+str(mean_absolute_percentage_error(y_train, y2_pred) * 100))
    print("mape y_test: "+str(mean_absolute_percentage_error(y_test, y2_pred_test) * 100))

    #task 2.6
    plt.plot(x_train, y_train, 'k.')
    plt.plot(x_train, y2_pred, 'r.')
    plt.plot(x_test, y2_pred_test, 'b.')
    plt.grid()
    plt.title('degree = 8')
    plt.show()




def model_regression():
    #task 3.1
    df = pd.read_csv("Student_Performance.csv")
    print(df.head().to_string())

    #task 3.2
    df['Extracurricular Activities'] = df['Extracurricular Activities'].replace(["Yes"], 1)
    df['Extracurricular Activities'] = df['Extracurricular Activities'].replace(["No"], 0)
    print(df.head().to_string())

    drop_df = df.drop_duplicates()
    print(drop_df.describe().to_string())

    df_not_null = drop_df.dropna()
    print(df_not_null.describe().to_string())





# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    #lin_regres()
    not_lin_regression()

