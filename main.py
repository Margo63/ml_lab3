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
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import SGDRegressor
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
    plt.legend(('train','test'))
    plt.show()
    plt.scatter(x2_train, y_train, c="b", marker=".")
    plt.scatter(x2_test, y_test, c="r", marker=".")
    plt.legend(('train','test'))
    plt.show()

    #task 1.3
    lin_reg = LinearRegression()
    data_train = np.column_stack([x1_train, x2_train])
    data_test = np.column_stack([x1_test, x2_test])

    lin_reg.fit(data_train, y_train)
    y_train_pred = lin_reg.predict(data_train)
    y_test_pred = lin_reg.predict(data_test)
    corr = r_regression(np.column_stack([x1_split, x2_split]), y_split.ravel() )
    print(lin_reg.coef_, lin_reg.intercept_)
    print(corr)
    # lin_reg2 = LinearRegression()
    # lin_reg2.fit(x2_train, y_train)
    # y_train_pred2 = lin_reg2.predict(x2_train)
    # y_test_pred2 = lin_reg2.predict(x2_test)
    # corr2 = r_regression(x2_split, y_split.ravel())
    #
    # print(lin_reg2.coef_, lin_reg2.intercept_)
    # print(corr2)
    #
    #

    # xp_m = np.mean(df["x1"])
    # yp_m = np.mean(df["y"])
    # ap = (df["x1"] - xp_m) @ (df["y"] - yp_m) / ((df["x1"] - xp_m) @ (df["x1"] - xp_m))
    # bp = yp_m - ap * xp_m


    # xp_line = np.array([np.min(np.array(df["x1"])), np.max(np.array(df["x1"]))]).reshape(-1,1)
    # yp_line = lin_reg.predict(xp_line)
    # plt.scatter(x1_train, y_train, c='b', marker='.')
    # plt.scatter(x1_test, y_test, c='r', marker='.')
    # plt.plot(xp_line, yp_line, 'k--')
    # plt.scatter(x1_test, y_test_pred, c='r', marker='x')
    # plt.grid()
    # plt.show()
    # plt.legend(('train','test','predict line','test predict'))
    #
    # xp_line2 = x2_train*lin_reg2.coef_[0]+lin_reg2.intercept_
    # yp_line2 = lin_reg.predict(xp_line2)
    # plt.scatter(x2_train, y_train, c='b', marker='.')
    # plt.scatter(x2_test, y_test, c='r', marker='.')
    # plt.plot(xp_line2, yp_line2, 'k--')
    # plt.scatter(x2_test, y_test_pred2, c='r', marker='x')
    # plt.grid()
    # plt.show()
    # plt.legend(('train', 'test', 'predict line', 'test predict'))

    # task 1.4
    print(r2_score(y_train, y_train_pred))
    print(r2_score(y_test, y_test_pred))
    print("mae y_train: "+str(mean_absolute_error(y_train, y_train_pred)))
    print("mae y_test: "+str(mean_absolute_error(y_test, y_test_pred)))
    print("mape y_train: "+str(mean_absolute_percentage_error(y_train, y_train_pred) * 100))
    print("mape y_test: "+str(mean_absolute_percentage_error(y_test, y_test_pred) * 100))

    # print(r2_score(y_train, y_train_pred2))
    # print(r2_score(y_test, y_test_pred2))
    # print("mae y_train2: "+str(mean_absolute_error(y_train, y_train_pred2)))
    # print("mae y_test2: "+str(mean_absolute_error(y_test, y_test_pred2)))
    # print("mape y_train2: "+str(mean_absolute_percentage_error(y_train, y_train_pred2) * 100))
    # print("mape y_test2: "+str(mean_absolute_percentage_error(y_test, y_test_pred2) * 100))

    #task 1.5.1
    print("\nLASSO")
    print('*'*100)
    lasso = Lasso(0.1)
    lasso.fit(data_train, y_train)
    lasso_pred_train = lasso.predict(data_train)
    print(lasso.coef_, lasso.intercept_)
    lasso_pred_test = lasso.predict(data_test)
    print(r2_score(y_train, lasso_pred_train))
    print(r2_score(y_test, lasso_pred_test))
    print("mae y_train: " + str(mean_absolute_error(y_train, lasso_pred_train)))
    print("mae y_test: " + str(mean_absolute_error(y_test, lasso_pred_test)))
    print("mape y_train: " + str(mean_absolute_percentage_error(y_train, lasso_pred_train) * 100))
    print("mape y_test: " + str(mean_absolute_percentage_error(y_test, lasso_pred_test) * 100))
    print('*' * 100)

    #task 1.5.2
    print("\nRidge")
    print('*' * 100)
    ridge = Ridge(1)
    ridge.fit(data_train, y_train)
    print(ridge.coef_, ridge.intercept_)
    ridge_pred_train = ridge.predict(data_train)
    ridge_pred_test = ridge.predict(data_test)
    print(r2_score(y_train, ridge_pred_train))
    print(r2_score(y_test, ridge_pred_test))
    print("mae y_train: " + str(mean_absolute_error(y_train, ridge_pred_train)))
    print("mae y_test: " + str(mean_absolute_error(y_test, ridge_pred_test)))
    print("mape y_train: " + str(mean_absolute_percentage_error(y_train, ridge_pred_train) * 100))
    print("mape y_test: " + str(mean_absolute_percentage_error(y_test, ridge_pred_test) * 100))
    print('*' * 100)

    #task 1.5.3
    print("\nElasticNet")
    print('*' * 100)
    elasticNet = ElasticNet(0.1)
    elasticNet.fit(data_train, y_train)
    print(elasticNet.coef_, elasticNet.intercept_)
    elasticNet_pred_train = elasticNet.predict(data_train)
    elasticNet_pred_test = elasticNet.predict(data_test)
    print(r2_score(y_train, elasticNet_pred_train))
    print(r2_score(y_test, elasticNet_pred_test))
    print("mae y_train: " + str(mean_absolute_error(y_train, elasticNet_pred_train)))
    print("mae y_test: " + str(mean_absolute_error(y_test, elasticNet_pred_test)))
    print("mape y_train: " + str(mean_absolute_percentage_error(y_train, elasticNet_pred_train) * 100))
    print("mape y_test: " + str(mean_absolute_percentage_error(y_test, elasticNet_pred_test) * 100))
    print('*' * 100)

    #task 1.5.4
    print("\nGradient")
    print('*' * 100)
    gradient = SGDRegressor(alpha=0.001, max_iter=1000)
    gradient.fit(data_train, y_train)
    print(gradient.coef_, gradient.intercept_)
    gradient_pred_train = gradient.predict(data_train)
    gradient_pred_test = gradient.predict(data_test)
    print(r2_score(y_train, gradient_pred_train))
    print(r2_score(y_test, gradient_pred_test))
    print("mae y_train: " + str(mean_absolute_error(y_train, gradient_pred_train)))
    print("mae y_test: " + str(mean_absolute_error(y_test, gradient_pred_test)))
    print("mape y_train: " + str(mean_absolute_percentage_error(y_train, gradient_pred_train) * 100))
    print("mape y_test: " + str(mean_absolute_percentage_error(y_test, gradient_pred_test) * 100))
    print('*' * 100)


    all_data = np.column_stack([x1_split,x2_split])
    data_predict = ridge.predict(all_data)
    plt.plot(x1_split, y_split, 'b.')
    plt.plot(x1_split, data_predict, 'kx')
    plt.grid()
    plt.legend(('train', 'predict'))
    plt.show()
    plt.plot(x2_split, y_split, 'b.')
    plt.plot(x2_split, data_predict, 'kx')
    plt.grid()
    plt.legend(('train', 'predict'))
    plt.show()






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
    plt.legend(('train', 'test'))
    plt.show()

    #task 2.3
    lin = LinearRegression()
    lin.fit(x_train, y_train)
    y_train_pred = lin.predict(x_train)
    y_test_pred = lin.predict(x_test)
    print(lin.coef_, lin.intercept_)

    plt.plot(x_train, y_train, 'k.')
    plt.plot(np.array(x_test), np.array(y_test), 'b.')
    plt.plot(np.array(x_test), np.array(y_test_pred), 'r-')
    plt.grid()
    plt.title('degree = 1')
    plt.legend(('data', 'test',"predict"))
    plt.show()

    #task 2.4
    # 2 -> 4 -> 6
    for n in range(1,41):
        X_train = PolynomialFeatures(n).fit_transform(x_train)
        X_test = PolynomialFeatures(n).fit_transform(x_test)
        lin2 = LinearRegression(fit_intercept=False)
        lin2.fit(X_train, y_train)

        # x_l_demo = np.linspace(-2, 2, 100).reshape(-1, 1)
        # x_l_demo2 = PolynomialFeatures(n, include_bias=False).fit_transform(x_l_demo)
        # y_l_demo = lin2.predict(x_l_demo2)



        print(lin2.coef_, lin2.intercept_)
        c2 = lin2.coef_[0]
        #y2_pred = for
        #(c2[0] + c2[1] * x_train #+ c2[2] * x_train ** 2 +c2[3] * x_train **3
                   #+ c2[4] * x_train ** 4 + c2[5] * x_train ** 5
                   #+ c2[6] * x_train * x_train * x_train * x_train * x_train * x_train
                   #+ c2[7] * x_train * x_train * x_train * x_train * x_train * x_train * x_train
                   #+ c2[8] * x_train * x_train * x_train * x_train * x_train * x_train * x_train * x_train
                   #)

       # y2_pred_test = (c2[0] + c2[1] * x_test# + c2[2] * x_test * x_test + c2[3] * x_test * x_test * x_test
                   #+ c2[4] * x_test * x_test * x_test * x_test + c2[5] * x_test * x_test * x_test * x_test * x_test
                   #+ c2[6] * x_test * x_test * x_test * x_test * x_test * x_test
                   #+ c2[7] * x_test * x_test * x_test * x_test * x_test * x_test * x_test
                   #+ c2[8] * x_test * x_test * x_test * x_test * x_test * x_test * x_test * x_test
                   #     )
        plt.plot(x_train, y_train, 'k.')
        plt.plot(x_train, np.array(lin2.predict(X_train)), 'r.')
        plt.plot(x_test, np.array(lin2.predict(X_test)), 'b.')
        plt.grid()
        plt.title('degree = '+str(n))
        plt.legend(("train","train_predict","test_predict"))
        plt.show()


    #task 2.5
    #>1/2 значит все хорошо
    # print(r2_score(y_train, y2_pred))
    # print(r2_score(y_test, y2_pred_test))
    # print("mae y_train: "+str(mean_absolute_error(y_train, y2_pred)))
    # print("mae y_test: "+str(mean_absolute_error(y_test, y2_pred_test)))
    # print("mape y_train: "+str(mean_absolute_percentage_error(y_train, y2_pred) * 100))
    # print("mape y_test: "+str(mean_absolute_percentage_error(y_test, y2_pred_test) * 100))
    #
    # #task 2.6
    # plt.plot(x_train, y_train, 'k.')
    # plt.plot(x_train, y2_pred, 'r.')
    # plt.plot(x_test, y2_pred_test, 'b.')
    # plt.grid()
    # plt.title('degree = 5')
    # plt.show()
    #
    # # task 2.7
    # x_tf = tf.constant(np.array(x_train), dtype=tf.float32)
    # y_tf = tf.constant(np.array(y_train), dtype=tf.float32)
    #
    # # генерируем случайные параметры модели
    # w = [tf.Variable(np.random.randn()) for _ in range(6)]
    # print(*w)
    #
    # # скорость обучения
    # alpha = tf.constant(0.001, dtype=tf.float32)
    # # количество итераций (эпох)
    # epoch_n = 9000
    #
    # print(w[0]**6)
    # # цикл обучения
    # for epoch in range(epoch_n):
    #     with tf.GradientTape() as tape:
    #         # 0.5 * (X - 3) * (X - 2) + 4 * np.sin(X)
    #         #y_pred = w[0] * x_tf * x_tf* x_tf * x_tf * x_tf * x_tf* x_tf * x_tf  + w[1] * x_tf* x_tf * x_tf * x_tf * x_tf* x_tf * x_tf  + w[2] * x_tf* x_tf * x_tf * x_tf* x_tf * x_tf + w[3] * x_tf * x_tf * x_tf* x_tf * x_tf + w[4] * x_tf * x_tf* x_tf * x_tf + w[5] * x_tf* x_tf * x_tf + w[6] * x_tf * x_tf + w[7] * x_tf + w[8]
    #         y_pred = w[0]* x_tf **5 + w[1] * x_tf**4+ w[2] * x_tf**3 + w[3] *x_tf**2 + w[4]*x_tf**1 + w[5]
    #
    #         loss = tf.reduce_mean(tf.square(y_tf - y_pred))
    #     grad = tape.gradient(loss, w)
    #     for i in range(6):
    #         w[i].assign_add(-(alpha * grad[i]))
    #     if (epoch + 1) % 750 == 0:
    #
    #         print(f"E: {epoch + 1}, L: {loss.numpy()}")
    #
    # y_line = w[0] * x_train ** 5 + w[1] * x_train ** 4 + w[2] * x_train ** 3 + w[3] * x_train ** 2 + w[4] * x_train ** 1 + w[5] * x_train ** 0 #+ w[6] * x_train ** 0# + w[7] * x_train + w[8]
    # y_line_test = w[0] * x_test ** 5 + w[1] * x_test ** 4 + w[2] * x_test ** 3 + w[3] * x_test ** 2 + w[
    #     4] * x_test ** 1 + w[5] * x_test ** 0
    # plt.plot(x_train, y_train, 'k.')
    # plt.plot(x_train, y_line, 'r.')
    # plt.plot(x_test, y_line_test, 'b.')
    # plt.grid()
    # plt.title('еа')
    # plt.show()
    #
    # print(r2_score(y_train, y_line))
    # print(r2_score(y_test, y_line_test))
    # print("mae y_train: " + str(mean_absolute_error(y_train, y_line)))
    # print("mae y_test: " + str(mean_absolute_error(y_test, y_line_test)))
    # print("mape y_train: " + str(mean_absolute_percentage_error(y_train, y_line) * 100))
    # print("mape y_test: " + str(mean_absolute_percentage_error(y_test, y_line_test) * 100))






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

