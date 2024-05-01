import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import r_regression
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import r2_score, mean_absolute_error,  mean_absolute_percentage_error
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import PolynomialFeatures
import tensorflow as tf
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import SGDRegressor
from sklearn.svm import SVR


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
    data = np.linspace(-3, 3, 100).reshape(-1, 1)
    data2 = np.linspace(-3, 3, 100).reshape(-1, 1)
    data_predict = ridge.predict(np.column_stack([data, data2]))
    all_data_predict = ridge.predict(all_data)

    plt.plot(x1_split, y_split, 'b.')
    plt.plot(data, data_predict, 'kx')
    plt.plot(x1_split, all_data_predict, 'rx')
    plt.grid()
    plt.legend(('train', 'predict'))
    plt.show()
    # plt.plot(x2_split, y_split, 'b.')
    # plt.plot(x2_split, data_predict, 'kx')
    # plt.grid()
    # plt.legend(('train', 'predict'))
    # plt.show()






def not_lin_regression():
    #task 2.1
    df = pd.read_csv("lab3_poly1.csv")
    df = df.sort_values(by = "x")
    print(df.head())
    plt.scatter(df["x"],df["y"], c="b", marker = ".")
    plt.show()

    #task 2.2
    x_split = np.array(df["x"]).reshape(-1, 1)
    y_split = np.array(df["y"]).reshape(-1, 1)

    x_train, x_test,  y_train, y_test = (
        train_test_split(x_split,  y_split, test_size=0.3))
    # plt.scatter(x_train, y_train, c="b", marker=".")
    # plt.scatter(x_test, y_test, c="r", marker=".")
    # plt.legend(('train', 'test'))
    # plt.show()

    #task 2.3
    # lin = LinearRegression()
    # lin.fit(x_train, y_train)
    # y_train_pred = lin.predict(x_train)
    # y_test_pred = lin.predict(x_test)
    # print(lin.coef_, lin.intercept_)
    #
    # plt.plot(x_train, y_train, 'k.')
    # plt.plot(np.array(x_test), np.array(y_test), 'b.')
    # plt.plot(np.array(x_test), np.array(y_test_pred), 'r-')
    # plt.grid()
    # plt.title('degree = 1')
    # plt.legend(('data', 'test',"predict"))
    # plt.show()

    #task 2.4
    # 2 -> 4 -> 6
    test_list = []
    train_list=[]
    list_index = [1,2,3,4,5,6,14,20,35,40]
    # for n in list_index:
    #     data = np.linspace(-2,2,100).reshape(-1,1)
    #     X_train = PolynomialFeatures(n).fit_transform(x_train)
    #     X_test = PolynomialFeatures(n).fit_transform(x_test)
    #     Demo = PolynomialFeatures(n).fit_transform(data)
    #     lin2 = LinearRegression(fit_intercept=False)
    #     lin2.fit(X_train, y_train)
    #     y_data = lin2.predict(Demo)
    #     #print(lin2.coef_, lin2.intercept_)
    #     c2 = lin2.coef_[0]
    #     y_train_pred2 = lin2.predict(X_train)
    #     y_test_pred2 =lin2.predict(X_test)
    #     plt.plot(x_train, y_train, 'k.')
    #     #plt.plot(data, y_data,"r-")
    #     plt.plot(x_train, np.array(y_train_pred2), 'r-')
    #     plt.plot(x_test, np.array(y_test_pred2), 'b-')
    #     plt.grid()
    #     plt.title('degree = '+str(n))
    #     plt.legend(("train","train_predict","test_predict"))
    #     plt.show()
    #     test_list.append(r2_score(y_train,y_train_pred2))
    #     train_list.append(r2_score(y_test, y_test_pred2))


    # plt.plot(test_list,"r-" )
    # plt.plot(train_list, "b-")
    # plt.grid()
    # plt.legend(('test', 'train'))
    # plt.show()

    # X_train = PolynomialFeatures(5).fit_transform(x_train)
    # X_test = PolynomialFeatures(5).fit_transform(x_test)
    # lin2 = LinearRegression(fit_intercept=False)
    # lin2.fit(X_train, y_train)
    #
    # print(lin2.coef_, lin2.intercept_)
    # c2 = lin2.coef_[0]
    # print(c2)
    # y_train_pred2 = lin2.predict(X_train)
    # y_test_pred2 = lin2.predict(X_test)
    #
    # #task 2.5
    # #>1/2 значит все хорошо
    # print(r2_score(y_train, y_train_pred2))
    # print(r2_score(y_test, y_test_pred2))
    # print("mae y_train: "+str(mean_absolute_error(y_train, y_train_pred2)))
    # print("mae y_test: "+str(mean_absolute_error(y_test, y_test_pred2)))
    # print("mape y_train: "+str(mean_absolute_percentage_error(y_train, y_train_pred2) * 100))
    # print("mape y_test: "+str(mean_absolute_percentage_error(y_test, y_test_pred2) * 100))
    # #
    # # #task 2.6
    # data = np.linspace(np.min(x_split),np.max(x_split),100).reshape(-1,1)
    # X_data = PolynomialFeatures(5).fit_transform(data)
    # plt.plot(x_split, y_split, 'k.')
    # plt.plot(data, lin2.predict(X_data), 'r-')
    # plt.grid()
    # plt.title('degree = 5')
    # plt.legend(("data","polynome"))
    # plt.show()
    #
    # task 2.7
    x_tf = tf.constant(np.array(x_train), dtype=tf.float32)
    y_tf = tf.constant(np.array(y_train), dtype=tf.float32)
    w = [tf.Variable(np.random.randn()) for _ in range(6)]
    print(*w)
    alpha = tf.constant(0.001, dtype=tf.float32)
    epoch_n = 9000
    for epoch in range(epoch_n):
        with tf.GradientTape() as tape:
            y_pred = w[0]* x_tf **5 + w[1] * x_tf**4+ w[2] * x_tf**3 + w[3] *x_tf**2 + w[4]*x_tf**1 + w[5]

            loss = tf.reduce_mean(tf.square(y_tf - y_pred))
        grad = tape.gradient(loss, w)
        for i in range(6):
            w[i].assign_add(-(alpha * grad[i]))
        if (epoch + 1) % 750 == 0:

            print(f"E: {epoch + 1}, L: {loss.numpy()}")

    y_line = w[0] * x_train ** 5 + w[1] * x_train ** 4 + w[2] * x_train ** 3 + w[3] * x_train ** 2 + w[4] * x_train ** 1 + w[5] * x_train ** 0
    y_line_test = w[0] * x_test ** 5 + w[1] * x_test ** 4 + w[2] * x_test ** 3 + w[3] * x_test ** 2 + w[4] * x_test ** 1 + w[5] * x_test ** 0
    plt.plot(x_train, y_train, 'k.')
    plt.plot(x_test, y_test, 'r.')
    plt.plot(x_test, y_line_test, 'b.')
    plt.grid()
    plt.title('tensor flow')
    plt.legend(("train","test","test_predict"))
    plt.show()

    print(r2_score(y_train, y_line))
    print(r2_score(y_test, y_line_test))
    print("mae y_train: " + str(mean_absolute_error(y_train, y_line)))
    print("mae y_test: " + str(mean_absolute_error(y_test, y_line_test)))
    print("mape y_train: " + str(mean_absolute_percentage_error(y_train, y_line) * 100))
    print("mape y_test: " + str(mean_absolute_percentage_error(y_test, y_line_test) * 100))






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

    plt.scatter(df_not_null["Hours Studied"], df_not_null["Performance Index"], c="b", marker=".")
    plt.xlabel("Hours Studied")
    plt.ylabel("Performance Index")
    plt.show()
    plt.scatter(df_not_null["Previous Scores"], df_not_null["Performance Index"], c="b", marker=".")
    plt.xlabel("Previous Scores")
    plt.ylabel("Performance Index")
    plt.show()
    plt.scatter(df_not_null["Extracurricular Activities"], df_not_null["Performance Index"], c="b", marker=".")
    plt.xlabel("Extracurricular Activities")
    plt.ylabel("Performance Index")
    plt.show()
    plt.scatter(df_not_null["Sleep Hours"], df_not_null["Performance Index"], c="b", marker=".")
    plt.xlabel("Sleep Hours")
    plt.ylabel("Performance Index")
    plt.show()
    plt.scatter(df_not_null["Sample Question Papers Practiced"], df_not_null["Performance Index"], c="b", marker=".")
    plt.xlabel("Sample Question Papers Practiced")
    plt.ylabel("Performance Index")
    plt.show()
    Y_data = df_not_null[['Performance Index']]
    X_data = df_not_null.drop(['Performance Index'], axis = 1)
    X1_train, X1_test,X2_train, X2_test,X3_train, X3_test,X4_train, X4_test,X5_train, X5_test, Y_train, Y_test \
        = (train_test_split(np.array(X_data["Hours Studied"]).reshape(-1,1),
                            np.array(X_data["Previous Scores"]).reshape(-1,1),
                            np.array(X_data["Extracurricular Activities"]).reshape(-1,1),
                            np.array(X_data["Sleep Hours"]).reshape(-1,1),
                            np.array(X_data["Sample Question Papers Practiced"]).reshape(-1,1),Y_data, test_size=0.3))
    X_train = np.column_stack([X1_train,X2_train,X3_train,X4_train,X5_train])
    X_test = np.column_stack([X1_test, X2_test, X3_test, X4_test, X5_test])
    models = [LinearRegression(),
              Lasso(0.1),
              Ridge(1),
              ElasticNet(0.1),
              SGDRegressor(alpha=0.001, max_iter=4000)]
    list_deter=[]
    list_mae = []
    list_mape = []
    for model in models:
        model.fit(X_train, Y_train)
        predict = model.predict(X_test)
        list_deter.append(r2_score(Y_test, predict))
        list_mae.append(mean_absolute_error(Y_test, predict))
        list_mape.append(mean_absolute_percentage_error(Y_test, predict)*100)
    print(list_deter)
    print(list_mae)
    print(list_mape)

    model = Ridge(1)
    model.fit(X_train, Y_train)
    print(model.coef_)
    predict = model.predict(X_test)
    plt.scatter(X1_train, Y_train, c="b", marker=".")
    plt.scatter(X1_test, Y_test, c="r", marker=".")
    plt.scatter(X1_test, predict, c="k", marker="x")
    plt.xlabel("Hours Studied")
    plt.ylabel("Performance Index")
    plt.legend(('train', 'test','predict'))
    plt.show()
    plt.scatter(X2_train, Y_train, c="b", marker=".")
    plt.scatter(X2_test, Y_test, c="r", marker=".")
    plt.scatter(X2_test, predict, c="k", marker="x")
    plt.xlabel("Previous Scores")
    plt.ylabel("Performance Index")
    plt.legend(('train', 'test', 'predict'))
    plt.show()
    plt.scatter(X3_train, Y_train, c="b", marker=".")
    plt.scatter(X3_test, Y_test, c="r", marker=".")
    plt.scatter(X3_test, predict, c="k", marker="x")
    plt.xlabel("Extracurricular Activities")
    plt.ylabel("Performance Index")
    plt.legend(('train', 'test', 'predict'))
    plt.show()
    plt.scatter(X4_train, Y_train, c="b", marker=".")
    plt.scatter(X4_test, Y_test, c="r", marker=".")
    plt.scatter(X4_test, predict, c="k", marker="x")
    plt.xlabel("Sleep Hours")
    plt.ylabel("Performance Index")
    plt.legend(('train', 'test', 'predict'))
    plt.show()
    plt.scatter(X5_train, Y_train, c="b", marker=".")
    plt.scatter(X5_test, Y_test, c="r", marker=".")
    plt.scatter(X5_test, predict, c="k", marker="x")
    plt.xlabel("Sample Question Papers Practiced")
    plt.ylabel("Performance Index")
    plt.legend(('train', 'test', 'predict'))
    plt.show()








# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    #lin_regres()
    not_lin_regression()
    #model_regression()
