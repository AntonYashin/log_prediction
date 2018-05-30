import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score

parameters = ["Vp", "VsElasticOriginal"]


# def read_log_info(parameter, well=1):
#     # file = os.path.abspath("C:\\Users\\anton.yashin\\log_prediction\\well_log_data\\Vp.Well.E\\Vp.Well.E.1.csv")
#     return pd.read_csv("C:/Users/anton.yashin/log_prediction/well_log_data/Vp.Well.E/Vp.Well.E.1.csv", delimiter=",", header=None)


data = pd.read_csv('C:\\Users\\anton.yashin\\log_prediction\\well_log_data\\Vp.Well.E\\Vp.Well.E.1.csv', delimiter=",", header=None)
depth = data[0].values
log_information = data[1].values

depth_train, depth_test, log_train, log_test = train_test_split(depth, log_information, test_size=0.25, random_state=42)

print(os.listdir(os.path.abspath(".\well_log_data\Vp.Well.E")))
plt.scatter(depth, log_information)
plt.scatter(depth, log_information)

plt.show()
#
# degrees = [1, 5, 10]
#
# for i in range(len(degrees)):
#     # ax = plt.subplot(1, len(degrees), i + 1)
#     plt.setp(ax, xticks=(), yticks=())
#
#     polynomial_features = PolynomialFeatures(degree=degrees[i],
#                                              include_bias=False)
#     linear_regression = LinearRegression()
#     pipeline = Pipeline([("polynomial_features", polynomial_features),
#                          ("linear_regression", linear_regression)])
#     pipeline.fit(depth_train[:, np.newaxis], log_train)
#
#     print(pipeline.steps[1][1].coef_)
#     # Evaluate the models using crossvalidation
#     scores = cross_val_score(pipeline, depth_train[:, np.newaxis], log_train,
#                              scoring="neg_mean_squared_error", cv=10)

    # plt.scatter(depth_test, pipeline.predict(depth_test[:, np.newaxis]), label="Model")
    # plt.plot(depth_test, log_test, label="True function")
    # plt.scatter(depth, log_information, edgecolor='b', s=20, label="Samples")
    # plt.xlabel("x")
    # plt.ylabel("y")
    # plt.xlim((0, 1))
    # plt.ylim((-2, 2))
    # plt.legend(loc="best")
    # plt.title("Degree {}\nMSE = {:.2e}(+/- {:.2e})".format(
    #     degrees[i], -scores.mean(), scores.std()))
# plt.show()

