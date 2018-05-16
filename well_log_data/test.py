import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline


def f(x):
    """ function to approximate by polynomial interpolation"""
    return x * np.sin(x)


def poly(x, coef):
    """ function to approximate by polynomial interpolation"""
    return 7


# generate points used to plot
x_plot = np.linspace(0, 10, 100)
# print(x_plot.shape)

# generate points and keep a subset of them
x = np.linspace(0, 10, 100)
rng = np.random.RandomState(0)
rng.shuffle(x)
x = np.sort(x[:20])
y = f(x)

# create matrix versions of these arrays
X = x[:, np.newaxis]
X_plot = x_plot[:, np.newaxis]

colors = ['teal', 'yellowgreen', 'gold']
# plt.plot(x_plot, f(x_plot), color='cornflowerblue', linewidth=2,
#          label="ground truth")
decomposition = np.array([])

model = make_pipeline(PolynomialFeatures(8), LinearRegression())
model.fit(X, y)

coef = model.steps[1][1].coef_
transform_data = model.steps[0][1].transform(X)
decomposition = coef*transform_data

_y_ = [v.sum() for v in decomposition]

print("Sample: ", y)
print("Predicted: ", _y_)

# ax = plt.subplot(1, 2, 0)
# plt.setp(ax, xticks=(), yticks=())

plt.plot(X, y)
plt.plot(X, _y_)
plt.show()

# for count, degree in enumerate([15]):
#     model = make_pipeline(PolynomialFeatures(degree), LinearRegression())
#     var = model.fit(X, y)
#
#     coef = model.steps[1][1].coef_
#     transform_data = model.steps[0][1].transform(X)
#     print(transform_data)
#     decomposition = coef*transform_data
#     for data in decomposition:
#         continue
#         # print(data.sum())
#
#     # print(y)
#     # print(poly.get_feature_names(input_features=None))
#     # print("model = ", type(model.steps[1][1].coef_))
#
#     y_plot = model.predict(X_plot)
#     plt.plot(x_plot, y_plot, color=colors[count], linewidth=lw,
#              label="degree %d" % degree)
#
# plt.legend(loc='lower left')

# plt.show()