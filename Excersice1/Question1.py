import numpy as np
from sklearn import linear_model
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt

training_set = np.array([(12.47, 1.12), (6.565, 0.795), (0.73, -0.004), (5.608, 0.802), (9.75, -0.995), (6.668, 0.81),
                         (12.271, 0.986), (14.338, -0.527), (5.72, 1.013), (8.805, -0.759), (16.204, 0.459),
                         (7.219, 0.476)])

x_training = np.array([])
y_training = np.array([])
for pair in training_set:
    x_training = np.append(x_training, pair[0])
    y_training = np.append(y_training, pair[1])

test_set = np.array([(3.974, 0.83), (0.114, -0.145), (7.001, 0.703), (9.894, -0.897), (7.053, 0.756), (4.243, 0.765),
                     (13.82, -0.38), (4.484, 0.998)])

x_test = np.array([])
y_test = np.array([])
for pair in test_set:
    x_test = np.append(x_test, pair[0])
    y_test = np.append(y_test, pair[1])


# no regularization

def fit_regression(degree):

    reg = linear_model.LinearRegression()
    polynomial_features = PolynomialFeatures(degree)
    x_training_degree = x_training[:, np.newaxis]
    x_training_transformed = polynomial_features.fit_transform(x_training_degree)
    reg.fit(x_training_transformed, y_training)
    x_test_degree = x_test[:, np.newaxis]
    x_test_transformed = polynomial_features.fit_transform(x_test_degree)
    y_predict = reg.predict(x_test_transformed)
    loss = 0.5 * np.sum((y_test - y_predict)**2)
    print("the loss for polynomial regression of degree " + str(degree) + " is " + str(loss))

    new_x = np.linspace(0, 15, 100)
    new_x_degree = new_x[:, np.newaxis]
    new_x_transformed = polynomial_features.fit_transform(new_x_degree)
    new_y_predict = reg.predict(new_x_transformed)

    plt.plot(x_test, y_test, '+r', new_x, new_y_predict, 'b')
    plt.title("polynomial regression with degree " + str(degree))
    plt.show()


fit_regression(2)
fit_regression(6)
fit_regression(16)


# with regularization

def fit_ridge_regression(degree, lambda1):
    reg = linear_model.Ridge(lambda1)
    polynomial_features = PolynomialFeatures(degree)
    x_training_degree = x_training[:, np.newaxis]
    x_training_transformed = polynomial_features.fit_transform(x_training_degree)
    reg.fit(x_training_transformed, y_training)
    x_test_degree = x_test[:, np.newaxis]
    x_test_transformed = polynomial_features.fit_transform(x_test_degree)
    y_predict = reg.predict(x_test_transformed)
    loss = 0.5 * np.sum((y_test - y_predict) ** 2) + 0.5 * lambda1 * np.square(np.linalg.norm(reg.coef_))
    print("the loss for polynomial regression of degree " + str(degree) + " with lambda " + str(lambda1) + " is " + str(loss))

    new_x = np.linspace(0, 15, 100)
    new_x_degree = new_x[:, np.newaxis]
    new_x_transformed = polynomial_features.fit_transform(new_x_degree)
    new_y_predict = reg.predict(new_x_transformed)

    plt.plot(x_test, y_test, '+r', new_x, new_y_predict, 'b')
    plt.title("polynomial regression with ridge regularization with degree " + str(degree) + " and lambda: " + str(lambda1))
    plt.show()

    return loss


lambdas = [0.0001, 0.001, 0.1, 1, 10, 100, 1000, 10000]
losses = np.array([])
for l in lambdas:
    loss = fit_ridge_regression(6, l)
    losses = np.append(losses, loss)

plt.plot(lambdas, losses)
plt.title("loss as function of lambda in ridge regression with degree 6")
plt.show()
