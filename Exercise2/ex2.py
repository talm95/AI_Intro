# import relevant libraries
import pandas as pd
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn import datasets
import matplotlib.pyplot as plt
import numpy as np
import time


# Put the csv file and this python file in the same directory

# Exploring the data, you can try other commands from pandas
bank_data = pd.read_csv("bill_authentication.csv")  # read csv file as a pandas data frame
print(bank_data.shape)  # see the shape of the data - (num_rows, num_cols)
print(bank_data.head())  # see the first 5 lines of the data

# Data preprocessing to:
# (1) divide the data into attributes (features) and labels
# (2) divide the data into train and test
X = bank_data.drop('Class', axis=1)  # all the columns except for the last one
y = bank_data['Class']  # only labels
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)  # split the data to 80% train and 20% test

# Training the algorithm
svclassifier = SVC(kernel='linear')  # Regular SVM
svclassifier.fit(X_train, y_train)  # Fit (train the data)

# Make Predictions
y_pred = svclassifier.predict(X_test)  # y_pred will give us the labels predictions of the test data

# Evaluation
# We will learn in the course how to evaluate the performance of the algorithms.
# Good examples are confusion matrix, accuracy, recall, f1-score ........
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))


"""
Now do the same with Kernel-SVM using the iris dataset.
Try Polynomial Kernel, Gaussian Kernel and Sigmoid Kernel.
1) Load the dataset using these commands: 
iris = datasets.load_iris()
X = iris.data[:, :2]
y = iris.target
2) Perform preprocessing to get train-test split (80-20)
3) Train the algorithm using, each time with a different kernel. Use the command 
SVC(kernel='____') where ____ is the specified kernel (you can look on the documentation of sklearn).
4) Evaluate the performance for all trained models.
5) Plot the partition using matplotlib - see this link for an example - 
https://scikit-learn.org/stable/auto_examples/svm/plot_iris_svc.html
"""

iris = datasets.load_iris()
X = iris.data[:, :2]
y = iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)


def make_meshgrid(x, y, h=.02):
    x_min, x_max = x.min() - 1, x.max() + 1
    y_min, y_max = y.min() - 1, y.max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    return xx, yy


def plot_contours(ax, clf, xx, yy, **params):
    y_pred = clf.predict(X_test)
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    out = ax.contourf(xx, yy, Z, **params)
    return y_pred


models = (SVC(kernel='rbf'),
          SVC(kernel='poly', degree=2),
          SVC(kernel='poly', degree=3),
          SVC(kernel='linear'))

models = (cls.fit(X_train, y_train) for cls in models)

titles = ('SVC with rbf kernel',
          'SVC with polynomial (degree 2) kernel',
          'SVC with polynomial (degree 3) kernel',
          'SVC with linear kernel')

fig, sub = plt.subplots(2, 2)
plt.subplots_adjust(wspace=0.4, hspace=0.4)

X0, X1 = X_test[:, 0], X_test[:, 1]
xx, yy = make_meshgrid(X0, X1)

for clf, title, ax in zip(models, titles, sub.flatten()):
    predicted_y = plot_contours(ax, clf, xx, yy,
                  cmap=plt.cm.coolwarm, alpha=0.8)
    print(title)
    print(classification_report(y_test, predicted_y))
    ax.scatter(X0, X1, c=y_test, cmap=plt.cm.coolwarm, s=20, edgecolors='k')
    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    ax.set_xlabel('Sepal length')
    ax.set_ylabel('Sepal width')
    ax.set_xticks(())
    ax.set_yticks(())
    ax.set_title(title)

plt.show()


