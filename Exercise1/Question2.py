from sklearn.svm import SVC

x_train = [[0, 0], [0, 1], [1, 0], [1, 1]]
y_train = [1, 0, 0, 1]
sv_classifier = SVC(kernel='rbf')
sv_classifier.fit(x_train, y_train)

y_predict = sv_classifier.predict(x_train)

print(y_predict)
