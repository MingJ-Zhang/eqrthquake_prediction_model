import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score, recall_score, f1_score
from sklearn import tree
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neural_network import MLPClassifier


def mlp():
    train_data = np.load('./data.npy')
    train_value = np.load('./label.npy')
    train_data = np.reshape(train_data,[185,720])


    X_train,X_test,y_train,y_test = train_test_split(train_data,train_value,test_size=0.2)
    # train_value = np.reshape(train_value,[185,1])
    print(train_value.shape)

    # mlp = MLPClassifier(solver='sgd', activation='relu', alpha=1e-4, hidden_layer_sizes=(50, 50), random_state=1,
    #                     max_iter=2000, verbose=True, learning_rate_init=0.0001)
    mlp_clf__tuned_parameters = {"hidden_layer_sizes": [(50,50,), (100, 30)],
                                 "solver": ['adam', 'sgd', 'lbfgs'],
                                 "max_iter": [1000],
                                 "verbose": [True]
                                 }
    mlp = MLPClassifier()
    estimator = GridSearchCV(mlp, mlp_clf__tuned_parameters, n_jobs=6)
    estimator.fit(X_train, y_train)


    result = estimator.predict(X_test)
    # mlp.fit(X_train, y_train)

    # 查看模型结果
    print(estimator.score(X_test, y_test))
    print(classification_report(y_test, result))









def _svm():
    train_data = np.load('./data.npy')
    train_value = np.load('./label.npy')
    train_data = np.reshape(train_data,[185,720])


    X_train,X_test,y_train,y_test = train_test_split(train_data,train_value,test_size=0.2)

    clf = SVC(kernel='rbf')  # 调参
    clf.fit(X_train, y_train)  # 训练
    print(clf.fit(X_train, y_train))  # 输出参数设置

    result = clf.predict(X_test)

    print(classification_report(y_test, result))
    print(accuracy_score(y_test, result))


def cart():
    train_data = np.load('./data.npy')
    train_value = np.load('./label.npy')
    train_data = np.reshape(train_data,[185,720])


    X_train,X_test,y_train,y_test = train_test_split(train_data,train_value,test_size=0.2)


    clf = tree.DecisionTreeClassifier(criterion="entropy", random_state=30, splitter="random")
    clf = clf.fit(X_train, y_train)
    result = clf.predict(X_test)
    print(classification_report(y_test, result))
    print(accuracy_score(y_test, result))


if __name__ == '__main__':
    mlp()


# train_data = np.load('./data.npy')
# train_value = np.load('./label.npy')
# train_data = np.reshape(train_data,[185,720])


# X_train,X_test,y_train,y_test = train_test_split(train_data,train_value,test_size=0.2)
# # train_value = np.reshape(train_value,[185,1])
# print(train_value.shape)

# # mlp = MLPClassifier(solver='sgd', activation='relu', alpha=1e-4, hidden_layer_sizes=(50, 50), random_state=1,
# #                     max_iter=2000, verbose=True, learning_rate_init=0.0001)
# mlp_clf__tuned_parameters = {"hidden_layer_sizes": [(50,50,), (100, 30)],
#                                  "solver": ['adam', 'sgd', 'lbfgs'],
#                                  "max_iter": [1000],
#                                  "verbose": [True]
#                                  }
# mlp = MLPClassifier()
# estimator = GridSearchCV(mlp, mlp_clf__tuned_parameters, n_jobs=6)
# estimator.fit(X_train, y_train)



# # mlp.fit(X_train, y_train)

# # 查看模型结果
# print(estimator.score(X_test, y_test))