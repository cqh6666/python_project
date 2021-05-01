# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：     MLJAR_learn
   Description :
   Author :       cqh
   date：          2021/4/28 22:45
-------------------------------------------------
   Change Activity:
                   2021/4/28:
-------------------------------------------------
"""
__author__ = 'cqh'

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_digits
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from supervised.automl import AutoML

train_titanic_path = "D:/python_project/machine_learning/Titanic/titanic/train.csv"
test_titanic_path = "D:/python_project/machine_learning/Titanic/titanic/test.csv"


def train_titanic(train_data):
    train_df = pd.read_csv(train_data)
    # test_df = pd.read_csv(test_data)

    # feature_cols = train_df.drop(['Survived', 'PassengerId', 'Name'], axis=1).columns
    feature_cols = train_df.columns[2:]
    target_cols = 'Survived'

    X_train, X_test, y_train, y_test = train_test_split(
        train_df[feature_cols], train_df[target_cols], test_size=0.25
    )

    automl = AutoML(results_path="AutoML_titanic")
    automl.fit(X_train, y_train)

    predictions = automl.predict(X_test)
    print(f"Accuracy: {accuracy_score(y_test, predictions) * 100.0:.2f}%")


def train_digits():
    digits = load_digits()
    X_train, X_test, y_train, y_test = train_test_split(
        pd.DataFrame(digits.data), digits.target, stratify=digits.target, test_size=0.25,
        random_state=123
    )

    # train models with AutoML
    automl = AutoML(mode="Perform", results_path="AutoML_digits")
    automl.fit(X_train, y_train)

    # compute
    predictions = automl.predict_all(X_test)
    print(predictions.head())
    print("Test accuracy:", accuracy_score(y_test, predictions["label"].astype(int)))

    plot_digits(X_test,predictions)

def plot_digits(x_test, predictions):

    _, axes = plt.subplots(nrows=1, ncols=4, figsize=(10, 3))
    for ax, image, prediction in zip(axes, x_test, predictions["label"]):
        ax.set_axis_off()
        image = image.reshape(8, 8)
        ax.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
        ax.set_title(f'Prediction: {prediction}')

    plt.show()

if __name__ == '__main__':
    train_digits()