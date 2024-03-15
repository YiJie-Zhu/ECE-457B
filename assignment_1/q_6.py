import numpy as np
import pandas as pd
import math
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import random


np_dataset = pd.read_csv("A1Q6RawData.csv", header=1).to_numpy()
np.random.seed(2024)
np.random.shuffle(np_dataset)

def split_dataset(dataset, training_ratio, validation_ratio, test_ratio):
    validation_index = math.floor(training_ratio * dataset.shape[0])
    test_index = math.floor(training_ratio * dataset.shape[0]) + math.floor(validation_ratio * dataset.shape[0])
    training_dataset = dataset[0:validation_index, :]
    validation_dataset = dataset[validation_index:test_index, :]
    test_dataset = dataset[test_index:, :]
    return training_dataset, validation_dataset, test_dataset

def train_models(classifier_list, threshold, training_dataset, validation_dataset):

    _, training_set, training_class = np.hsplit(training_dataset, np.array([1, 3]))
    _, validation_set, validation_class = np.hsplit(validation_dataset, np.array([1, 3]))

    for j, classifier in enumerate(classifier_list):
        classifier.fit(training_set, training_class)

        z_count = 0
        o_count = 0
        z_accuracy_count = 0
        o_accuracy_count = 0

        confusion_matrix = {
            "TP": 0,
            "TN": 0,
            "FP": 0,
            "FN": 0
        }

        for i, row in enumerate(validation_set):
            prediction = classifier.predict(row.reshape(1, -1))[0]
            if isinstance(prediction, np.ndarray):
                prediction = prediction[0]

            if prediction > threshold[j]:
                prediction = 1
            else:
                prediction = 0

            if validation_class[i] == 0.0:
                if prediction == validation_class[i]:
                    z_accuracy_count += 1
                z_count += 1
            else:
                if prediction == validation_class[i]:
                    o_accuracy_count += 1
                o_count += 1

            # confusion matrix
            if validation_class[i] == 1.0 and prediction == 0:
                confusion_matrix["FN"] += 1
            elif validation_class[i] == 1.0 and prediction == 1:
                confusion_matrix["TP"] += 1
            elif validation_class[i] == 0.0 and prediction == 1:
                confusion_matrix["FP"] += 1
            elif validation_class[i] == 0.0 and prediction == 0:
                confusion_matrix["TN"] += 1
        print("Confusion Matrix")
        print(confusion_matrix)
        print("Accuracy")
        print(z_accuracy_count/z_count, o_accuracy_count/o_count)
        print("Overall Accuracy")
        print((z_accuracy_count+o_accuracy_count)/(z_count+o_count))



## part 1.a
def part1_a():
    classifier_list = []

    classifier_list.append(KNeighborsClassifier())
    classifier_list.append(DecisionTreeClassifier())
    classifier_list.append(LinearRegression())
    classifier_list.append(LogisticRegression())

    training_dataset, validation_dataset, test_dataset = split_dataset(np_dataset, 0.8, 0.1, 0.1)
    _, training_set, training_class = np.hsplit(training_dataset, np.array([1, 3]))
    _, validation_set, validation_class = np.hsplit(validation_dataset, np.array([1, 3]))

    threshold = [0.5, 0.5, 0, 0.5]

    for j, classifier in enumerate(classifier_list):
        classifier.fit(training_set, training_class)

        accuracy_count = 0

        for i, row in enumerate(validation_set):
            prediction = classifier.predict(row.reshape(1, -1))[0]

            if prediction > threshold[j]:
                prediction = 1
            else:
                prediction = 0

            if prediction == validation_class[i]:
                accuracy_count += 1

        print(accuracy_count/validation_class.shape[0])

# # part 1.c
def part1_c():
    x = []
    y = []
    c = []

    z_count = 0
    o_count = 0

    for row in np_dataset:
        x.append(row[1])
        y.append(row[2])
        if row[3] == 0.0:
            c.append("b")
            z_count += 1
        else:
            c.append("r")
            o_count += 1

    print(z_count, o_count)

    plt.scatter(x, y, c=c, s=5)
    plt.xlim(-2, 3)
    plt.ylim(-5, 5)
    plt.show()

# part 1.d and 1.e
def part1_d():
    training_dataset, validation_dataset, test_dataset = split_dataset(np_dataset, 0.8, 0.1, 0.1)

    classifier_list = []
    threshold = [0.5, 0.5, 0, 0.5]

    classifier_list.append(KNeighborsClassifier())
    classifier_list.append(DecisionTreeClassifier())
    classifier_list.append(LinearRegression())
    classifier_list.append(LogisticRegression())

    train_models(classifier_list, threshold, training_dataset, validation_dataset)


# part 1.h
def oversample(np_dataset):
    class_1_list = np_dataset[np_dataset[:, 3] == 1.0]

    new_dataset = np_dataset

    while new_dataset.shape[0] < 2*(np_dataset.shape[0]-class_1_list.shape[0]):
        index = random.randrange(class_1_list.shape[0])
        new_dataset = np.vstack((new_dataset, class_1_list[index]))
    
    np.random.shuffle(new_dataset)
    return new_dataset

def part1_h():
    training_dataset, validation_dataset, test_dataset = split_dataset(np_dataset, 0.8, 0.1, 0.1)

    classifier_list = []
    threshold = [0.5, 0.5, 0, 0.5]

    classifier_list.append(KNeighborsClassifier())
    classifier_list.append(DecisionTreeClassifier())
    classifier_list.append(LinearRegression())
    classifier_list.append(LogisticRegression())

    train_models(classifier_list, threshold, oversample(training_dataset), validation_dataset)

# # part 1.i
def undersample(np_dataset):
    class_1_list = np_dataset[np_dataset[:, 3] == 1.0]
    class_0_list = np_dataset[np_dataset[:, 3] == 0.0]
    print(class_1_list.shape[0])

    new_dataset = class_1_list

    while new_dataset.shape[0] < 2*(class_1_list.shape[0]):
        index = random.randrange(class_0_list.shape[0])
        new_dataset = np.vstack((new_dataset, class_0_list[index]))
    
    np.random.shuffle(new_dataset)
    return new_dataset

def part1_i():
    training_dataset, validation_dataset, test_dataset = split_dataset(np_dataset, 0.8, 0.1, 0.1)

    classifier_list = []
    threshold = [0.5, 0.5, 0, 0.5]

    classifier_list.append(KNeighborsClassifier())
    classifier_list.append(DecisionTreeClassifier())
    classifier_list.append(LinearRegression())
    classifier_list.append(LogisticRegression())

    train_models(classifier_list, threshold, undersample(training_dataset), validation_dataset)


# # part 2.b
def part2_b():
    training_dataset, validation_dataset, test_dataset = split_dataset(np_dataset, 0.8, 0.1, 0.1)

    classifier_list = []
    threshold = [0.5, 0.5, 0, 0.5]

    classifier_list.append(KNeighborsClassifier(n_neighbors=1))
    classifier_list.append(KNeighborsClassifier(n_neighbors=3))
    classifier_list.append(KNeighborsClassifier(n_neighbors=5))

    train_models(classifier_list, threshold, training_dataset, validation_dataset)


# part 2.c
def part2_c():
    training_dataset, validation_dataset, test_dataset = split_dataset(np_dataset, 0.8, 0.1, 0.1)

    classifier_list = []
    threshold = [0.5, 0.5, 0, 0.5]

    classifier_list.append(DecisionTreeClassifier(max_depth=1, random_state=123123))
    classifier_list.append(DecisionTreeClassifier(max_depth=2, random_state=123123))
    classifier_list.append(DecisionTreeClassifier(max_depth=3, random_state=123123))

    train_models(classifier_list, threshold, training_dataset, validation_dataset)

part1_d()