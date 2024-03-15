import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
import random
from xgboost import XGBClassifier

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier, GradientBoostingClassifier

np_dataset = pd.read_csv("A1Q3DecisionTrees.csv", header=0).to_numpy()
np.random.seed(2024)
np.random.shuffle(np_dataset)

def split_dataset(dataset, training_ratio, validation_ratio, test_ratio):
    validation_index = math.floor(training_ratio * dataset.shape[0])
    test_index = math.floor(training_ratio * dataset.shape[0]) + math.floor(validation_ratio * dataset.shape[0])
    training_dataset = dataset[0:validation_index, :]
    validation_dataset = dataset[validation_index:test_index, :]
    test_dataset = dataset[test_index:, :]
    return training_dataset, validation_dataset, test_dataset

## part 1
def part_1():
    training_dataset, validation_dataset, test_dataset = split_dataset(np_dataset, 0.8, 0.1, 0.1)

    num_base = 100
    step = training_dataset.shape[0]//num_base
    tree_list = []

    # training base classifiers
    for i in range(num_base):
        training_split = training_dataset[i*step:(i+1)*step]

        while training_split.shape[0] < training_dataset.shape[0]:
            index = random.randrange(training_dataset.shape[0])
            training_split = np.vstack((training_split, training_dataset[index]))

        training_set, training_class = np.hsplit(training_split, np.array([9]))
        tree = DecisionTreeClassifier(criterion="entropy")
        tree.fit(training_set, training_class)
        tree_list.append(tree)

    tree_validation_accuracy_count = [0] * num_base
    tree_test_accuracy_count = [0] * num_base

    ensemble_validation_accuracy_count = 0
    ensemble_test_accuracy_count = 0

    # getting validation prediction for ensemble
    validation_set, validation_class = np.hsplit(validation_dataset, np.array([9]))
    for i, x in enumerate(validation_set):
        ensemble_validation_count = 0
        for j, tree in enumerate(tree_list):
            prediction = tree.predict(x.reshape(1, -1))[0]
            ensemble_validation_count += prediction
            if prediction == validation_class[i]:
                tree_validation_accuracy_count[j] += 1
        # if more than 50 base classifiers predict 1
        if ensemble_validation_count > num_base/2:
            ensemble_prediction = 1
        else:
            ensemble_prediction = 0

        if ensemble_prediction == validation_class[i]:
            ensemble_validation_accuracy_count += 1
            

    # getting test prediction for ensemble
    test_set, test_class = np.hsplit(test_dataset, np.array([9]))
    for i, x in enumerate(test_set):
        ensemble_test_count = 0
        for j, tree in enumerate(tree_list):
            prediction = tree.predict(x.reshape(1, -1))[0]
            ensemble_test_count += prediction
            if prediction == test_class[i]:
                tree_test_accuracy_count[j] += 1

        # if more than 50 base classifiers predict 1
        if ensemble_test_count > num_base/2:
            ensemble_prediction = 1
        else:
            ensemble_prediction = 0

        if ensemble_prediction == test_class[i]:
            ensemble_test_accuracy_count += 1

    # calculating data rate
    total_validation_data_count = validation_set.shape[0]
    total_test_data_count = test_set.shape[0]

    tree_validation_accuracy_rate = [x / total_validation_data_count for x in tree_validation_accuracy_count] + [ensemble_validation_accuracy_count/total_validation_data_count]
    tree_test_accuracy_rate = [x / total_test_data_count for x in tree_test_accuracy_count] + [ensemble_test_accuracy_count/total_test_data_count]

    plt.figure(1)
    plt.title("Histogram of validation accuracy count across 100 base classifiers")
    plt.ylabel("Number of classifiers")
    plt.xlabel("Accuracy rate")
    plt.hist(tree_validation_accuracy_rate)

    plt.figure(2)
    plt.title("Histogram of test accuracy count across 100 base classifiers")
    plt.ylabel("Number of classifiers")
    plt.xlabel("Accuracy rate")
    plt.hist(tree_test_accuracy_rate)

    # plt.show()

    print(ensemble_validation_accuracy_count/total_validation_data_count)
    print(ensemble_test_accuracy_count/total_test_data_count)


## part 2
def part_2():
    training_dataset, validation_dataset, test_dataset = split_dataset(np_dataset, 0.8, 0.1, 0.1)
    training_set, training_class = np.hsplit(training_dataset, np.array([9]))
    training_class = training_class.reshape(-1)
    validation_set, validation_class = np.hsplit(validation_dataset, np.array([9]))

    tree = None

    classifier = GradientBoostingClassifier()
    classifier.fit(training_set, training_class)

    accuracy_count = 0
    for i, row in enumerate(validation_set):
        if classifier.predict(row.reshape(1, -1))[0] == validation_class[i]:
            accuracy_count += 1

    print(f"Accuracy: {accuracy_count/validation_set.shape[0]}")

    xgboost = XGBClassifier()
    xgboost.fit(training_set, training_class)

    accuracy_count = 0
    for i, row in enumerate(validation_set):
        if xgboost.predict(row.reshape(1, -1))[0] == validation_class[i]:
            accuracy_count += 1

    print(f"XGBoost Accuracy: {accuracy_count/validation_set.shape[0]}")

part_2()



