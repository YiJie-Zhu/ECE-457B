import pandas as pd
import numpy as np
import math
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt

np_dataset = pd.read_csv("A1Q3DecisionTrees.csv", header=0).to_numpy()
np.random.seed(2024)
np.random.shuffle(np_dataset)
####### part 1.a, Sex: 1=Male, 0=Female
def part1_a():
    male_wrong_count = 0
    male_total_count = 0
    female_wrong_count = 0
    female_total_count = 0
    for row in np_dataset:
        if row[8] == 1:
            male_total_count += 1
            if row[9] == 1:
                male_wrong_count += 1
        elif row[8] == 0:
            female_total_count += 1
            if row[9] == 0:
                female_wrong_count += 1

    print(female_total_count, female_wrong_count)
    print(male_total_count, male_wrong_count)

####### part 1.c
def part1_c():
    young_died_count = 0
    old_survived_count = 0
    threshold = 65
    for row in np_dataset:
        # <= if threshold == 25
        if row[1] < threshold and row[9] == 0:
            young_died_count += 1
        elif row[1] >= threshold and row[9] == 1:
            old_survived_count += 1

    print(young_died_count, old_survived_count)

####### part 1.d
def get_age(row):
    return row[1]

def get_gender(row):
    return row[8]

def get_survived(row):
    return row[9]

def part1_d():
    impurity = 0
    for row in np_dataset:
        if get_gender(row) == 1:
            if get_age(row) < 65:
                if get_survived(row) == 0:
                    impurity += 1
            else:
                if get_survived(row) == 1:
                    impurity += 1
        else:
            if get_age(row) < 65:
                if get_survived(row) == 0:
                    impurity += 1
            else:
                print("HERE")
                if get_survived(row) == 1:
                    impurity += 1
    print(impurity)

    impurity = 0
    for row in np_dataset:
        if get_age(row) <= 25:
            if get_gender(row) == 0:
                if get_survived(row) == 0:
                    impurity += 1
            else:
                if get_survived(row) == 1:
                    impurity += 1
        else:
            if get_gender(row) == 0:
                if get_survived(row) == 0:
                    impurity += 1
            else:
                if get_survived(row) == 1:
                    impurity += 1
    print(impurity)

######## part e
def gini_index(dataset, feature_index, value_index):
    feature_map = {}
    total_count = 0

    for row in dataset:
        feature = row[feature_index]
        value = row[value_index]

        if feature not in feature_map:
            feature_map[feature] = {
                "total": 0
            }
        
        feature_map[feature][value] = feature_map[feature].get(value, 0) + 1
        feature_map[feature]["total"] += 1
        total_count += 1
    print(feature_map)
    weighted_gini = 0

    for feature, data in feature_map.items():
        prob_sum = 0
        feature_total = data["total"]
        for key, val in data.items():
            if key == "total":
                continue
            
            prob_sum += (val/feature_total) ** 2

        weighted_gini += (1-prob_sum)*(feature_total/total_count)
    
    print(weighted_gini)

def shannon_entropy(dataset, feature_index, value_index):
    feature_map = {}
    total_count = 0

    for row in dataset:
        feature = row[feature_index]
        value = row[value_index]

        if feature not in feature_map:
            feature_map[feature] = {
                "total": 0
            }
        
        feature_map[feature][value] = feature_map[feature].get(value, 0) + 1
        feature_map[feature]["total"] += 1
        total_count += 1
    print(feature_map)
    weighted_gini = 0

    for feature, data in feature_map.items():
        prob_sum = 0
        feature_total = data["total"]
        for key, val in data.items():
            if key == "total":
                continue
            p = val/feature_total
            prob_sum += p*math.log2(p)

        weighted_gini += (-1*prob_sum)*(feature_total/total_count)
    
    print(weighted_gini)
def part1_e():
    shannon_entropy(np_dataset, 8, 9)
    shannon_entropy(np_dataset, 1, 9)

######## part 2.a
def part2_a():
    training_set, training_class = np.hsplit(np_dataset, np.array([9]))

    tree = DecisionTreeClassifier(max_depth=3, random_state=123123, splitter="random")
    tree.fit(training_set, training_class)

    plot_tree(tree, class_names=["Survived", "Not Survived"])
    plt.show()

####### part 3
np_dataset = pd.read_csv("A1Q3DecisionTrees.csv", header=0).to_numpy()
def split_dataset(dataset, training_ratio, validation_ratio, test_ratio):
    validation_index = math.floor(training_ratio * dataset.shape[0])
    test_index = math.floor(training_ratio * dataset.shape[0]) + math.floor(validation_ratio * dataset.shape[0])
    training_dataset = dataset[0:validation_index, :]
    validation_dataset = dataset[validation_index:test_index, :]
    test_dataset = dataset[test_index:, :]
    return training_dataset, validation_dataset, test_dataset

def part3():
    splits = [
        (0.80, 0.10, 0.10),
        (0.34, 0.33, 0.33),
        (0.25, 0.25, 0.50)
    ]

    criterions = ["gini", "entropy"]

    for criteria in criterions:
        for train_ratio, valid_ratio, test_ratio in splits:
            tree = DecisionTreeClassifier(max_depth=3, random_state=123123, criterion=criteria)
            training_dataset, validation_dataset, test_dataset = split_dataset(np_dataset, train_ratio, valid_ratio, test_ratio)

            training_set, training_class = np.hsplit(training_dataset, np.array([9]))
            training_class = training_class.reshape(-1)
            validation_set, validation_class = np.hsplit(validation_dataset, np.array([9]))
            validation_class = validation_class.reshape(-1)
            test_set, test_class = np.hsplit(test_dataset, np.array([9]))

            tree.fit(training_set, training_class)

            accuracy_count = 0

            for i, x in enumerate(validation_set):
                if tree.predict(x.reshape(1, -1))[0] == validation_class[i]:
                    accuracy_count += 1

            print(f"Validation Accuracy score for criteria:{criteria} on splits of: {train_ratio} {valid_ratio} {test_ratio} is {accuracy_count/validation_class.shape[0]}")

            accuracy_count = 0

            for i, x in enumerate(test_set):
                if tree.predict(x.reshape(1, -1))[0] == test_class[i]:
                    accuracy_count += 1

            print(f"Test Accuracy score for criteria:{criteria} on splits of: {train_ratio} {valid_ratio} {test_ratio} is {accuracy_count/test_class.shape[0]}")

part1_d()
