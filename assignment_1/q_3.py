import pandas as pd
import numpy as np
import math
from collections import deque
import heapq
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from numpy import dot
from numpy.linalg import norm
import matplotlib.pyplot as plt


def knn(num_neighbour, metric, training_set, validation_set, test_set, sklearn = False):
    validation_set, validation_class = np.hsplit(validation_set, np.array([3]))
    validation_class = validation_class.reshape(-1)

    training_set, training_class = np.hsplit(training_set, np.array([3]))
    training_class = training_class.reshape(-1)

    test_set, test_class = np.hsplit(test_set, np.array([3]))
    test_class = test_class.reshape(-1)

    validation_error = 0

    # use sklearn if metric is None
    if sklearn:
        neigh = KNeighborsClassifier(n_neighbors=num_neighbour, metric=metric)
        neigh.fit(training_set, training_class)

    # calculate error in validation set
    for i, x in enumerate(validation_set):
        if sklearn:
            if neigh.predict(x.reshape(1, -1))[0] != validation_class[i]:
                class_sum = -1
            else:
                class_sum = 1
        else:
            k_neighbours = []

            # add distances of between validation point and all training points into min heap
            for j, y in enumerate(training_set):
                distance = metric(x, y)
                heapq.heappush(k_neighbours, (distance, training_class[j]))

            # use min heap to find k closest training class points from validation point
            class_sum = 0
            for k in range(num_neighbour):
                distance, classification = heapq.heappop(k_neighbours)
                
                # class_sum finds if majority of neighbours will belong to the correct validation class
                if classification == validation_class[i]:
                    class_sum += 1
                else:
                    class_sum -= 1

        # if the major of neighbours not in same class as the validation point, that means there will be
        # classification error
        if class_sum < 0:
            validation_error += 1

    # same operations as validation set
    test_error = 0
    for i, x in enumerate(test_set):
        if sklearn:
            if neigh.predict(x.reshape(1, -1))[0] != test_class[i]:
                class_sum = -1
            else:
                class_sum = 1
        else:
            k_neighbours = []
            for j, y in enumerate(training_set):
                distance = metric(x, y)
                heapq.heappush(k_neighbours, (distance, training_class[j]))

            class_sum = 0
            for k in range(num_neighbour):
                distance, classification = heapq.heappop(k_neighbours)
                if classification == test_class[i]:
                    class_sum += 1
                else:
                    class_sum -= 1

        if class_sum < 0:
            test_error += 1

    return validation_error, test_error

            
        

def split_dataset(dataset, training_ratio, validation_ratio, test_ratio):
    validation_index = math.floor(training_ratio * dataset.shape[0])
    test_index = math.floor(training_ratio * dataset.shape[0]) + math.floor(validation_ratio * dataset.shape[0])
    training_dataset = dataset[0:validation_index, :]
    validation_dataset = dataset[validation_index:test_index, :]
    test_dataset = dataset[test_index:, :]

    return training_dataset, validation_dataset, test_dataset

def euclid_dist(x, y):
    res = 0
    for i in range(x.shape[0]):
        res += pow((x[i] - y[i]), 2)

    return math.sqrt(res)

def cosine_sim(x, y):
    # subtract 1 for distance
    return 1 - (dot(x, y) / (norm(x) * norm(y)))


np_array = pd.read_csv("A1Q4NearestNeighbors.csv", names=["Glucose", "BMI", "Age", "Diabetes"]).to_numpy()
def part_2():
    # 2.a
    training_dataset, validation_dataset, test_dataset = split_dataset(np_array, 0.8, 0.1, 0.1)
    validation_error, test_error = knn(1, euclid_dist, training_dataset, validation_dataset, test_dataset, False)
    print(f"\nResults for KNN using Euclidian distance with 0.8, 0.1, 0.1 splits: \nValidation Error: {validation_error/validation_dataset.shape[0]}, Test Error: {test_error}")
    validation_error, test_error = knn(1, cosine_sim, training_dataset, validation_dataset, test_dataset, False)
    print(f"\nResults for KNN using Cosine Similarity with 0.8, 0.1, 0.1 splits: \nValidation Error: {validation_error/validation_dataset.shape[0]}, Test Error: {test_error}")
    validation_error, test_error = knn(1, "euclidean", training_dataset, validation_dataset, test_dataset, True)
    print(f"\nResults for KNN using Sklearn with 0.8, 0.1, 0.1 splits: \nValidation Error: {validation_error/validation_dataset.shape[0]}, Test Error: {test_error}")
    validation_error, test_error = knn(1, "cosine", training_dataset, validation_dataset, test_dataset, True)
    print(f"\nResults for KNN using Sklearn with 0.25, 0.25, 0.50 splits: \nValidation Error: {validation_error/validation_dataset.shape[0]}, Test Error: {test_error}")

    # 2.b
    training_dataset, validation_dataset, test_dataset = split_dataset(np_array, 0.34, 0.33, 0.33)
    validation_error, test_error = knn(1, euclid_dist, training_dataset, validation_dataset, test_dataset, False)
    print(f"\nResults for KNN using Euclidian distance with 0.34, 0.33, 0.33 splits: \nValidation Error: {validation_error/validation_dataset.shape[0]}, Test Error: {test_error}")
    validation_error, test_error = knn(1, cosine_sim, training_dataset, validation_dataset, test_dataset, False)
    print(f"\nResults for KNN using Cosine Similarity with 0.34, 0.33, 0.33 splits: \nValidation Error: {validation_error/validation_dataset.shape[0]}, Test Error: {test_error}")
    validation_error, test_error = knn(1, "euclidean", training_dataset, validation_dataset, test_dataset, True)
    print(f"\nResults for KNN using Sklearn with 0.34, 0.33, 0.33 splits: \nValidation Error: {validation_error/validation_dataset.shape[0]}, Test Error: {test_error}")
    validation_error, test_error = knn(1, "cosine", training_dataset, validation_dataset, test_dataset, True)
    print(f"\nResults for KNN using Sklearn with 0.25, 0.25, 0.50 splits: \nValidation Error: {validation_error/validation_dataset.shape[0]}, Test Error: {test_error}")

    # 2.c
    training_dataset, validation_dataset, test_dataset = split_dataset(np_array, 0.25, 0.25, 0.50)
    validation_error, test_error = knn(1, euclid_dist, training_dataset, validation_dataset, test_dataset, False)
    print(f"\nResults for KNN using Euclidian distance with 0.25, 0.25, 0.50 splits: \nValidation Error: {validation_error/validation_dataset.shape[0]}, Test Error: {test_error}")
    validation_error, test_error = knn(1, cosine_sim, training_dataset, validation_dataset, test_dataset, False)
    print(f"\nResults for KNN using Cosine Similarity with 0.25, 0.25, 0.50 splits: \nValidation Error: {validation_error/validation_dataset.shape[0]}, Test Error: {test_error}")
    validation_error, test_error = knn(1, "euclidean", training_dataset, validation_dataset, test_dataset, True)
    print(f"\nResults for KNN using Sklearn with 0.25, 0.25, 0.50 splits: \nValidation Error: {validation_error/validation_dataset.shape[0]}, Test Error: {test_error}")
    validation_error, test_error = knn(1, "cosine", training_dataset, validation_dataset, test_dataset, True)
    print(f"\nResults for KNN using Sklearn with 0.25, 0.25, 0.50 splits: \nValidation Error: {validation_error/validation_dataset.shape[0]}, Test Error: {test_error}")

def part_3():
    k_list = [1, 3, 5, 11]
    splits = [[0.8, 0.1, 0.1], [0.34, 0.33, 0.33], [0.25, 0.25, 0.5]]
    euclid_valid_error_list = [[] for x in range(3)]
    cosine_valid_error_list = [[] for x in range(3)]
    euclid_test_error_list = [[] for x in range(3)]
    cosine_test_error_list = [[] for x in range(3)]
    for i in range(len(splits)):
        train_r, valid_r, test_r = splits[i]
        for k in k_list:
            training_dataset, validation_dataset, test_dataset = split_dataset(np_array, train_r, valid_r, test_r)
            validation_error, test_error = knn(k, euclid_dist, training_dataset, validation_dataset, test_dataset, False)
            euclid_valid_error_list[i].append(validation_error/validation_dataset.shape[0])
            euclid_test_error_list[i].append(test_error/test_dataset.shape[0])

            validation_error, test_error = knn(k, cosine_sim, training_dataset, validation_dataset, test_dataset, False)
            cosine_valid_error_list[i].append(validation_error/validation_dataset.shape[0])
            cosine_test_error_list[i].append(test_error/test_dataset.shape[0])

    metrics = ["euclidian", "cosine similarity"]

    plt.figure(1)
    for error_list in euclid_test_error_list:
        plt.plot(k_list, error_list)

    plt.legend(["80/10/10", "34/33/33", "0.25/0.25/0.50"])

    plt.show()


    



part_2()