import random
import matplotlib.pyplot as plt

def get_avg(list):
    return sum(list)/len(list)

def get_sd(list):
    mean = sum(list)/len(list)
    variance = sum([((x - mean) ** 2) for x in list]) / len(list) 
    return variance ** 0.5

def q2_2():
    num_samples = 100
    num_dim = 11
    euclid_dist_avg = []
    manhattan_dist_avg = []

    euclid_dist_sd = []
    manhattan_dist_sd = []
    
    # generate dimensions
    d = []
    for i in range(num_dim):
        d.append(pow(2, i))

    for dim in d:
        # generate points, x_list[dimension][point], each dimension should have 100 random value
        x_list = [[] for i in range(dim)]
        for i in range(dim):
            for j in range(num_samples):
                x_list[i].append(random.random())
        
        euclid_dist_list = []
        manhattan_dist_list = []
        # loops through all points
        for j1 in range(num_samples):
            euclid_dist = 0
            manhattan_dist = 0
            for j2 in range(j1 + 1, num_samples):
                euclid_dist = 0
                manhattan_dist = 0
                for i in range(dim):
                    euclid_dist += pow(x_list[i][j1] - x_list[i][j2], 2)
                    manhattan_dist += abs(x_list[i][j1] - x_list[i][j2])
                euclid_dist_list.append(euclid_dist)
                manhattan_dist_list.append(manhattan_dist)
        
        euclid_dist_avg.append(get_avg(euclid_dist_list))
        euclid_dist_sd.append(get_sd(euclid_dist_list)/get_avg(euclid_dist_list))

        manhattan_dist_avg.append(get_avg(manhattan_dist_list))
        manhattan_dist_sd.append(get_sd(manhattan_dist_list)/get_avg(manhattan_dist_list))

    # plotting results
    x_axis = []
    for i in range(num_dim):
        x_axis.append("2^" + str(i))

    plt.figure(1)
    plt.plot(x_axis, euclid_dist_avg)
    plt.ylabel("Average Euclidian distance")
    plt.xlabel("Dimension")

    plt.figure(2)
    plt.plot(x_axis, euclid_dist_sd)
    plt.ylabel("Euclidian distance relative standard deviation")
    plt.xlabel("Dimension")

    plt.figure(3)
    plt.plot(x_axis, manhattan_dist_avg)
    plt.ylabel("Average Manhattan distance")
    plt.xlabel("Dimension")

    plt.figure(4)
    plt.plot(x_axis, manhattan_dist_sd)
    plt.ylabel("Manhattan distance relative standard deviation")
    plt.xlabel("Dimension")

    plt.show()
    

q2_2()

        

            


        

        
    