import numpy as np
import math
import sys
import time


def convert_to_bin(data):
    result = data / 127
    result = result.astype("int")
    result = result.astype("float64")
    return result


def softmax(input):
    e_x = np.exp(input - np.max(input))
    return e_x / e_x.sum()


def EM(data, labels):
    number_of_images = 60000
    number_of_pixel = 784
    number_of_cluster = 10
    diff = 0
    pi = [0.1 for i in range(number_of_cluster)]
    pi = np.asarray(pi, dtype='float64')
    mu = np.random.rand(number_of_cluster, number_of_pixel)

    N = [0 for i in range(number_of_cluster)]
    start_time = time.time()
    iter = 0
    while 1:
        # Estep
        z = []

        for i in range(number_of_images):
            one = np.ones(28)
            denominator = 0.0
            numerator_list = []
            for j in range(number_of_cluster):
                numerator = pi[j]
                for k in range(0, number_of_pixel, 28):
                    temp_mu = mu[j][k:k + 28]
                    temp_data = data[i][k:k + 28]
                    numerator = numerator * 1e7

                    numerator = numerator * np.prod(temp_mu ** temp_data) * np.prod(
                        (one - temp_mu) ** (one - temp_data))

                numerator_list.append(numerator)

            sum = np.sum(numerator_list)

            for j in range(number_of_cluster):
                numerator_list[j] = numerator_list[j] / sum

            z.append(numerator_list)

        z = np.asarray(z, dtype='float64')

        sum_of_z = np.sum(z, axis=0)

        N = sum_of_z
        sum_of_N = np.sum(N)

        store_mu = np.zeros((number_of_cluster, number_of_pixel))

        for i in range(number_of_cluster):
            for j in range(number_of_images):
                store_mu[i] = store_mu[i] + z[j][i] * data[j]

        for i in range(number_of_cluster):
            store_mu[i] = (store_mu[i] + 1e-8) / (N[i] + number_of_pixel * 1e-8)

        mu_diff = np.sum(np.absolute(store_mu - mu))

        mu = store_mu

        store_pi = np.zeros(number_of_cluster)

        for i in range(number_of_cluster):
            store_pi[i] = (N[i] + 1e-8) / (sum_of_N + 1e-8 * number_of_cluster)

        pi_diff = np.sum(np.absolute(store_pi - pi))

        pi = store_pi

        print(iter)
        for i in range(number_of_cluster):
            print("cluster %d" % (i))
            for j in range(28):
                for k in range(28):
                    if mu[i][j * 28 + k] > 0.5:
                        print("1", end='')
                    else:
                        print("0", end='')
                print("")

        iter = iter + 1

        if abs(diff - mu_diff - pi_diff) < 20:
            return mu, pi
        diff = mu_diff + pi_diff

        print(mu_diff)
        print(pi_diff)

        if iter > 20:
            return mu, pi


def main():
    data_type = np.dtype("int32").newbyteorder('>')

    data = np.fromfile("../dataset/MNIST/raw/train-images-idx3-ubyte", dtype="ubyte")
    magic_number, number_of_images, number_of_rows, number_of_columns = np.frombuffer(data[:4 * data_type.itemsize],
                                                                                      data_type)

    data = data[4 * data_type.itemsize:].astype("int").reshape([number_of_images, number_of_rows * number_of_columns])

    labels = np.fromfile("../dataset/MNIST/raw/train-labels-idx1-ubyte", dtype="ubyte").astype("int")
    labels = labels[2 * data_type.itemsize:]

    one = np.ones(28)
    bin = convert_to_bin(data)
    # print(number_of_images)
    mu, pi = EM(bin, labels)
    predict = []

    for i in range(60000):
        prob = []
        for j in range(10):
            numerator = pi[j]
            for k in range(0, 784, 28):
                temp_mu = mu[j][k:k + 28]
                temp_data = bin[i][k:k + 28]
                numerator = numerator * 1e7
                numerator = numerator * np.prod(temp_mu ** temp_data) * np.prod((one - temp_mu) ** (one - temp_data))
            prob.append(numerator)
        predict.append(np.argmax(prob))

    result = np.zeros((10, 10))

    for i in range(60000):
        result[predict[i]][labels[i]] = result[predict[i]][labels[i]] + 1

    cluster_label = dict()
    for i in range(10):
        cluster_label[i] = -1

    for i in range(10):
        max = np.max(result)
        pos_y = int(np.argmax(result) % 10)
        pos_x = int(np.floor(np.argmax(result) / 10))

        cluster_label[pos_x] = pos_y
        for j in range(10):
            result[pos_x][j] = -1
            result[j][pos_y] = -1

    print(cluster_label)

    label_cluster = dict()
    for i in range(10):
        label_cluster[cluster_label[i]] = i

    acurracy = 0
    for i in range(10):
        tp = 0
        tn = 0
        fp = 0
        fn = 0
        for j in range(60000):
            if predict[j] == label_cluster[i] and labels[j] == i:
                tp = tp + 1
            if predict[j] != label_cluster[i] and labels[j] == i:
                fn = fn + 1
            if predict[j] == label_cluster[i] and labels[j] != i:
                fp = fp + 1
            if predict[j] != label_cluster[i] and labels[j] != i:
                tn = tn + 1
        acurracy = acurracy + tn + tp
        print("label %d" % (i))

        print("Confusion matrix:")

        print('%20s  number%d  %10s  not number%d' % ('Predict', i, 'Predict', i))
        print("Is number%d %10d%20d" % (i, tp, fn))
        print("Isn't number%d %10d%20d" % (i, fp, tn))

        print("Sensitivity (Successfully prefict number %d): %2f" % (i, tp / (tp + fn)))
        print("Specificity (Successfully predict not number%d): %2f" % (i, tn / (fp + tn)))
    print("error rate: %f" % (1 - acurracy / 600000))


if __name__ == '__main__':
    main()
