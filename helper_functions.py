import random

def train_test_split(X, y, split_size):
    '''
    :param X: data
    :param y: label
    :param split_size: split size (example: split_size = .2 -> test=20%, train=80%)
    :return: X_train, X_test, y_train, y_test
    '''

    X_train = []
    X_test = []
    y_train = []
    y_test = []

    train_len = int((1-split_size) * len(X)) + 1
    
    list_random = [i for i in range(len(X))]
    train_index = []

    for i in range(train_len):
        random_index = random.choice(list_random)
        while list_random[random_index] == -1:
            random_index = random.choice(list_random)

        train_index.append(random_index)
        list_random[random_index] = -1

    for i in range(len(X)):
        if i in train_index:
            X_train.append(X[i])
            y_train.append(y[i])
        else:
            X_test.append(X[i])
            y_test.append(y[i])

    return X_train, X_test, y_train, y_test


def accuracy_test(true_label, prediction):
    '''
    :param true_label: list contains true label
    :param prediction: list contains predicted value
    :return accuracy: accuracy score
    '''

    if len(true_label) != len(prediction):
        print('Panjang list tidak sama!')
        return

    n_true = 0
    for i in range(len(prediction)):
        if true_label[i] == prediction[i]:
            n_true += 1

    accuracy = (n_true / len(prediction)) * 100

    return accuracy
