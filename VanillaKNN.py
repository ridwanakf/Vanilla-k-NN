# Created by Ridwan Afwan Karim Fauzi
# https://www.github.com/ridwanakf

class VanillaKNN():
    def fit(self, X_train, y_train, num_of_k=1):
        self.X_train = X_train
        self.y_train = y_train
        self.num_of_k = num_of_k
        self.k_list_x = []
        self.k_list_y = []

    def predict(self, X_test):
        predictions = []
        for row in X_test:
            label = self.closest(row)
            predictions.append(label)
        return predictions

    def closest(self, row):
        self.k_list_x.clear()
        self.k_list_x.append(self.euclidean_distance(row, self.X_train[0]))

        self.k_list_y.clear()
        self.k_list_y.append(0)

        for i in range(1, len(self.X_train)):
            dist_now = self.euclidean_distance(row, self.X_train[i])
            if len(self.k_list_x) < self.num_of_k:
                self.k_list_x.append(dist_now)
                self.k_list_y.append(i)
            elif dist_now < max(self.k_list_x):
                index_max = self.k_list_x.index(max(self.k_list_x))
                self.k_list_x[index_max] = dist_now
                self.k_list_y[index_max] = i

        # get all classes of nearest neighbors
        y_predict = [y_train[self.k_list_y[i]]
                     for i in range(len(self.k_list_y))]

        # count for the most appeared class
        predicted = list(max(zip((y_predict.count(item)
                                  for item in set(y_predict)), set(y_predict))))

        # if all of the nearest classes are unique, get the closest one
        if len(set(y_predict)) == self.num_of_k:
            index_min = self.k_list_x.index(min(self.k_list_x))
            return y_train[self.k_list_y[index_min]]
        else: # else got the most appeared class
            return predicted[1]

    def euclidean_distance(self, a, b):
        if len(a) != len(b):
            print('Vectors are not in the same length!')
            return 0

        distance = 0
        for i in range(len(a)):
            distance += pow((a[i] - b[i]), 2)
        return distance
