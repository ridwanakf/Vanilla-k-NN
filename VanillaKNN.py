class VanillaKNN():
    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train

    def predict(self, X_test):
        predictions = []
        for row in X_test:
            label = self.closest(row)
            predictions.append(label)
        return predictions

    def closest(self, row):
        best_dist = euclidean_distance(row, self.X_train[0])
        best_index = 0

        for i in range(1, len(self.X_train)):
            if euclidean_distance(row, self.X_train[i]) < best_dist:
                best_dist = euclidean_distance(row, self.X_train[i])
                best_index = i
        return self.y_train[best_index]

    def euclidean_distance(a, b):
        if len(a) != len(b):
            print('Vectors are not in the same length!')
            return 0

        distance = 0
        for i in range(len(a)):
            distance += pow((a[i] - b[i]), 2)
        return distance
