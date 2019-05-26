import VanillaKNN as vknn
from helper_functions import *

### IMPORTING AND CLEANING DATASET ###
'''
    For this example, i used dataset of a bunch of animal classes and their features.
    first element of the list is the animal class, second element is the animal subclass, and the rest of them are its features.
    so on this example, i'll just add the first element to label list, and add third to the last elements to feature list.
'''
# Read the txt file, parse it, and then put it on a list
zoo_data = [line.rstrip('\n') for line in open('zoo_data.txt', 'r')]
zoo_data = [zoo.split(',') for zoo in zoo_data]

zoo_label = []

# Getting the first element to the zoo_label list, and drop the second element for the feature list (3rd-last elements)
for i in range(len(zoo_data)):
    zoo_label.append(int(zoo_data[i][0]))
    del zoo_data[i][0:2]
    zoo_data[i] = [int(zoo) for zoo in zoo_data[i]]

# It's just for knowing how much samples each class has
num_data_per_class = [0, 0, 0, 0, 0, 0, 0]
for i in range(len(zoo_label)):
    num_data_per_class[zoo_label[i] - 1] += 1
for i in range(len(num_data_per_class)):
    print('Class no:', i + 1, ' has', num_data_per_class[i], ' samples.')

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(zoo_data, zoo_label, split_size=.2)


### START INFERNCE ###

# Instantiate classifier
my_classifier = vknn.VanillaKNN()

# Try classifying using k = [1:10]
for i in range(1,11):
    print('//////////////// k =', i, ' ////////////////')

    my_classifier.fit(X_train, y_train, num_of_k=i)
    predictions = my_classifier.predict(X_test)

    print('accuracy :', accuracy_test(y_test, predictions))
    print()
