import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import collections
# load data
train = pd.read_csv('/home/quan/PycharmProjects/sign-language-gesture-recognition/Data/Test.csv')
# test = pd.read_csv('E:\PROJECTS\PyCharm_Projects\sign-language-gesture-recognition\Test2\Test_file.csv')

# numbers of data
x = train.iloc[:, 1:].values
# labels of data
y = train.iloc[:, :1].values.flatten() #Return a copy of the array collapsed into one dimension.

#definite each batch of data
def next_batch(batch_size, data, labels):
    idx = np.arange(0, len(data))
    np.random.shuffle(idx)
    idx = idx[: batch_size]
    data_shuffle = [data[i] for i in idx]
    labels_shuffle = [labels[i] for i in idx]
    return np.asarray(data_shuffle), np.asarray(labels_shuffle)

# display data
def display_images(data):
    x, y = data
    fig, axes = plt.subplots(3, 3)
    fig.subplots_adjust(hspace = 0.5, wspace = 0.5)
    for i, ax in enumerate(axes.flat):
        ax.imshow(x[i].reshape(64, 64), cmap = 'binary')
        ax.set_xlabel(chr(y[i] + 65))
        ax.set_xticks([])
        ax.set_yticks([])
    plt.show()
    pass
def frequency(y):
    unique_val = np.array(y)
    z = np.unique(unique_val)

    z = dict(collections.Counter(list(y)))
    print("z: ", z)
    frequencies = [z[i] for i in z.keys()]
    labels = [chr(i + 65) for i in z.keys()]

    plt.figure(figsize=(15, 5))
    plt.bar(labels, frequencies)
    plt.title('Frequency Distribution of Alphabets', fontsize=20)
    plt.show()
    pass


print('Dataframe Shape:', train.shape)
print("Number of pixels in each image:", x.shape[1])

# from sklearn.model_selection import train_test_split
# X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.33, random_state = 42)
# print('X train shape', X_train.shape)
# print('y train shape', y_train.shape)
# print('X test shape', X_test.shape)
# print('y test shape', y_test.shape)
# print('label : ', y.shape[0])


display_images(next_batch(9, x, y))


