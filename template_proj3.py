import os
os.add_dll_directory("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.4/bin")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.utils import np_utils
import numpy as np
import pandas as pd

# Model Template

img = np.load("images.npy")
labels = np.load("labels.npy")
hot_labels = np_utils.to_categorical([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], num_classes=10)
# print(img.shape)
# print(labels.shape)
# print(img[1, :])
# print(labels[1])

df = pd.DataFrame(img)
# print(df)

df[784] = labels


#
# print(df.describe())


def split(df: pd.DataFrame):
    df_train = df.sample(frac=.6)
    df = df.drop(df_train.index)
    df_test = df.sample(frac=.625)
    df_val = df.drop(df_test.index)

    return df_train, df_test, df_val


train_frames = []
test_frames = []
val_frames = []
for i in range(10):
    df_new = df[df[784] == i]
    df_new_train, df_new_test, df_new_val = split(df_new)
    train_frames.append(df_new_train)
    test_frames.append(df_new_test)
    val_frames.append(df_new_val)

train = pd.concat(train_frames)
train = train.sample(frac=1)  # shuffle
test = pd.concat(test_frames)
test = test.sample(frac=1)
val = pd.concat(val_frames)
val = val.sample(frac=1)


# print(train.shape, test.shape, val.shape)


"""Converts dataframe outputs to ndarrays for keras usage"""


def get_fitting_outputs(sample):
    output = np.empty((0, 10))
    # get last column and convert to hot-vector
    # print(sample.iloc[:, -1:])
    for index, row in sample.iterrows():
        # print(row[-1:])
        output = np.append(output, hot_labels[row[-1:]], axis=0)
    return output


x_train = train.drop(columns=784).to_numpy()
y_train = get_fitting_outputs(train)

x_val = val.drop(columns=784).to_numpy()
y_val = get_fitting_outputs(val)

x_test = test.drop(columns=784).to_numpy()
y_test = test.iloc[:, -1:].to_numpy()

# print(x_test)
# print(y_test)

# print(x_train)
# print(y_train)
# print(x_train.shape, y_train.shape)

model = Sequential()  # declare model
model.add(Dense(10, input_shape=(28 * 28,), kernel_initializer='he_normal'))  # first layer
model.add(Activation('relu'))
#
#
#
# Fill in Model Here
model.add(Dense(10, kernel_initializer='he_normal'))
model.add(Activation('sigmoid'))
# model.add(Dense(10, kernel_initializer='he_normal'))
# model.add(Activation('softplus'))
# model.add(Dense(10, kernel_initializer='he_normal'))
# model.add(Activation('softsign'))
model.add(Dense(10, kernel_initializer='he_normal'))
model.add(Activation('tanh'))
#
#
model.add(Dense(10, kernel_initializer='he_normal'))  # last layer
model.add(Activation('softmax'))

# Compile Model
model.compile(optimizer='sgd',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train Model
history = model.fit(x_train, y_train,
                    validation_data=(x_val, y_val),
                    epochs=80,
                    batch_size=512)

# Report Results

print(history.history)
predictions = model.predict(x=x_test)
print(predictions.shape)

"""Converts hot_vector to categorical value"""


def hot_to_num(results):
    categories = np.empty((0, len(results)))
    for hot_vector in results:
        # print(hot_vector)
        categories = np.append(categories, np.where(hot_vector == np.amax(hot_vector)))
    return categories


predictions = hot_to_num(predictions)
# print(predictions)

confusion_matrix = np.zeros((10, 10))
for iteration, prediction in enumerate(predictions):
    # print(prediction)
    np.add.at(confusion_matrix, tuple(np.array([int(y_test[iteration]), int(prediction)]).T), 1)

confusion_matrix = pd.DataFrame(confusion_matrix)
print(confusion_matrix)
