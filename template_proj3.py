import os

os.add_dll_directory("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.4/bin")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.utils import np_utils
from keras.metrics import Precision, Recall
from PIL import Image
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Model Template
"""PRE PROCESSING"""
img = np.load("images.npy")
labels = np.load("labels.npy")
hot_labels = np_utils.to_categorical([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], num_classes=10)
# print(img.shape)
# print(labels.shape)
# print(img[1, :])
# print(labels[1])

df = pd.DataFrame(img)
# print(df)

df[784] = labels  # column for matching correct labels

#
# print(df.describe())


"""Splits the data by category for stratified sampling"""


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

"""MAKE MODEL"""

model = Sequential()  # declare model
model.add(Dense(20, input_shape=(28 * 28,), kernel_initializer='he_normal'))  # first layer
model.add(Activation('selu'))
#
#
#
# Fill in Model Here

model.add(Dense(40, kernel_initializer='glorot_normal'))
model.add(Activation('tanh'))

model.add(Dense(30, kernel_initializer='random_normal'))
model.add(Activation('tanh'))

model.add(Dense(20, kernel_initializer='random_normal'))
model.add(Activation('relu'))

model.add(Dense(15, kernel_initializer='he_normal'))
model.add(Activation('tanh'))

#
#
model.add(Dense(10, kernel_initializer='he_normal'))  # last layer
model.add(Activation('softmax'))

# Compile Model
model.compile(optimizer='sgd',
              loss='categorical_crossentropy',
              metrics=['accuracy', Precision(), Recall()])

# Train Model
history = model.fit(x_train, y_train,
                    validation_data=(x_val, y_val),
                    epochs=1000,
                    batch_size=200,
                    verbose=0)

# Report Results

# print(max(history.history['accuracy']))
predictions = model.predict(x=x_test)
# print(predictions.shape)

print("GETTING REPORT")
"""Converts hot_vector to categorical value"""


def hot_to_num(results):
    categories = np.empty((0, len(results)))
    for hot_vector in results:
        # print(hot_vector)
        categories = np.append(categories, np.where(hot_vector == np.amax(hot_vector)))
    return categories


predictions = hot_to_num(predictions)
# print(predictions)

"""MAKE CONFUSION MATRIX"""
confusion_matrix = np.zeros((10, 10))
for iteration, prediction in enumerate(predictions):
    # print(prediction)
    np.add.at(confusion_matrix, tuple(np.array([int(y_test[iteration]), int(prediction)]).T), 1)

"""MAKE INCORRECT IMAGES"""
wrong_index = [0, 0, 0]
counter = 0
for iteration, prediction in enumerate(predictions):
    # print(prediction)
    if counter == 3:
        break
    if y_test[iteration] != prediction:
        wrong_index[counter] = iteration
        counter += 1

for i in range(3):
    wrong_image = np.reshape(x_test[wrong_index[i]], (28, 28))
    Image.fromarray(wrong_image).save('incorrect' + str(i) + '.png')

"""GET ACCURACY,PRECISION,RECALL"""


def get_prediction_accuracy(results):
    total = len(results)
    count = 0
    for counter, result in enumerate(results):
        if y_test[counter] == result:
            count += 1
    return float(count / total)


def get_prediction_precision(confusion):
    true_positive = 0
    false_positive = 0

    for i in range(10):
        true_positive += confusion[i, i]
        for j in range(10):
            if i != j:
                false_positive += confusion[j, i]
    return true_positive / (true_positive + false_positive)


print(get_prediction_accuracy(predictions))
print(history.history["precision"][-1])
print(history.history["recall"][-1])

"""COMMENT OR UNCOMMENT THIS LINE BELOW TO ACTUALLY SAVE/NOT SAVE MODEL"""
# model.save('best_trained_model', save_format='tf')

"""PRINT CONFUSION MATRIX"""
confusion_matrix = pd.DataFrame(confusion_matrix)
print(confusion_matrix)


"""PLOTS FROM TRAINING"""
f, (ax1, ax2, ax3) = plt.subplots(1, 3)
ax1.plot(history.history["accuracy"])
ax1.plot(history.history['val_accuracy'])
ax1.set_title('model accuracy')
ax1.set(xlabel='epoch', ylabel='accuracy')
ax1.legend(['train', 'validate'], loc='upper left')

ax2.plot(history.history['precision'])
ax2.plot(history.history['val_precision'])
ax2.set_title('model precision')
ax2.set(xlabel='epoch', ylabel='precision')
ax2.legend(['train', 'validate'], loc='upper left')

ax3.plot(history.history['recall'])
ax3.plot(history.history['val_recall'])
ax3.set_title('model recall')
ax3.set(xlabel='epoch', ylabel='recall')
ax3.legend(['train', 'validate'], loc='upper left')
plt.show()
