from keras.models import Sequential
from keras.layers import Dense, Activation
import numpy as np
import pandas as pd


# Model Template

img = np.load("images.npy")
labels = np.load("labels.npy")
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

print(train.shape, test.shape, val.shape)

# model = Sequential() # declare model
# model.add(Dense(10, input_shape=(28*28, ), kernel_initializer='he_normal'))  # first layer
# model.add(Activation('relu'))
# #
# #
# #
# # Fill in Model Here
# #
# #
# model.add(Dense(10, kernel_initializer='he_normal')) # last layer
# model.add(Activation('softmax'))
#
#
# # Compile Model
# model.compile(optimizer='sgd',
#               loss='categorical_crossentropy',
#               metrics=['accuracy'])
#
# print("all good")

# Train Model
# history = model.fit(x_train, y_train,
#                     validation_data = (x_val, y_val),
#                     epochs=10,
#                     batch_size=512)
#
#
# # Report Results
#
# print(history.history)
# model.predict()