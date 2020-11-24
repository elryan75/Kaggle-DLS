from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import pandas as pd
import zipfile
import matplotlib.pyplot as plt
import csv


def show_image(arr):
    two_d = (np.reshape(arr, (128, 128)) * 255).astype(np.uint8)
    plt.imshow(two_d, interpolation='nearest')
    plt.show()


###################
# LOAD IMAGES
###################
train_images = np.load("train_x.npy")
shape = train_images.shape
train_images = train_images.reshape(shape[0], shape[1], shape[2], 1)
test_images = np.load("test_x.npy")
y_train = pd.read_csv('train_y.csv')
y_train = y_train['Label'].to_numpy()



###################
# PREPROCESS THE DATA
###################
scaled_train_images = []
for i in train_images:
  scaled_train_images.append((i/255).astype(np.uint8))

scaled_train_images = np.array(scaled_train_images)
show_image(train_images[1])
show_image(scaled_train_images[1])



y = keras.utils.to_categorical(y_train, num_classes=10, dtype="float32")


###################
# MODEL DEFINITION
###################
input_shape=(128,128,1)
model = keras.Sequential()

model.add(keras.layers.InputLayer(input_shape))

model.add(keras.layers.Conv2D(strides=1, filters=15, kernel_size=(5, 5), padding='same', activation='relu'))
model.add(keras.layers.MaxPool2D(pool_size=(2,2)))
model.add(keras.layers.Dropout(0.2))

model.add(keras.layers.Conv2D(strides=1, filters=30, kernel_size=(10, 10), padding='same', activation='relu'))
model.add(keras.layers.MaxPool2D(pool_size=(2,2)))
model.add(keras.layers.Dropout(0.2))

model.add(keras.layers.Conv2D(strides=1, filters=10, kernel_size=(10, 10), padding='same', activation='relu'))
model.add(keras.layers.MaxPool2D(pool_size=(2,2)))
model.add(keras.layers.Dropout(0.2))

model.add(keras.layers.Flatten())

model.add(keras.layers.Dense(20, activation='relu'))

outputs = [keras.layers.Dense(10, activation='softmax')(model.output) for _ in range(1)]

model = keras.Model(model.input, outputs)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['categorical_accuracy'])

print(model.summary())

###################
# MODEL TRAINING
###################

model.fit(scaled_train_images, y, 
          batch_size=6,
          epochs=25,
          verbose=2)


###################
# PREDICT TEST IMAGES
###################

scaled_test_images = []
for i in test_images:
  scaled_test_images.append((i/255).astype(np.uint8))

scaled_test_images = np.array(scaled_test_images)


predict=model.predict(scaled_test_images)


###################
# SAVE PREDICTIONS IN CSV
###################

predictions = []
predictions.append(['Id', 'Label'])

for i in range(0,scaled_test_images.shape[0]):
  p = np.argmax(predict[i])
  predictions.append([i, p])

with open('predictions.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerows(predictions)




