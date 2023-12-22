import sys
sys.path.append('/Users/yueteng/opt/anaconda3/lib/python3.9/site-packages')
import os
import cv2
import numpy as np
import tensorflow
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D

directory = "/Users/yueteng/Desktop/faces"
file_list = [os.path.join(directory, f) for f in os.listdir(directory) if os.path.isdir(os.path.join(directory, f))]

#importint the data
x_data = []
y_data = []
for folder in file_list:
    folder_path = os.path.join(directory, folder)
    for image_file in os.listdir(folder_path):
        if not image_file.endswith('.pgm'):
            continue

        image_path = os.path.join(folder_path, image_file)
        if not os.path.isfile(image_path) or not image_file.endswith(".pgm"):
            continue
        
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (30, 32))
        x_data.append(img)
        
        if "neutral" in image_file:
            y_data.append(1)
        elif "happy" in image_file:
            y_data.append(2)
        elif "sad" in image_file:
            y_data.append(3)
        elif "angry" in image_file:
            y_data.append(4)
        else:
            y_data.append(5)


x_data = np.array(x_data)
y_data = np.array(y_data)

x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.2, random_state=42)
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=42)

num_classes = 5
y_train = to_categorical(y_train, num_classes)
y_val = to_categorical(y_val, num_classes)
y_test = to_categorical(y_test, num_classes)


input_shape = (32, 30, 1)
x_train = x_train.reshape(x_train.shape[0], 32, 30, 1)
x_val = x_val.reshape(x_val.shape[0], 32, 30, 1)
x_test = x_test.reshape(x_test.shape[0], 32, 30, 1)


model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(256, kernel_size=(2, 2), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss=tensorflow.keras.losses.categorical_crossentropy,
              optimizer=tensorflow.keras.optimizers.Adam(),
              metrics=['accuracy'])

model.fit(x_train, y_train,
          batch_size=32,
          epochs=30,
          verbose=1,
          validation_data=(x_val, y_val))

score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])


#prediction
img = cv2.imread('/Users/yueteng/Desktop/faces/test.jpg', cv2.IMREAD_GRAYSCALE)
img = cv2.resize(img, (32, 30))
img = img.reshape(1, 32, 30, 1)
img = img / 255.0
pred = model.predict(img)
label = np.argmax(pred)
print("Predicted label:", label)
