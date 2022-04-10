from __future__ import print_function
import keras
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
import os

batch_size = 32
num_classes = 10
epochs = 100
data_augmentation = True
num_predictions = 20
save_dir = os.path.join(os.getcwd(), 'saved_models')
model_name = 'keras_cifar10_trained_model.h5'

# The data, split between train and test sets:
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

print('Training samples:', x_train.shape[0])
print('Testing samples:', x_test.shape[0])

# 256 Pixels
x_train = x_train / 255
y_train = y_train / 255

model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
model.add(MaxPooling2D((2,2)))
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(Flatten())
model.add(Dense(32, activation='relu'))
model.add(Dense(10, activation='softmax'))

# Adam Optimizer
optimizer = keras.optimizers.Adam(learning_rate=0.0001)

# Train model on Categorical Cross Entropy
model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizer)

model.summary()

model.fit(x_train, x_test, epochs=10)

test_loss, test_accuracy = model.evaluate(y_train, y_test, verbose=2)

print(test_loss)
print(test_accuracy)
