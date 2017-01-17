from sklearn.cross_validation import train_test_split
from sklearn.datasets import fetch_lfw_people
from keras.constraints import maxnorm
from keras.layers import Dense, Flatten, Dropout, Activation
from keras.layers import Convolution2D, MaxPooling2D
from keras.models import Sequential
from keras.utils import np_utils
from keras import backend as K
import numpy

K.set_image_dim_ordering('th')

#download dataset of images 250*250
lfw_people = fetch_lfw_people(min_faces_per_person=10, resize=0.4, download_if_missing=True)

X = lfw_people.images
Y = lfw_people.target
_, img_rows, img_cols = X.shape

#preparing data
num_classes = numpy.unique(Y).shape[0]
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=42, stratify=lfw_people.target)

X_train = X_train.reshape(X_train.shape[0], 1, img_rows, img_cols)
X_test = X_test.reshape(X_test.shape[0], 1, img_rows, img_cols)

X_train = X_train / 255.0
X_test = X_test / 255.0

Y_train = np_utils.to_categorical(Y_train, num_classes)
Y_test = np_utils.to_categorical(Y_test, num_classes)

#model
model = Sequential()
model.add(Convolution2D(32, 3, 3, input_shape=X_train.shape[1:], activation='relu', border_mode='same'))
model.add(Activation('relu'))
model.add(Convolution2D(32, 3, 3, activation='relu', border_mode='same'))
model.add(Dropout(0.2))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Convolution2D(64, 3, 3, activation='relu', border_mode='same'))
model.add(Dropout(0.2))
model.add(Convolution2D(64, 3, 3, activation='relu', border_mode='same'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(1024, activation='relu', W_constraint=maxnorm(3)))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))
epochs = 20

model.compile(optimizer="adam", loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, Y_train, nb_epoch=epochs, batch_size=32,  verbose=1)

accuracy = model.evaluate(X_test, Y_test, verbose=0)[1] * 100

print("Accuracy: %.2f%%" % accuracy)

