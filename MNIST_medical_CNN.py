#importing necessary librarires
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import np_utils
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D

# path for csv file
dataset ='PATH_TO_CSV/medical_mnist.csv'

dataset = np.array(pd.read_csv(dataset))
m, n= dataset.shape

# separating data and labels
labels = dataset[:, 0].reshape(m,1) # labels are column 0
data = dataset[:, 1:]               # data is from column 1-end

# Train-test split
x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

X_train=x_train.reshape(x_train.shape[0], 1,64,64).astype('float32')
X_test=x_test.reshape(x_test.shape[0], 1,64,64).astype('float32')

# normalizing data
X_train=X_train/255
X_test=X_test/255
y_train= np_utils.to_categorical(y_train)
y_test= np_utils.to_categorical(y_test)
num_classes=y_train.shape[1]
print(num_classes)

#defining a function for cnn
def cnn_model():
    model=Sequential()
    model.add(Conv2D(32,5,5, padding='same',input_shape=(1,64,64), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2), padding='same'))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

model=cnn_model()
model.fit(X_train, y_train, validation_data=(X_test,y_test),epochs=10, batch_size=200, verbose=2)
score= model.evaluate(X_test, y_test, verbose=0)
print('The error is: %.2f%%'%(100-score[1]*100))





