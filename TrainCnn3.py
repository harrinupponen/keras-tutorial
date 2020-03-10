import keras
from keras.models import Sequential
from keras.layers import Activation
from keras.layers import Dense
from keras.optimizers import Adam
from keras.metrics import categorical_crossentropy
import numpy as np
from random import randint
from sklearn.preprocessing import MinMaxScaler

train_lables = []
train_samples = []

#create mock data
for i in range(1000):
    random_younger = randint(13, 64)
    train_samples.append(random_younger)
    train_lables.append(0)

    random_older = randint(65, 100)
    train_samples.append(random_older)
    train_lables.append(1)

for i in range(50):
    random_younger = randint(13, 64)
    train_samples.append(random_younger)
    train_lables.append(1)

    random_older = randint(65, 100)
    train_samples.append(random_older)
    train_lables.append(0)

#print raw data
""" for i in train_samples:
    print(i)

for i in train_lables:
    print(i) """

#convert to numpy arrays
train_lables = np.array(train_lables)
train_samples = np.array(train_samples)

#scale to 0-1
scaler = MinMaxScaler(feature_range=(0,1))
scaled_train_samples = scaler.fit_transform((train_samples).reshape(-1,1))

#print scaled data
#for i in scaled_train_samples:
#   print(i)

#create the model (layers)
model = Sequential([
    Dense(16, input_shape=(1,), activation='relu'),
    Dense(32, activation='relu'),
    Dense(2, activation='softmax')
])

#print out the details of the model
#model.summary()

#compile the model
model.compile(Adam(lr=.0001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(scaled_train_samples, train_lables, batch_size=10, epochs=20, shuffle=True, verbose=2)
