import keras
from keras.models import Sequential
from keras.layers import Activation
from keras.layers import Dense
from keras.optimizers import Adam
from keras.metrics import categorical_crossentropy
import numpy as np
from random import randint
from sklearn.preprocessing import MinMaxScaler


#create mock data for TRAINING
train_lables = []
train_samples = []

for i in range(50):
    random_younger = randint(13, 64)
    train_samples.append(random_younger)
    train_lables.append(1)

    random_older = randint(65, 100)
    train_samples.append(random_older)
    train_lables.append(0)

for i in range(1000):
    random_younger = randint(13, 64)
    train_samples.append(random_younger)
    train_lables.append(0)

    random_older = randint(65, 100)
    train_samples.append(random_older)
    train_lables.append(1)

#convert to numpy arrays
train_lables = np.array(train_lables)
train_samples = np.array(train_samples)

#scale to 0-1
scaler = MinMaxScaler(feature_range=(0,1))
scaled_train_samples = scaler.fit_transform((train_samples).reshape(-1,1))


#create mock data for PREDICTIONS/TESTING
test_lables = []
test_samples = []

for i in range(10):
    random_younger = randint(13, 64)
    test_samples.append(random_younger)
    test_lables.append(1)

    random_older = randint(65, 100)
    test_samples.append(random_older)
    test_lables.append(0)

for i in range(200):
    random_younger = randint(13, 64)
    test_samples.append(random_younger)
    test_lables.append(0)

    random_older = randint(65, 100)
    test_samples.append(random_older)
    test_lables.append(1)

#convert to numpy arrays
test_lables = np.array(test_lables)
test_samples = np.array(test_samples)

#scale to 0-1
scaler2 = MinMaxScaler(feature_range=(0,1))
scaled_test_samples = scaler2.fit_transform((test_samples).reshape(-1,1))

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
model.fit(scaled_train_samples, train_lables, validation_split=0.1, batch_size=10, epochs=20, shuffle=True, verbose=2)

predictions = model.predict(scaled_test_samples, batch_size=10, verbose=0)

for i in predictions:
    print(i)

rounded_predictions = model.predict_classes(scaled_test_samples, batch_size=10, verbose=0)

for i in rounded_predictions:
    print(i)
