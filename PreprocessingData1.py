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
for i in scaled_train_samples:
    print(i)
