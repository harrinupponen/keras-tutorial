import tensorflow as tf
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import categorical_crossentropy
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import itertools

train_path = 'men_women/train'
valid_path = 'men_women/valid'
test_path = 'men_women/test'

train_batches = ImageDataGenerator().flow_from_directory(train_path, target_size=(224, 224), classes=['man', 'woman'], batch_size=10)
valid_batches = ImageDataGenerator().flow_from_directory(valid_path, target_size=(224, 224), classes=['man', 'woman'], batch_size=4)
test_batches = ImageDataGenerator().flow_from_directory(test_path, target_size=(224, 224), classes=['man', 'woman'], batch_size=10)

vgg16_model = tf.keras.applications.vgg16.VGG16()

model = tf.keras.Sequential()
for layer in vgg16_model.layers[:-1]:
    model.add(layer)

for layer in model.layers:
    layer.trainable = False

model.add(Dense(2, activation='softmax'))

model.summary()

model.compile(Adam(lr=.0001), loss='categorical_crossentropy', metrics=['accuracy'])
model.fit_generator(train_batches, steps_per_epoch=10, validation_data=valid_batches,
                    validation_steps=10, epochs=5, verbose=2)

#predictions = model.predict_generator(test_batches, steps=1, verbose=0)
#print(predictions)

model.save('men_women_vgg16_based_model2.h5')