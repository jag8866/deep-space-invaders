'''
Creates a neural network and trains it on human play data, then saves it to disk
and outputs a graph of training and validation loss.
'''

from keras import *
from keras.utils import to_categorical
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

#Load our human play data, X being frames and Y being actions taken
X = np.load('X.npy')
Y = to_categorical(np.load('Y.npy'))

#Build the network one layer at a time
model = Sequential()
#Lambda layer to normalize the data
model.add(layers.Lambda(lambda x: x / 255.0, input_shape=(105, 80, 1)))
#Convolutional layer (good for learning spatial data like patterns and objects in images, perfect for game frames)
model.add(layers.Conv2D(12, 3, activation='relu'))
model.add(layers.Dropout(.4))
model.add(layers.BatchNormalization())
model.add(layers.Flatten())
model.add(layers.Dense(16, activation='relu'))
model.add(layers.Dropout(.4))
model.add(layers.Dense(4, activation='softmax'))

#Build it and print out a summary showing its structure
model.build()
print(model.summary())

#Now compile the model and fit it to the data.
model.compile(loss='mean_squared_error', optimizer=optimizers.RMSprop(lr=0.00025, rho=0.95, epsilon=0.01))
try:
    history = model.fit(X, Y, validation_split=0.1, epochs=500, batch_size=8, verbose=2)
except Exception as ex:
    f = open('errorlog.txt','w')
    print >>f, ex

#Save the structure of the net to a json file and the weights to an h5 file
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
model.save_weights("model.h5")

#Now we put together a nice graph of the loss and validation loss and output it
#Helps test for overfitting and other issues
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(len(loss))
plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title("Second CNN Attempt - Training and Validation Loss")
plt.legend()
plt.savefig('cnn_loss_valloss.png')
plt.show()
