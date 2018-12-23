'''
Collects all the files in ./Human_Play_Data and wraps them all up into one big data file.
'''

import numpy as np
import os

X = None
Y = None

for filename in os.listdir("./Human_Play_Data"):
    if filename.startswith("X_"):
        if not isinstance(X, np.ndarray):
            X = np.load("./Human_Play_Data/" + filename)
        else:
            X = np.append(X, np.load("./Human_Play_Data/" + filename), axis=0)

        if not isinstance(Y, np.ndarray):
            Y = np.load("./Human_Play_Data/" + "Y" + filename[1:])
        else:
            Y = np.append(Y, np.load("./Human_Play_Data/" + "Y" + filename[1:]))

np.save("X.npy", X)
np.save("Y.npy", Y)


       
