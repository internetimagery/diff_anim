# Learn the pattern between the inputs
from __future__ import print_function
from keras.layers import Dense, Activation
from keras.models import Sequential
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2" # shut up

class Brain(object):
    def __init__(s):
        s._model = None

    def load_state(s, path):
        pass

    def save_state(s, path):
        pass

    def train(s, features, labels, epochs=1000):
        if not s._model:
            s._model = model = Sequential()
            model.add(Dense(512, input_dim=len(features[0])))
            model.add(Dense(512))
            model.add(Dense(len(labels[0])))
            model.compile(
                optimizer="adam",
                loss="mse",
                metrics=["accuracy"])

        print("Training. Please wait...")
        s._model.fit(features, labels, epochs=epochs, verbose=0)
        return s

    def evaluate(s, features, labels):
        if not s._model:
            raise RuntimeError("Model not yet created")
        return s._model.evaluate(features, labels, verbose=0)

    def predict(s, features):
        if not s._model:
            raise RuntimeError("Model not yet created")
        return s._model.predict(features)
