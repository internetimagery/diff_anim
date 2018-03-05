# Learn the pattern between the inputs
from __future__ import print_function
from keras.layers import Dense, Activation
from keras.models import Sequential
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2" # shut up

class Brain(object):
    def __init__(s):
        s.model = None

    def load_state(s, path):
        pass

    def save_state(s, path):
        pass

    def learn(s, features, labels):
        if not s.model:
            s.model = model = Sequential()
            model.add(Dense(512, input_dim=len(features[0])))
            model.add(Dense(512))
            model.add(Dense(len(labels[0])))
            model.compile(
                optimizer="adam",
                loss="mse",
                metrics=["accuracy"])

        s.model.fit(features, labels, epochs=5000, verbose=0)
        return s

    def evaluate(s, features, labels):
        if not s.model:
            raise RuntimeError("Model not yet created")
        return s.model.evaluate(features, labels, verbose=0)

    def predict(s, features):
        return model.predict(features)
