# Basic learning testing
from keras.models import Sequential
from keras.layers import Dense, Activation
import numpy as np
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2" # shut up

def main():
    features = np.array([np.array([10*a for a in np.random.random(3)]) for _ in range(100)])
    labels = np.array([np.aray([b*c for b, c in enumerate(a)]) for a in features])

    model = Sequential()
    model.add(Dense(512, input_dim=3))
    model.add(Dense(512))
    model.add(Dense(3))

    model.compile(
        optimizer="adam",
        loss="mse",
        metrics=["accuracy"])

    model.fit(features, labels, epochs=5000, verbose=0)

    print "accuracy", model.evaluate(features, labels, verbose=0)
    print "expect", [1*0, 2*1, 3*2]
    print "got", model.predict(np.array([1.0,2.0,3.0]))

if __name__ == '__main__':
    main()
