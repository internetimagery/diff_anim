# Learn the pattern between the inputs
from __future__ import print_function
from keras.layers import Dense, Activation
from keras.models import Sequential, model_from_json
import os.path
import json
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2" # shut up

class Brain(object):
    def __init__(s):
        s._model = None
        s.weights = "weights.hdf5"
        s.struct = "struct.json"
        s.meta = "metadata.json"

    def _compile(s):
        s._model.compile(
            optimizer="RMSprop",
            # optimizer="adam",
            loss="mse",
            metrics=["accuracy"])

    def load_state(s, path):
        if not os.path.isdir(path):
            raise RuntimeError("Path does not exist: %s" % path)
        weight_path = os.path.join(path, s.weights)
        struct_path = os.path.join(path, s.struct)
        meta_path = os.path.join(path, s.meta)
        if not os.path.isfile(meta_path) not os.path.isfile(weight_path) or not os.path.isfile(struct_path):
            raise RuntimeError("Weights, meta, and structs cannot be found at %s" % path)
        with open(struct_path, "r") as f:
            s._model = model_from_json(f.read())
        with open(meta_path, "r") as f:
            metadata = json.load(f)
        s._model.load_weights(weight_path)
        s._compile()
        return s

    def save_state(s, path):
        if not s._model:
            raise RuntimeError("Machine not yet trained.")
        if not os.path.isdir(path):
            raise RuntimeError("Path does not exist: %s" % path)
        weight_path = os.path.join(path, s.weights)
        struct_path = os.path.join(path, s.struct)
        meta_path = os.path.join(path, s.meta)
        with open(struct_path, "w") as f:
            f.write(s._model.to_json())
        with open(meta_path, "w") as f:
            json.dump({}, f, indent=4)
        s._model.save_weights(weight_path)
        return s

    def train(s, features, labels, epochs=2000, debug=False):
        if not s._model:
            s._model = model = Sequential()
            model.add(Dense(512, input_dim=len(features[0])))
            model.add(Dense(512))
            model.add(Dense(len(labels[0])))
            s._compile()
        print("Training. Please wait...")
        s._model.fit(features, labels, epochs=epochs, verbose=1 if debug else 0)
        return s

    def evaluate(s, features, labels, debug=False):
        if not s._model:
            raise RuntimeError("Machine not yet trained.")
        return s._model.evaluate(features, labels, verbose=1 if debug else 0)

    def predict(s, features):
        if not s._model:
            raise RuntimeError("Machine not yet trained.")
        return s._model.predict(features)
