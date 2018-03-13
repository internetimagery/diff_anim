# Learn the pattern between the inputs
from __future__ import print_function
from keras import losses
from keras.layers import Dense, Activation, Dropout
from keras.models import Sequential, model_from_json
import numpy as np
import os.path
import json
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2" # shut up tensorflow

class Brain(object):
    def __init__(s):
        s._model = None
        s._metadata = {}
        s.weights = "weights.hdf5"
        s.struct = "struct.json"
        s.meta = "metadata.json"

    def __getitem__(s, item):
        return s._metadata[item]

    def __setitem__(s, item, val):
        s._metadata[item] = val

    def _compile(s):
        s._model.compile(
            # optimizer="RMSprop",
            optimizer="adam",
            # optimizer="SGD",
            loss="mean_squared_error",
            # loss="mean_absolute_error",
            # loss="mean_squared_logarithmic_error",
            metrics=["accuracy"])

    def _format_named(s, data):
        """ Format dict rows into vector. ie: [{col1:val,col2:val},{col1:val,col2:val}, ... ] """
        cols = s._metadata.get("cols", [])
        res = []
        try:
            for row in data:
                if not cols:
                    cols = row.keys()
                    s._metadata["cols"] = cols
                res.append(np.array([row[a] for a in cols]))
        except KeyError as err:
            print(err)
            raise RuntimeError("Not all columns present. %s" % cols)

        if not res:
            raise RuntimeError("Empty data.")
        return np.array(res)

    def load_state(s, path):
        if not os.path.isdir(path):
            raise RuntimeError("Path does not exist: %s" % path)
        weight_path = os.path.join(path, s.weights)
        struct_path = os.path.join(path, s.struct)
        meta_path = os.path.join(path, s.meta)
        if not os.path.isfile(meta_path) or not os.path.isfile(weight_path) or not os.path.isfile(struct_path):
            raise OSError("Weights, meta, and structs cannot be found at %s" % path)
        with open(struct_path, "r") as f:
            s._model = model_from_json(f.read())
        with open(meta_path, "r") as f:
            s._metadata = json.load(f)
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
            json.dump(s._metadata, f, indent=4)
        s._model.save_weights(weight_path)
        return s

    def train(s, features, labels, epochs=500, debug=False):
        features = s._format_named(features)
        labels = s._format_named(labels)
        if not s._model:
            s._model = model = Sequential([
                Dense(256, input_dim=len(features[0])),
                Dense(256),
                Dense(256, activation="relu"),
                Dense(256),
                Dense(len(labels[0]))])
            s._compile()
        print("Training. Please wait...")
        res = s._model.fit(
            features,
            labels,
            batch_size=100,
            shuffle=True,
            epochs=epochs,
            verbose=1 if debug else 0)
        # print(res.history["acc"])
        return s

    def evaluate(s, features, labels, debug=False):
        features = s._format_named(features)
        labels = s._format_named(labels)
        if not s._model:
            raise RuntimeError("Machine not yet trained.")
        return s._model.evaluate(features, labels, verbose=1 if debug else 0)

    def predict(s, features):
        features = s._format_named(features)
        if not s._model:
            raise RuntimeError("Machine not yet trained.")
        res = s._model.predict(features)
        # reconstruct named columns
        return [{c: float(b) for b, c in zip(a, s._metadata["cols"])} for a in res]
