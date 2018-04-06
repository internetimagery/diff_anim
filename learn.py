# Learn the pattern between the inputs
from __future__ import print_function
from keras import losses, callbacks
from keras.layers import Dense, Activation, Dropout
from keras.models import Sequential, load_model
import numpy as np
import os.path
import json
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2" # shut up tensorflow

class Brain(object):
    def __init__(s, path):
        s._path = path
        s._model = None
        s._metadata = {}
        s.state = "state.hdf5"
        s.best = "best.hdf5"
        s.meta = "metadata.json"

        if not os.path.isdir(path):
            raise RuntimeError("Path not found %s" % path)

        if os.listdir(path):
            s.load_state(path)

    def __getitem__(s, item):
        return s._metadata[item]

    def __setitem__(s, item, val):
        s._metadata[item] = val

    def _compile(s):
        s._model.compile(
            optimizer="RMSprop",
            # optimizer="adam",
            # optimizer="SGD",
            loss="mean_squared_error",
            # loss="mean_absolute_error",
            # loss="mean_squared_logarithmic_error",
            metrics=["acc"])

    def _format_stream(s, stream):
        """ Expect generator that produce dicts of data """
        if "cols" not in s._metadata: # Add column headers
            row1, row2 = stream.next()
            s._metadata["cols"] = list(set(row1) | set(row2))
        for bef, aft in stream:
            yield np.array([bef.get(a, 0.0) for a in s._metadata["cols"]]), np.array([aft.get(a, 0.0) for a in s._metadata["cols"]])

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
        s._path = path
        state_path = os.path.join(path, s.state)
        meta_path = os.path.join(path, s.meta)
        if not os.path.isfile(meta_path) or not os.path.isfile(state_path):
            raise OSError("Data cannot be found at %s" % path)
        with open(meta_path, "r") as f:
            s._metadata = json.load(f)

        best = os.path.join(s._path, s.best)
        # if os.path.isfile(best):
        #     s._model = load_model(best)
        # else:
        s._model = load_model(state_path)
        # s._compile()
        return s

    def save_state(s, path=""):
        path = path or s._path
        if not s._model:
            raise RuntimeError("Machine not yet trained.")
        if not os.path.isdir(path):
            raise RuntimeError("Path does not exist: %s" % path)
        s._path = path
        state_path = os.path.join(path, s.state)
        meta_path = os.path.join(path, s.meta)
        with open(meta_path, "w") as f:
            json.dump(s._metadata, f, indent=4)
        s._model.save(state_path)
        return s

    def train(s, stream, epochs=200, callback=(lambda x: x), debug=False):

        features, labels = zip(*s._format_stream(stream))
        features, labels = np.array(features), np.array(labels)

        if not s._model:
            num_features = len(features[0])
            s._model = Sequential([
                Dense(num_features**2, input_dim=num_features),
                Dense(num_features*2),
                Dense(num_features)])
            s._compile()
        print("Training. Please wait...")

        prog = 1.0 / epochs
        best_callback = callbacks.ModelCheckpoint(os.path.join(s._path, s.best), save_best_only=True, monitor="acc")
        # stop_callback = callbacks.EarlyStopping(monitor="loss")
        epoch_callback = callbacks.LambdaCallback(on_epoch_end=lambda x,_: callback(x * prog))

        res = s._model.fit(
            features,
            labels,
            shuffle=True,
            # validation_split=0.5,
            epochs=epochs,
            callbacks=[best_callback, epoch_callback],#, stop_callback],
            verbose=1 if debug else 0)

        s.save_state()
        # print(res.history["acc"])
        return s

    def evaluate(s, stream, debug=False):
        features, labels = zip(*s._format_stream(stream))
        features, labels = np.array(features), np.array(labels)
        if not s._model:
            raise RuntimeError("Machine not yet trained.")
        best = os.path.join(s._path, s.best)
        if os.path.isfile(best):
            return load_model(best).evaluate(features, labels, verbose=1 if debug else 0)
        return s._model.evaluate(features, labels, verbose=1 if debug else 0)

    def predict(s, features):
        features = s._format_named(features)
        if not s._model:
            raise RuntimeError("Machine not yet trained.")
        best = os.path.join(s._path, s.best)
        if os.path.isfile(best):
            res = load_model(best).predict(features)
        else:
            res = s._model.predict(features)
        # reconstruct named columns
        return [{c: float(b) for b, c in zip(a, s._metadata["cols"])} for a in res]
