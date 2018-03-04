# Anim exports

from __future__ import print_function
import os.path
import shutil
import json
import csv
import os

def save(path, header=None, data=None):
    """ Export data file from maya
        path = "path/to/file"
        header = {header:elements}
        data = [{row: value}, ...]
    """
    header = header or {}
    data = data or []
    tmp = path + ".tmp"
    # Write new file
    with open(tmp, "wb") as f:
        f.write(json.dumps(header, indent=4)+"\n")
        writer = None
        for row in data:
            if not writer:
                writer = csv.DictWriter(f, fieldnames=row.keys())
                writer.writeheader()
            writer.writerow(row)
    # File written
    try:
        os.remove(path)
    except OSError:
        pass
    shutil.move(tmp, path)

def load(path):
    """ Load file from path. Return {} header and [{}] data """
    with open(path, "rb") as f:
        buff = ""
        header = None
        data = []
        for row in f:
            buff += row
            try:
                header = json.loads(buff)
                break
            except (AttributeError, ValueError):
                pass
        data = [a for a in csv.DictReader(f)]
    return header, data
