# Anim exports

from __future__ import print_function
import datetime
import os.path
import getpass
import shutil
import json
import csv
import os

def read_header(reader):
    buff = ""
    header = None
    for row in reader:
        buff += row
        try:
            header = json.loads(buff)
            break
        except (AttributeError, ValueError):
            pass
    return header

def write_header(reader, header):
    # Add general metadata
    header["user"] = getpass.getuser()
    header["created"] = str(datetime.datetime.now())
    reader.write(json.dumps(header, indent=4)+"\n")


def save(path, header=None, data=None):
    """ Export data file from maya
        path = "path/to/file"
        header = {header:elements}
        data = [{row: value}, ...]
    """
    data = data or []
    tmp = path + ".tmp"

    # Write new file
    with open(tmp, "wb") as f:
        write_header(f, header or {})
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
        header = read_header(f)
        data = [a for a in csv.DictReader(f)]
    return header, data

def load_stream(path):
    """ Open file. Return header and then data """
    with open(path, "rb") as f:
        yield read_header(f)
        for row in csv.DictReader(f):
            yield row

def merge(output, *paths):
    """ Merge multiple files into one """
    with open(output, "wb") as o:
        writer = None
        framecol = None
        for i, path in enumerate(paths):
            with open(path, "rb") as r:
                header = read_header(r)
                for row in csv.DictReader(r):
                    if not writer:
                        framecol = header["frame_col"]
                        write_header(o, {"merged_from": paths, "frame_col": framecol})
                        writer = csv.DictWriter(o, fieldnames=row.keys())
                        writer.writeheader()
                    tmp_frm = row[header["frame_col"]]
                    del row[header["frame_col"]]
                    row[framecol] = "%s:%s" % (i, tmp_frm)
                    writer.writerow(row)
