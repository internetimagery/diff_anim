# Utilities from within maya
from __future__ import print_function, division
import maya.cmds as cmds
import collections
import format
import time

NUM_TYPES = set(["double", "doubleLinear", "doubleAngle"])

def strip_namespace(name):
    return name.rsplit(":", 1)[-1]

def add_namespace(namespace, name):
    if namespace:
        return namespace + ":" + name
    return name

def collect_anim(Fstart=None, Fend=None, step=1, attrs=None):
    """ Pull out anim data """
    # Validate inputs
    Fstart = cmds.playbackOptions(q=True, min=True) if Fstart == None else Fstart
    Fend = cmds.playbackOptions(q=True, max=True) if Fend == None else Fend
    Fstart, Fend = min(Fstart, Fend), max(Fstart, Fend)
    Fend += step

    step = abs(step)
    if not step:
        raise RuntimeError("Step is %s" % step)

    if attrs is None:
        selection = cmds.ls(sl=True)
        if not selection:
            raise RuntimeError("No attributes provided and nothing selected.")
        attrs = [a+"."+b for a in selection for b in cmds.listAttr(a, k=True) or []]

    # Filter attrs
    attrs = [a for a in attrs if cmds.getAttr(a, type=True) in NUM_TYPES]

    diff = Fend - Fstart
    frame = Fstart
    res = collections.OrderedDict()
    while frame < Fend:
        cmds.currentTime(frame)
        res[frame] = {a: cmds.getAttr(a) for a in attrs}
        frame += step
    return res

def export_anim(path, data, frame_col="[FRAME]"):
    """ Export animation file """

    header = {
        "frame_col": frame_col,
        "scene": cmds.file(q=True, sn=True)
    }
    res = []
    for frame, val in data.items():
        val[frame_col] = frame
        res.append(val)
    format.save(path, header, res)

def import_anim(path, namespace=""):
    """ Pull back animation """
    header, data = format.load(path)
    frame_col = header.get("frame_col")
    if not frame_col:
        raise RuntimeError("Frame name missing...")

    res = {}
    for row in data:
        fr = row[frame_col]
        del row[frame_col]
        res[fr] = {add_namespace(namespace, a): float(row[a]) for a in row}
    return res

def filter_frames(data1, data2):
    """ Filter out only data that has frames and attributes in common """
    res1 = {}
    res2 = {}
    cols = None
    for frame in data1:
        try:
            if not cols:
                cols = set(data1[frame].keys()) & set(data2[frame].keys())
            res1[frame] = {a: data1[frame][a] for a in cols}
            res2[frame] = {a: data2[frame][a] for a in cols}
        except KeyError:
            pass
    return res1, res2
