# Utilities from within maya
from __future__ import print_function, division
import maya.cmds as cmds
import format
import time

def strip_namespace(name):
    return name.rsplit(":", 1)[-1]

def add_namespace(namespace, name):
    if namespace:
        return namespace + ":" + name
    return name

def export_anim(path, Fstart=None, Fend=None, step=1, attrs=None, frame_name="[FRAME]"):
    """ Export animation file """

    header = {
        "frame_name": frame_name,
        "created": time.time()
    }
    data = []

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
    attrs = [a for a in attrs if cmds.getAttr(a, type=True) == "double"]

    diff = Fend - Fstart
    frame = Fstart
    while frame < Fend:
        cmds.currentTime(frame)
        d = {strip_namespace(a): cmds.getAttr(a) for a in attrs}
        d[frame_name] = frame
        data.append(d)
        frame += step
    format.save(path, header, data)

def import_anim(path, namespace=""):
    """ Pull back animation """
    header, data = format.load(path)
    frame_name = header.get("frame_name")
    if not frame_name:
        raise RuntimeError("Frame name missing...")

    res = {}
    for row in data:
        fr = row[frame_name]
        del row[frame_name]
        res[fr] = {add_namespace(namespace, a): float(row[a]) for a in row}
    return res
