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

def get_channelbox():
    return [c+"."+b for a in "msho" for b in cmds.channelBox("mainChannelBox", q=True, **{"s%sa"%a:True}) or [] for c in cmds.channelBox("mainChannelBox", q=True, **{"%sol"%a:True})]

def collect_anim(Fstart=None, Fend=None, Fstep=1, attrs=None):
    """ Pull out anim data """
    # Validate inputs
    Fstart = cmds.playbackOptions(q=True, min=True) if Fstart == None else Fstart
    Fend = cmds.playbackOptions(q=True, max=True) if Fend == None else Fend
    Fstart, Fend = min(Fstart, Fend), max(Fstart, Fend)
    Fend += Fstep

    Fstep = abs(Fstep)
    if not Fstep:
        raise RuntimeError("Step is %s" % Fstep)

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
    while frame <= Fend:
        # cmds.currentTime(frame)
        res[frame] = {a: cmds.getAttr(a, t=frame) for a in attrs}
        # res[frame] = {strip_namespace(a): cmds.getAttr(a, t=frame) for a in attrs}
        frame += Fstep
    return res

def drive_anim(data):
    for frame in data:
        for attr, val in data[frame].items():
            cmds.setKeyframe(attr, v=val, t=frame)

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

def load_stream(path):
    """ Split file into "frame", "{col: data}". Sorting out metadata (frames) """
    data = format.load_stream(path)
    header = data.next()
    frame_col = header.get("frame_col")
    if not frame_col:
        raise RuntimeError("Frame column missing...")
    for row in data:
        fr = row[frame_col]
        del row[frame_col]
        yield fr, row

def join_streams(before_path, after_path):
    """ EGON! Join two streams matching frames. Assuming frames in order """
    before_stream = load_stream(before_path)
    after_stream = load_stream(after_path)

    for bframe, bdata in before_stream:
        aframe, adata = after_stream.next()
        while aframe < bframe: # Fast forward after_stream stream to catch up
            aframe, adata = after_stream.next()
        while aframe > bframe: # Fast forward before_stream stream to catch up
            bframe, bdata = before_stream.next()
        yield bdata, adata

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
