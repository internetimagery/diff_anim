# GUI
from __future__ import print_function
import maya.cmds as cmds
import numpy as np
import maya_utils
import os.path
import learn

class PathBrowse(object):
    def __init__(s, label, filter="", fm=1):
        s.filter = filter
        s.fm = fm
        s._gui = cmds.textFieldButtonGrp(l=label, bl="Browse", bc=s.browse, adj=2)
    def get_text(s):
        return cmds.textFieldButtonGrp(s._gui, q=True, tx=True).strip()
    def browse(s):
        path = cmds.fileDialog2(fm=s.fm, ff=s.filter)
        if path:
            cmds.textFieldButtonGrp(s._gui, e=True, tx=path[0])

class Window(object):
    def __init__(s):
        cmds.window(t="Anim Train!")
        cmds.columnLayout(adj=True)
        tabs = cmds.tabLayout()

        # Apply
        apply_col = cmds.columnLayout(adj=True, p=tabs)
        s.apply_path = PathBrowse("Training Data (training):", False, fm=3)
        cmds.button(l="Apply to selected!", c=s.apply)

        # Export
        export_col = cmds.columnLayout(adj=True, p=tabs)
        s.export_path = PathBrowse("Export (anim):", "*.anim", fm=0)
        cmds.button(l="Export Animation", c=s.export)

        # Train
        train_col = cmds.columnLayout(adj=True, p=tabs)
        s.train_path_train = PathBrowse("Training Data (training):", False, fm=3)
        s.train_path_source = PathBrowse("Source (anim):", "*.anim", fm=1)
        s.train_path_expect = PathBrowse("Expected (anim):", "*.anim", fm=1)
        cmds.button(l="Train", c=s.train)

        cmds.tabLayout(tabs, e=True, tl=[
            (apply_col, "Apply"),
            (export_col, "Export"),
            (train_col, "Train")])
        cmds.showWindow()

    def export(s, *_):
        """ Export animation to file """
        path = s.export_path.get_text()
        if not os.path.isdir(os.path.dirname(path)):
            raise RuntimeError("Export path does not exist: %s" % path)
        print("Exporting animation to:", path)
        data = maya_utils.collect_anim()
        maya_utils.export_anim(path, data)

    def train(s, *_):
        """ Begin/continue training on new data """
        train_path = s.train_path_train.get_text()
        source_path = s.train_path_source.get_text()
        expect_path = s.train_path_expect.get_text()
        if not os.path.isdir(train_path) or not os.path.isfile(source_path) or not os.path.isfile(expect_path):
            raise RuntimeError("One or more paths are invalid.")
        print("Training. Please wait. This can take a while.")
        source_data = maya_utils.import_anim(source_path)
        expect_data = maya_utils.import_anim(expect_path)
        # Filter out only frames and attributes in common
        source_data, expect_data = maya_utils.filter_frames(source_data, expect_data)

        try:
            brain = learn.Brain().load_state(train_path)
            print("Loaded previous instance.")
        except OSError:
            print("Training new instance.")
            brain = learn.Brain()
            for key in source_data:
                brain["cols"] = source_data[key].keys() # record keys
                break
        # Format data into vectors
        frames = source_data.keys() # Maintain frame order and column order
        source_data = np.array([np.array([source_data[a][b] for b in brain["cols"]]) for a in frames])
        expect_data = np.array([np.array([expect_data[a][b] for b in brain["cols"]]) for a in frames])

        brain.train(source_data, expect_data).save_state(train_path)
        print("Training complete. Accuracy:", brain.evaluate(source_data, expect_data))

    def apply(s, *_):
        """ Apply training to animation """
        path = s.apply_path.get_text()
        if not os.path.isdir(path):
            raise RuntimeError("Path does not exist: %s" % path)
        print("Applying animation. \"This is what we've trained for!\"")

        brain = learn.Brain().load_state(path)

        data = maya_utils.collect_anim()
        frames = data.keys()
        format_dict = np.array([np.array([data[a][b] for b in brain["cols"]]) for a in frames])
        keys = brain.predict(format_dict)

        for frame, row in zip(frames, keys):
            for attr, val in zip(brain["cols"], row):
                cmds.setKeyframe(attr, v=float(val), t=frame)
