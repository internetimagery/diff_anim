# GUI
from __future__ import print_function
import maya.cmds as cmds
import os.path

class PathBrowse(object):
    def __init__(s, label, is_file=True):
        s.is_file = is_file
        s._gui = cmds.textFieldButtonGrp(l=label, bl="Browse", bc=s.browse, adj=2)
    def get_text(s):
        return cmds.textFieldButtonGrp(s._gui, q=True, tx=True).strip()
    def browse(s):
        path = cmds.fileDialog2(fm=0 if s.is_file else 3)
        if path:
            cmds.textFieldButtonGrp(s._gui, e=True, tx=path[0])

class Window(object):
    def __init__(s):
        cmds.window(t="Anim Train!")
        cmds.columnLayout(adj=True)
        tabs = cmds.tabLayout()

        # Apply
        apply_col = cmds.columnLayout(adj=True, p=tabs)
        s.apply_path = PathBrowse("Training Data (training):", False)
        cmds.button(l="Apply to selected!", c=s.apply)

        # Export
        export_col = cmds.columnLayout(adj=True, p=tabs)
        s.export_path = PathBrowse("Export (anim):")
        cmds.button(l="Export Animation", c=s.export)

        # Train
        train_col = cmds.columnLayout(adj=True, p=tabs)
        s.train_path_train = PathBrowse("Training Data (training):", False)
        s.train_path_source = PathBrowse("Source (anim):")
        s.train_path_expect = PathBrowse("Expected (anim):")
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
        print("Export path", path)

    def train(s, *_):
        """ Begin/continue training on new data """
        train_path = s.train_path_train.get_text()
        source_path = s.train_path_source.get_text()
        expect_path = s.train_path_expect.get_text()
        if not os.path.isdir(train_path) or not os.path.isfile(source_path) or not os.path.isfile(expect_path):
            raise RuntimeError("One or more paths are invalid.")
        print("training!")

    def apply(s, *_):
        """ Apply training to animation """
        path = s.apply_path.get_text()
        if not os.path.isdir(path):
            raise RuntimeError("Path does not exist: %s" % path)
        print("train path!", path)
