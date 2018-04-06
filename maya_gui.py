# GUI
from __future__ import print_function
import maya.cmds as cmds
import maya.mel as mel
import maya_utils
import itertools
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

class FrameRange(object):
    def __init__(s):
        s._gui = cmds.floatFieldGrp(
            nf=3, pre=2, l="Frames (start / end / step)",
            v1=cmds.playbackOptions(q=True, min=True),
            v2=cmds.playbackOptions(q=True, max=True),
            v3=1.0)
    def get_values(s):
        return cmds.floatFieldGrp(s._gui, q=True, v=True)


class Window(object):
    def __init__(s):
        cmds.window(t="Anim Train!")
        cmds.columnLayout(adj=True)
        tabs = cmds.tabLayout()

        # Apply
        apply_col = cmds.columnLayout(adj=True, p=tabs)
        cmds.text(l="Apply training to current animation, on selected objects.")
        s._apply_path = PathBrowse("Training Data (training):", False, fm=3)
        s._apply_range = FrameRange()
        cmds.button(l="Apply to selected!", c=s.apply)

        # Export
        export_col = cmds.columnLayout(adj=True, p=tabs)
        cmds.text(l="Export animation from selected objects to be used for later training.")
        s._export_path = PathBrowse("Export (anim):", "*.anim", fm=0)
        s._export_range = FrameRange()
        cmds.button(l="Export Animation", c=s.export)

        # Train
        train_col = cmds.columnLayout(adj=True, p=tabs)
        cmds.text(l="Learn differences between animation data.")
        s._train_path_train = PathBrowse("Training Data (training):", False, fm=3)
        s._train_path_source = PathBrowse("Source (anim):", "*.anim", fm=1)
        s._train_path_expect = PathBrowse("Expected (anim):", "*.anim", fm=1)
        cmds.button(l="Train", c=s.train)

        # Test
        test_col = cmds.columnLayout(adj=True, p=tabs)
        cmds.text(l="Check accuracy.")
        s._test_path_train = PathBrowse("Training Data (training):", False, fm=3)
        s._test_path_source = PathBrowse("Source (anim):", "*.anim", fm=1)
        s._test_path_expect = PathBrowse("Expected (anim):", "*.anim", fm=1)
        cmds.button(l="Test", c=s.test)


        cmds.tabLayout(tabs, e=True, tl=[
            (apply_col, "Apply"),
            (export_col, "Export"),
            (train_col, "Train"),
            (test_col, "Test")])
        cmds.showWindow()

    def export(s, *_):
        """ Export animation to file """
        path = s._export_path.get_text()
        Fstart, Fend, Fstep = s._export_range.get_values()
        if not os.path.isdir(os.path.dirname(path)):
            raise RuntimeError("Export path does not exist: %s" % path)
        print("Exporting animation to:", path)
        cb = maya_utils.get_channelbox()
        data = maya_utils.collect_anim(Fstart=Fstart, Fend=Fend, Fstep=Fstep, attrs=cb)
        maya_utils.export_anim(path, data)

    def train(s, *_):
        """ Begin/continue training on new data """
        train_path = s._train_path_train.get_text()
        source_path = s._train_path_source.get_text()
        expect_path = s._train_path_expect.get_text()
        if not os.path.isdir(train_path) or not os.path.isfile(source_path) or not os.path.isfile(expect_path):
            raise RuntimeError("One or more paths are invalid.")
        print("Training. Please wait. This can take a while. Press Esc to abort.")

        data_stream = itertools.chain(maya_utils.join_streams(source_path, expect_path))

        brain = learn.Brain(train_path)

        progctrl = mel.eval("$tmp = $gMainProgressBar")
        cmds.progressBar(progctrl, e=True, bp=True, st="Thinking...", max=100, ii=True)
        def update(prog):
            if cmds.progressBar(progctrl, q=True, ic=True):
                raise StopIteration
            prog *= 100
            if not prog % 1:
                cmds.progressBar(progctrl, e=True, progress=prog)

        cmds.refresh(suspend=True)
        try:
            d1, d2 = itertools.tee(data_stream, 2)
            brain.train(d1, epochs=500, callback=update)
            acc = brain.evaluate(d2)[1]
            print("Training complete. Estimated accuracy:", acc)
        except StopIteration:
            print("Training cancelled!")
        finally:
            cmds.refresh(suspend=False)
            cmds.progressBar(progctrl, e=True, ep=True)

    def apply(s, *_):
        """ Apply training to animation """
        path = s._apply_path.get_text()
        Fstart, Fend, Fstep = s._export_range.get_values()
        if not os.path.isdir(path):
            raise RuntimeError("Path does not exist: %s" % path)
        print("Applying animation. \"This is what we've trained for!\"")

        brain = learn.Brain(path)
        sel = maya_utils.get_channelbox()

        data = maya_utils.collect_anim(Fstart=Fstart, Fend=Fend, Fstep=Fstep)
        frames = data.keys()
        format_dict = [data[a] for a in frames]
        keys = brain.predict(format_dict)

        maya_utils.drive_anim({a: {maya_utils.add_namespace("", c): b[c] for c in b if c in sel} for a, b in zip(frames, keys)})

    def test(s, *_):
        """ Check accuracy """
        train_path = s._test_path_train.get_text()
        source_path = s._test_path_source.get_text()
        expect_path = s._test_path_expect.get_text()
        if not os.path.isdir(train_path) or not os.path.isfile(source_path) or not os.path.isfile(expect_path):
            raise RuntimeError("One or more paths are invalid.")
        print("Checking. Please wait.")

        data_stream = itertools.chain(maya_utils.join_streams(source_path, expect_path))

        brain = learn.Brain(train_path)
        accuracy = brain.evaluate(data_stream)[1]
        cmds.confirmDialog(t="Accuracy", m="Predicted accuracy: %s%%" % round(accuracy*100))
