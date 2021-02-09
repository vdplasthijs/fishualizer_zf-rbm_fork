import sys, os, argparse
# print(os.getcwd(), __file__)
fishualizer_folder = __file__.rstrip('Fishualizer.py')
sys.path.append(os.path.join(fishualizer_folder, 'Source/Tools'))
import logging
from logging.handlers import RotatingFileHandler
import numpy as np
from PyQt5 import QtCore, QtWidgets, QtGui, Qt
import pyqtgraph.opengl as gl
import pyqtgraph as pg
import OpenGL.GL, OpenGL.GL.EXT.blend_minmax
from pyqtgraph.dockarea import DockArea, Dock
import matplotlib.pyplot as plt
from Controls import ColorRangePicker, ShortcutsDialog, StaticDataDialog, SelectCellDialog,\
    GLViewWidgetPos, LogWindow, CorrelationDialog, ColoringDialog, OpenDataDialog, ExtendedComboBox,\
    WarningDialog, ColorMapDialog
from zecording import ShelveWindow, Zecording
import scipy
import scipy.sparse
from scipy.cluster.vq import kmeans, vq
from scipy.spatial import Delaunay, ConvexHull
import pandas as pd
import psutil as ps
import time, datetime, random, pickle
from functools import partial
from internal_ipkernel import InternalIPKernel
from imageio import get_writer, imwrite
import utilities
import colors
import alpha_shape
from utilities import load_config
from Analysis import correlation_1D2D
sys.path.append(os.path.join(fishualizer_folder, 'Source/OASIS-master'))
try:  # load Friedrich et alii deconvolution package
    """ For OASIS packages download & demo, see
    https://github.com/j-friedrich/OASIS
    """
    from oasis import oasisAR1
    import oasis.functions
except ModuleNotFoundError:
    curr_dir = os.getcwd()
    try:
        import subprocess
        os.chdir('OASIS-master/')
        print('Installing OASIS')
        subprocess.check_call(['python', 'setup.py', 'build_ext', '--inplace'])  # try to install automatically
        print('Installation done')
    except:
        print("OASIS could not be installed automatically, \n if you would like to"
              "use online deconvolution, please see their github page github.com/j-friedrich/OASIS")
    os.chdir(curr_dir)
    try:
        from oasis import oasisAR1
        import oasis.functions
    except:
        print("OASIS module not loaded, spikes will not be deconvolved online. \n"
              "Make sure OASIS is installed correctly "
              "(see their brief Github installation guide: github.com/j-friedrich/OASIS )")


# TODO: Add to compute Menu: CV, Cluster?
# TODO: Generalize center_view to density plot
# TODO: Change the loading to be usually read only
# TODO: Add roation of density
# for iA in range(10):
#     self.density_plot.rotate(35,0,1,0)
#     self.view3D.update()
#     self.view2D.update()
#     pg.QtGui.QApplication.processEvents()
# TODO: Show numeric values along color scale bar
# TODO: make all graphs as Docks (like the neuronal activity and the behavior)
# TODO: take colormaps out of the zip file
# TODO: logarithmic color scale
# TODO: define Ctrl-Mouse Drag to drag the view for trackpads?
# TODO: Add 'reset view' shortcut (reset zoom, pan, selection, colors, transparency)

# NOTES on Color:
# start_color, stop_color : define the points where the variable color-range starts and stops
# disp_start, disp_stop : define the data values, which are mapped to this color range
# when clicking on the colorbar, only start_color and stop_color are changed
# when the edits are changed, then disp_start and disp_stop are changed


# noinspection PyUnresolvedReferences
class Viewer(QtWidgets.QMainWindow):
    """
    Base class that is not directly used. Its code is defining the basic elements of the GUI
    without having any real logic behind it. It could be automatically created from a GUI creation tool like QTDesigner.
    The idea is to separate form (the GUI elements) and function (what do they do)
    """

    def __init__(self):
        # create and initialize window
        super(Viewer, self).__init__()  # Calling the parent class __init__ method
        self.setAttribute(QtCore.Qt.WA_DeleteOnClose)

        ## Menu bars
        self.file_menu = QtWidgets.QMenu('&File', self)
        self.edit_menu = QtWidgets.QMenu('&Edit', self)
        self.view_menu = QtWidgets.QMenu('&View', self)
        self.compute_menu = QtWidgets.QMenu('&Compute', self)
        self.extra_menu = QtWidgets.QMenu('E&xtra', self)
        self.help_menu = QtWidgets.QMenu('&Help', self)

        ## FILE MENU
        self.file_menu.addAction('&Open data', self.select_data_file, QtCore.Qt.CTRL + QtCore.Qt.Key_O)
        self.file_menu.addAction('&Add static data', self.add_static_gui, QtCore.Qt.CTRL + QtCore.Qt.Key_A)
        self.file_menu.addAction('Add density data', self.load_density)
        self.file_menu.addAction('&Quit', self.close, QtCore.Qt.CTRL + QtCore.Qt.Key_Q)

        ## EDIT MENU
        # Change selected neurons
        self.edit_menu.addAction('Select neuron by id', self.show_select_cell_dialog, QtCore.Qt.CTRL + QtCore.Qt.Key_G)
        self.hide_action_min = self.edit_menu.addAction('&Hide points below threshold', partial(self.edit_hide,threshold='min'))
        self.hide_action_max = self.edit_menu.addAction('&Hide points above threshold', partial(self.edit_hide, threshold='max'))
        self.reset_view_action = self.edit_menu.addAction('&View all neurons', self.reset_view)
        self.hide_action_min.setEnabled(False)
        self.hide_action_max.setEnabled(False)
        self.reset_view_action.setEnabled(False)
        limit_group = QtWidgets.QActionGroup(self)
        self.limit_menu = QtWidgets.QMenu('Limit axis extent')
        self.x_action = QtWidgets.QAction('X-axis', self, checkable=True)
        self.x_action.setShortcut( QtCore.Qt.CTRL + QtCore.Qt.Key_1)
        self.x_action.triggered.connect(self.limitx)
        self.x_action = limit_group.addAction(self.x_action)
        self.limit_menu.addAction(self.x_action)
        self.y_action = QtWidgets.QAction('Y-axis', self, checkable=True)
        self.y_action.setShortcut( QtCore.Qt.CTRL + QtCore.Qt.Key_2)
        self.y_action.triggered.connect(self.limity)
        self.y_action = limit_group.addAction(self.y_action)
        self.limit_menu.addAction(self.y_action)
        self.z_action = QtWidgets.QAction('Z-axis', self, checkable=True)
        self.z_action.triggered.connect(self.limitz)
        self.z_action.setShortcut(QtCore.Qt.CTRL + QtCore.Qt.Key_3)
        self.z_action = limit_group.addAction(self.z_action)
        self.limit_menu.addAction(self.z_action)
        self.edit_menu.addMenu(self.limit_menu)
        self.edit_menu.addSeparator()

        # Density
        self.toggle_density_action = QtWidgets.QAction('Toggle density plot', self, checkable=True)
        self.toggle_density_action.setChecked(False)
        self.toggle_density_action.setEnabled(False)
        self.toggle_density_action.triggered.connect(self.toggle_density_plot)
        self.edit_menu.addAction(self.toggle_density_action)
        self.edit_menu.addSeparator()

        # Change coordinates
        self.choose_coordinates_action = self.edit_menu.addAction('&Choose coordinate system', self.choose_coordinates)
        self.choose_coordinates_action.setEnabled(False)
        self.copy_clipboard_action = QtWidgets.QAction('&Selected coords to clipboard', self, checkable=True)
        self.copy_clipboard_action.triggered.connect(self.copy_clipboard)
        self.copy_clipboard_action.setChecked(True)
        self.edit_menu.addAction(self.copy_clipboard_action)
        self.edit_menu.addSeparator()

        # Change axes
        self.edit_menu.addAction('Inverse x axis', self.inverse_x_axis)
        self.edit_menu.addAction('Inverse y axis', self.inverse_y_axis)
        self.edit_menu.addAction('Inverse z axis', self.inverse_z_axis)
        self.edit_menu.addAction('Reverse x coordinates', self.call_reverse_x_coords)
        self.edit_menu.addAction('Reverse y coordinates', self.call_reverse_y_coords)
        self.edit_menu.addAction('Reverse z coordinates', self.call_reverse_z_coords)
        self.edit_menu.addAction('Switch x / y axes', self.call_switch_xy_coords)

        ## VIEW MENU
        # Change appearance of both 2D & 3D plots
        self.view_menu.addAction('&Color map', self.edit_cmap)
        self.alpha_menu = QtWidgets.QMenu('Transparency mode', self)
        max_callback = partial(self.switch_alpha_mode, 'maximum')
        add_callback = partial(self.switch_alpha_mode, 'additive')
        trans_callback = partial(self.switch_alpha_mode, 'translucent')
        op_callback = partial(self.switch_alpha_mode, 'opaque')
        alpha_group = QtWidgets.QActionGroup(self)
        self.max_action = QtWidgets.QAction('Maximum', self, checkable=True)
        self.max_action.triggered.connect(max_callback)
        self.max_action = alpha_group.addAction(self.max_action)
        self.max_action.setChecked(True)
        self.alpha_menu.addAction(self.max_action)
        self.a_action = QtWidgets.QAction('Additive', self, checkable=True)
        self.a_action.triggered.connect(add_callback)
        self.a_action = alpha_group.addAction(self.a_action)
        self.alpha_menu.addAction(self.a_action)
        self.t_action = QtWidgets.QAction('Translucent', self, checkable=True)
        self.t_action.triggered.connect(trans_callback)
        self.t_action = alpha_group.addAction(self.t_action)
        self.alpha_menu.addAction(self.t_action)
        self.o_action = QtWidgets.QAction('Opaque', self, checkable=True)
        self.o_action.triggered.connect(op_callback)
        self.o_action = alpha_group.addAction(self.o_action)
        self.alpha_menu.addAction(self.o_action)
        self.alpha_menu.setEnabled(False)
        self.view_menu.addMenu(self.alpha_menu)
        self.nan_action = QtWidgets.QAction('Toggle NaN visibility', self, checkable=True)
        self.nan_action.setShortcut(QtCore.Qt.CTRL + QtCore.Qt.SHIFT + QtCore.Qt.Key_N,)
        self.nan_action.setChecked(True)
        self.nan_action.triggered.connect(self.toggle_nanview)
        self.view_menu.addAction(self.nan_action)
        self.randomz_action = QtWidgets.QAction('Toggle Jitter Z-coordinates', self, checkable=True)
        self.randomz_action.setShortcut(QtCore.Qt.CTRL + QtCore.Qt.SHIFT + QtCore.Qt.Key_Z,)
        self.randomz_action.setChecked(False)
        self.randomz_action.triggered.connect(self.toggle_randomzfill)
        self.view_menu.addAction(self.randomz_action)
        self.color_cluster_action = self.view_menu.addAction('Color &clusters', self.cluster_color)
        self.color_region_action = self.view_menu.addAction('Color &Regions', self.region_color)
        self.color_region_action.setEnabled(False)
        self.color_cluster_action.setEnabled(False)
        self.change_region_view_action = self.view_menu.addAction('Toggle region view (local/ZBrainAtlas)', self.change_region_view)
        self.view_menu.addAction('Decrease Dot Size', self.decrease_dot_size, QtCore.Qt.CTRL +  QtCore.Qt.SHIFT + QtCore.Qt.Key_Minus)
        self.view_menu.addAction('Increase Dot Size', self.increase_dot_size, QtCore.Qt.CTRL +  QtCore.Qt.SHIFT + QtCore.Qt.Key_Equal)
        self.grid_action = QtWidgets.QAction('Toggle Grid', self, checkable=True)
        self.grid_action.setChecked(True)
        self.grid_action.setShortcut(QtCore.Qt.CTRL + QtCore.Qt.SHIFT + QtCore.Qt.Key_G)
        self.grid_action.triggered.connect(self.toggle_hidegrid)
        self.view_menu.addAction(self.grid_action)
        self.somas_action = QtWidgets.QAction('Toggle Somata', self, checkable=True)
        self.somas_action.setChecked(True)
        self.somas_action.setShortcut(QtCore.Qt.CTRL + QtCore.Qt.SHIFT + QtCore.Qt.Key_P)
        self.somas_action.triggered.connect(self.toggle_somas)
        self.view_menu.addAction(self.somas_action)
        self.view_menu.addSeparator()

        # Change 3D plot
        self.view_menu.addAction('Pan left', self.pan_left, QtCore.Qt.CTRL + QtCore.Qt.Key_Left)
        self.view_menu.addAction('Pan right', self.pan_right, QtCore.Qt.CTRL + QtCore.Qt.Key_Right)
        self.view_menu.addAction('Pan front', self.pan_front, QtCore.Qt.CTRL + QtCore.Qt.Key_Up)
        self.view_menu.addAction('Pan back', self.pan_back, QtCore.Qt.CTRL + QtCore.Qt.Key_Down)
        self.view_menu.addSeparator()

        # Activity or HU view mode
        self.view_menu.addAction('Activity view mode', self.activity_view_mode)
        self.view_menu.addAction('HU view mode', self.hu_view_mode)
        self.view_menu.addSeparator()

        # Change 3D or 2D plot
        self.perspective_menu = QtWidgets.QMenu('Change Perspective', self)
        perspective_group = QtWidgets.QActionGroup(self)
        self.perspective_2d_action = QtWidgets.QAction('2D Perspective', self, checkable=True)
        self.perspective_2d_action.triggered.connect(lambda: self.set_perspective_2d(view=None))
        self.perspective_2d_action.setChecked(False)
        self.perspective_2d_action = perspective_group.addAction(self.perspective_2d_action)
        self.perspective_menu.addAction(self.perspective_2d_action)
        self.perspective_3d_action = QtWidgets.QAction('3D Perspective', self, checkable=True)
        self.perspective_3d_action.triggered.connect(lambda: self.set_perspective_3d(view=None))
        self.perspective_3d_action.setChecked(True)
        self.perspective_3d_action = perspective_group.addAction(self.perspective_3d_action)
        self.perspective_menu.addAction(self.perspective_3d_action)
        self.view_menu.addMenu(self.perspective_menu)
        self.view_menu.addAction('Center view', self.center_view, QtCore.Qt.CTRL + QtCore.Qt.Key_K)
        self.view_menu.addSeparator()

        # Animation
        self.view_menu.addAction('Change Playing Speed', self.change_playtimer_interval)
        self.anim_action = QtWidgets.QAction('Toggle animation', self, checkable=True)
        self.anim_action.setShortcut(QtCore.Qt.CTRL + QtCore.Qt.Key_P)
        self.anim_action.triggered.connect(self.toggle_animation)
        self.view_menu.addAction(self.anim_action)
        dyn_action = QtWidgets.QAction('Toggle dynamics during animation', self, checkable=True)
        dyn_action.setChecked(True)
        dyn_action.triggered.connect(self.toggle_dynamics)
        self.view_menu.addAction(dyn_action)
        rot_action = QtWidgets.QAction('Toggle rotation during animation', self, checkable=True)
        rot_action.setShortcut(QtCore.Qt.CTRL + QtCore.Qt.Key_R)
        rot_action.triggered.connect(self.switch_rotation)
        self.view_menu.addAction(rot_action)
        self.view_menu.addSeparator()

        # What data to show
        self.view_menu.addAction('Toggle current/previous dataset', self.selection_toggle,
                                 QtCore.Qt.CTRL + QtCore.Qt.Key_T)
        fit_action = QtWidgets.QAction('Toggle calcium fit plot', self, checkable=True)
        fit_action.setShortcut(QtCore.Qt.CTRL + QtCore.Qt.Key_Y)
        fit_action.triggered.connect(self.toggle_calcium_fit)
        self.view_menu.addAction(fit_action)
        neuropil_action = QtWidgets.QAction('Toggle neuropil plot', self, checkable=True)
        neuropil_action.setShortcut(QtCore.Qt.CTRL + QtCore.Qt.Key_N)
        neuropil_action.triggered.connect(self.toggle_hideneuropil)
        self.view_menu.addAction(neuropil_action)
        self.view_menu.addSeparator()

        # Projections
        self.view_menu.addAction('Switch to Top-Side View',self.switch_topside,QtCore.Qt.CTRL + QtCore.Qt.SHIFT + QtCore.Qt.Key_T)
        self.view_menu.addAction('View y-z projection', self.viewyz, QtCore.Qt.CTRL + QtCore.Qt.ALT + QtCore.Qt.Key_1)
        self.view_menu.addAction('View x-z projection', self.viewxz, QtCore.Qt.CTRL + QtCore.Qt.ALT + QtCore.Qt.Key_2)
        self.view_menu.addAction('View x-y projection', self.viewxy, QtCore.Qt.CTRL + QtCore.Qt.ALT + QtCore.Qt.Key_3)
        self.view_menu.addSeparator()

        # Outlines
        self.outline_menu = QtWidgets.QMenu('Draw 2D Outline')
        self.xy_outline_action = QtWidgets.QAction('XY Outline', self, checkable=True)
        self.xy_outline_action.triggered.connect(self.toggle_xy_outline)
        self.xy_outline_action.setChecked(False)
        self.outline_menu.addAction(self.xy_outline_action)
        self.yz_outline_action = QtWidgets.QAction('YZ Outline', self, checkable=True)
        self.yz_outline_action.triggered.connect(self.toggle_yz_outline)
        self.yz_outline_action.setChecked(False)
        self.outline_menu.addAction(self.yz_outline_action)
        self.xz_outline_action = QtWidgets.QAction('XZ Outline', self, checkable=True)
        self.xz_outline_action.triggered.connect(self.toggle_xz_outline)
        self.xz_outline_action.setChecked(False)
        self.outline_menu.addAction(self.xz_outline_action)
        self.view_menu.addMenu(self.outline_menu)
        self.view_menu.addAction('Show top/side view outlines', self.show_top_side_outlines)
        self.view_menu.addSeparator()

        ## COMPUTE MENU
        self.compute_menu.addAction('&Mean Activity', self.compute_mean_activity )
        self.compute_menu.addAction('&Correlation', self.compute_correlation_gui )

        ## EXTRA MENU
        self.extra_menu.addAction('&Log', self.help_log, QtCore.Qt.CTRL + QtCore.Qt.ALT + QtCore.Qt.Key_W)
        self.extra_menu.addAction('&Console', self.console, QtCore.Qt.CTRL + QtCore.Qt.ALT + QtCore.Qt.Key_C)
        self.extra_menu.addSeparator()

        # Capture screenshots
        self.video_action = QtWidgets.QAction('Start/Stop video export', self, checkable=True)
        self.video_action.setShortcut(QtCore.Qt.CTRL + QtCore.Qt.ALT + QtCore.Qt.Key_V)
        self.video_action.triggered.connect(self.video_export)
        self.extra_menu.addAction(self.video_action)
        self.screenshot_action = QtWidgets.QAction('Take a screenshot', self, checkable=False)
        self.screenshot_action.setShortcut(QtCore.Qt.CTRL + QtCore.Qt.SHIFT + QtCore.Qt.Key_S)
        self.screenshot_action.triggered.connect(self.grab_plot)
        self.extra_menu.addAction(self.screenshot_action)

        ## HELP MENU
        self.help_menu.addAction('&Shortcuts', self.help_shortcuts)

        self.menuBar().addMenu(self.file_menu)
        self.menuBar().addMenu(self.edit_menu)
        self.menuBar().addMenu(self.view_menu)
        self.menuBar().addMenu(self.compute_menu)
        self.menuBar().addMenu(self.extra_menu)
        self.menuBar().addMenu(self.help_menu)

        # MAIN LAYOUT
        window_size = [1500, 1200]
        self.v_layout_main = QtWidgets.QVBoxLayout()  # complete application

        # create a main widget
        self.main_widget = QtWidgets.QWidget(self)
        self.main_widget.setLayout(self.v_layout_main)
        self.main_widget.setGeometry(100, 500, window_size[0], window_size[1])
        self.main_widget.setWindowTitle('Fishualizer - no Data')
        self.main_widget.show()

        # LAYOUTS
        self.v_layout_left = QtWidgets.QVBoxLayout()
        self.v_layout_center = QtWidgets.QVBoxLayout()
        self.v_layout_center_right = QtWidgets.QVBoxLayout()
        self.v_layout_right = QtWidgets.QVBoxLayout()

        self.h_layout_top = QtWidgets.QHBoxLayout()
        self.h_layout_top.addLayout(self.v_layout_left)
        self.h_layout_top.addLayout(self.v_layout_center)
        self.h_layout_top.addLayout(self.v_layout_center_right)
        self.h_layout_top.addLayout(self.v_layout_right)

        self.v_layout_main.addLayout(self.h_layout_top)
        self.v_splitter = QtWidgets.QSplitter(QtCore.Qt.Vertical)

        # Set background color of window
        self.main_widget.setAutoFillBackground(True)
        palette = self.main_widget.palette()
        # palette.setColor(self.MWidget.backgroundRole(),Qt.qRgb(0.2,0.2,0.2))
        palette.setColor(self.main_widget.backgroundRole(), Qt.QColor(Qt.qRgb(100, 100, 100)))
        self.main_widget.setPalette(palette)

        #  Try to add certain keyboard shortcuts,
        # e.g. for toggling between the average activity view and the current data
        layerlimit_shortcut = QtWidgets.QShortcut('Ctrl+-', self.main_widget)
        layerlimit_shortcut.activated.connect(self.layers_scroll_top)
        selectcell_shortcut = QtWidgets.QShortcut('Ctrl+S', self.main_widget)
        selectcell_shortcut.activated.connect(self.select_cell)
        exportselected_shortcut = QtWidgets.QShortcut('Ctrl+E', self.main_widget)
        exportselected_shortcut.activated.connect(self.export_selected)
        fwd_shortcut = QtWidgets.QShortcut('Ctrl+F', self.main_widget)
        fwd_shortcut.activated.connect(self.step_fwd)
        bwd_shortcut = QtWidgets.QShortcut('Ctrl+B', self.main_widget)
        bwd_shortcut.activated.connect(self.step_bwd)
        showregioncurrentcell_shortcut = QtWidgets.QShortcut('Ctrl+Alt+G', self.main_widget)
        showregioncurrentcell_shortcut.activated.connect(self.show_region_current_cell)
        swapcoloraxis_shortcut = QtWidgets.QShortcut('Ctrl+C', self.main_widget)
        swapcoloraxis_shortcut.activated.connect(self.swap_color_axis)

        # Create Center Viewer
        self.last_viewbox = [None]  # list, where 1st object will hold the last clicked viewbox (make list so it is mutable when passed to the GLViewWidgetPos classes)
        self.view3D = GLViewWidgetPos(self, name='view3D', last_vb=self.last_viewbox)
        self.view3D.setMinimumHeight(200)

        # Create 2D Viewer(s)
        self.view2D = GLViewWidgetPos(self, name='view2D', last_vb=self.last_viewbox)
        self.view2D.setMinimumHeight(0)

        self.last_viewbox[0] = self.view3D  # initialise on 3D viewbox

        # Add 3 grids to the plot
        self.Grids = [None] * 3
        for iG in range(3):
            self.Grids[iG] = gl.GLGridItem()
            self.Grids[iG].setSpacing(x=0.1, y=0.1, z=0.1)

        # Notes: X is left-right, Y is front-back, Z is top-bottom

        # XY Grid
        self.Grids[0].setSize(x=0.5, y=1, z=0)
        self.Grids[0].translate(0.25,0.5,0)

        # XZ Grid
        self.Grids[1].setSize(x=0.5, y=0.3, z=0)
        self.Grids[1].translate(0.25,0.15,0)
        self.Grids[1].rotate(90, 1, 0, 0)

        # YZ Grid
        self.Grids[2].setSize(x=0.3, y=1, z=0)
        self.Grids[2].translate(0.15,0.5,0)
        self.Grids[2].rotate(-90, 0, 1, 0)

        for vb in [self.view3D, self.view2D]:
            for iG in range(3):
                vb.addItem(self.Grids[iG])

        # Create Play Button
        self.play_btn = QtWidgets.QPushButton('>')
        self.play_btn.setToolTip('Play/Pause')
        self.play_btn.setMaximumWidth(40)
        self.play_btn.setCheckable(True)
        self.play_btn.setChecked(False)
        self.play_btn.clicked.connect(self.toggle_animation)

        # Create slider
        self.frame_sb = QtWidgets.QSlider(QtCore.Qt.Horizontal)  # QScrollBar(QtCore.Qt.Horizontal)
        self.frame_sb.valueChanged.connect(self.time_scroll)  # Callback function for scrollbar value change

        # Create Frame spin box
        self.frame_spb = pg.SpinBox()  # use pyqgraph spinbox for sigValueChanged() function compatibility
        self.frame_spb.setMaximumWidth(80)
        self.frame_spb.setMinimum(0)
        self.frame_spb.sigValueChanged.connect(self.time_set)  # this function lets rapid subsequent changes be viewed as 1 change.

        # LIMIT THE VIEW
        # EDIT TOP
        self.limit_stop_le = QtWidgets.QLineEdit('1')
        self.limit_stop_le.setMaximumWidth(40)
        self.limit_stop_le.editingFinished.connect(self.limit_stop)

        # EDIT BOTTOM
        self.limit_start_le = QtWidgets.QLineEdit('0')
        self.limit_start_le.setMaximumWidth(40)
        self.limit_start_le.editingFinished.connect(self.limit_start)

        # SCROLLBARS
        # TOP
        self.limit_stop_sb = QtWidgets.QScrollBar(QtCore.Qt.Vertical)
        self.limit_stop_sb.valueChanged.connect(self.limit_stop_scroll)
        self.limit_stop_sb.setInvertedAppearance(True)

        # BOTTOM
        self.limit_start_sb = QtWidgets.QScrollBar(QtCore.Qt.Vertical)
        self.limit_start_sb.valueChanged.connect(self.limit_start_scroll)
        self.limit_start_sb.setInvertedAppearance(True)

        ## dataset scroll bar j
        self.datasets_scrollbar = QtWidgets.QScrollBar(QtCore.Qt.Vertical)
        self.datasets_scrollbar.valueChanged.connect(self.select_index_dataset_scrollbar)
        self.datasets_scrollbar.setRange(0, 0)
        self.datasets_scrollbar.setInvertedAppearance(True)


        # STRUCTURAL DATA MODE
        self.structural_data_sb = QtWidgets.QScrollBar(QtCore.Qt.Vertical)
        self.structural_data_sb.setHidden(True)
        self.structural_data_sb.setInvertedAppearance(True)
        self.structural_data_sb.valueChanged.connect(self.reset_structural_data_plot)

        self.structural_data_le = QtWidgets.QLineEdit(str(self.structural_data_sb.value()+1))
        self.structural_data_le.setHidden(True)
        self.structural_data_le.setMaximumHeight(20)
        self.structural_data_le.setMaximumWidth(20)

        # COLORMAP SELECTOR for selecting Activity Scale (left/right click)
        self._data_loaded = False
        self.cmap = None
        self.alpha_map = None
        self.make_alphamap()
        self._start_color = 0.
        self._stop_color = 1.
        self._alpha = 0.6
        self.data_set = None
        self._cmap_name = 'inferno'
        self._color_normalizer = colors.normalize
        self.make_cmap()
        self.rate_sel_cp = ColorRangePicker(self.cmap)
        width = 30
        height = 300
        self.rate_sel_cp.setMaximumWidth(width)
        self.rate_sel_cp.setMaximumHeight(height)
        self.rate_sel_cp.top_limit.connect(self.limit_color_stop)
        self.rate_sel_cp.bottom_limit.connect(self.limit_color_start)

        # Auto/manual limiting of colors toggle:
        # self.toggle_limiter_description_label = QtWidgets.QLabel('Color limiter:')
        self.toggle_manual_auto_limit_cb = QtWidgets.QComboBox()
        self.toggle_manual_auto_limit_cb.addItem('Auto limits')
        self.toggle_manual_auto_limit_cb.addItem('Manual limits')
        self.toggle_manual_auto_limit_cb.setMaximumWidth(80)
        self.toggle_manual_auto_limit_cb.currentTextChanged.connect(self.toggle_manual_auto_limit)

        # COLORMAP  STOP EDIT
        self.limit_color_stop_sb = QtWidgets.QDoubleSpinBox()  # QLineEdit('0')
        self.limit_color_stop_sb.setValue(1)
        self.limit_color_stop_sb.setFixedWidth(80)
        self.limit_color_stop_sb.setRange(-1000, 10000)
        self.limit_color_stop_sb.setDecimals(2)
        self.limit_color_stop_sb.setSingleStep(0.1)
        # self.LimitColorStartE.setMaximumWidth(30)
        self.limit_color_stop_sb.valueChanged.connect(self.limit_colorbar_stop)

        # COLORMAP  START EDIT
        self.limit_color_start_sb = QtWidgets.QDoubleSpinBox()  # QLineEdit('0')
        self.limit_color_start_sb.setValue(0)
        self.limit_color_start_sb.setFixedWidth(80)
        self.limit_color_start_sb.setRange(-1000, 10000)
        self.limit_color_start_sb.setDecimals(2)
        self.limit_color_start_sb.setSingleStep(0.1)
        # self.LimitColorStartE.setMaximumWidth(30)
        self.limit_color_start_sb.valueChanged.connect(self.limit_colorbar_start)

        # Log color scale switch
        self.log_colorscale_cb = QtWidgets.QCheckBox('Log color\nscale')
        self.log_colorscale_cb.stateChanged.connect(self.log_color_scale)

        # Create Alpha Value selector via Colormap
        self.alpha_sel_cp = ColorRangePicker(self.alpha_map)  # change to gray gradient
        self.alpha_sel_cp.setMaximumWidth(width)
        self.alpha_sel_cp.setMaximumHeight(int(height / 2))
        self.alpha_sel_cp.top_limit.connect(self.alpha_setter)  # connect to callback that sets the overall alpha value

        # Create Checkbox Group
        self.options_box = QtWidgets.QGroupBox('Visual Options')
        self.button_layout = QtWidgets.QHBoxLayout()

        # Dropdown menu with data selection options
        self.data_sel_cb = QtWidgets.QComboBox()
        self.data_sel_cb.setFixedWidth(100)
        self.data_sel_cb.addItem('df')
        self.data_sel_cb.currentIndexChanged.connect(self.selection_change)
        self.button_layout.addWidget(self.data_sel_cb)

        # Button to add additional options to data selection dropdown menu (self.DataSel)
        self.add_data_btn = QtWidgets.QPushButton('Add static Data', self)
        self.add_data_btn.setFixedWidth(120)
        self.add_data_btn.clicked.connect(self.add_static_gui)
        self.button_layout.addWidget(self.add_data_btn)
        # adddata_shortcut = QtWidgets.QShortcut('Ctrl+A',self.MWidget)
        # adddata_shortcut.activated.connect(self.load_static)

        # self.activity_view_btn = QtWidgets.QPushButton('Activity view mode', self)
        # self.activity_view_btn.setFixedWidth(140)
        # self.activity_view_btn.clicked.connect(self.activity_view_mode)
        # self.button_layout.addWidget(self.activity_view_btn)
        #
        # self.hu_view_btn = QtWidgets.QPushButton('HU view mode', self)
        # self.hu_view_btn.setFixedWidth(100)
        # self.hu_view_btn.clicked.connect(self.hu_view_mode)
        # self.button_layout.addWidget(self.hu_view_btn)

        # Region
        self.region_check_cb = QtWidgets.QCheckBox('Draw region hulls')
        self.button_layout.addWidget(self.region_check_cb)
        self.region_check_cb.stateChanged.connect(self.region_show)

        self.neuron_check_cb = QtWidgets.QCheckBox('Select neurons in region')
        self.button_layout.addWidget(self.neuron_check_cb)
        self.neuron_check_cb.stateChanged.connect(self.neuron_show)

        self.structural_data_check_cb = QtWidgets.QCheckBox('Struct. data')
        self.button_layout.addWidget(self.structural_data_check_cb)
        self.structural_data_check_cb.stateChanged.connect(self.structural_data_input)

        self.region_sel_cb = ExtendedComboBox()  # QtWidgets.QComboBox()
        self.region_sel_cb.setFixedWidth(300)
        self.region_sel_cb.currentIndexChanged[str].connect(self.region_changed_by_combobox)
        self.button_layout.addWidget(self.region_sel_cb)

        self.labelled_cb = QtWidgets.QCheckBox('Labelled only')
        self.button_layout.addWidget(self.labelled_cb)
        self.labelled_cb.stateChanged.connect(self.set_labelled)

        self.options_box.setLayout(self.button_layout)
        self.options_box.setMaximumHeight(70)

        # Behavior + Neuronal activity plots
        self.area_plot = DockArea()
        self.beh_dock = Dock('Stimulus and Behavior')
        self.activity_dock = Dock('Neuronal Activity')
        self.area_plot.addDock(self.beh_dock, 'bottom')
        self.area_plot.addDock(self.activity_dock, 'left', self.beh_dock)
        self.area_plot.moveDock(self.beh_dock, 'above', self.activity_dock)
        # Behavior plot
        self.behavior_pw = pg.PlotWidget()
        self.behavior_pw.setYRange(-2, 2)  # XXRange YRange: default values, are updated upon data loading
        self.behavior_pw.setXRange(0, 3000)
        self.behavior_pw.setLabel('left', 'stimulus and behavior', color='r')
        # self.behavior_pw.getAxis('left').setPen(pg.mkPen(color='r'))
        self.behavior_pw.setLabel('bottom', 'Time', unit='s')
        self.behavior_pw.showAxis('right')  # use non-used right axis for mean selections
        self.behavior_pw.setLabel('right', 'df/f of selection', color='w')
        self.behavior_pw.hideButtons()  # The innate functionality of the 'A'-button is hidden, because it is not compatible with the current lay-out (due to multiple viewboxes?)

        """To add DF/F data in behavior/stimulus plot, two different scales are
        needed for visibility. DF/F data can be plotted by using the 'hide points'
        functionality (i.e. edit_hide()).
        To do this one needs to put plots in ViewBoxes, and link these VBs to
        the main widget (behavior_pw)

        #TODO: There is still one issue; which is that stimulus_plot and
        behavior_plot overload the left axis of behavior_pw. This is relevant for
        data sets containing both behavioral and stimulus data. Currently
        stimulus_plot is prioritized. This could maybe be solved by allowing an
        additional third axis (second on the left), but that is more involved..
        """
        self.stimulus_plot = pg.ViewBox()  # Viewbox for plot of stimulus
        self.behavior_pw.addItem(self.stimulus_plot)
        self.behavior_plot = pg.ViewBox()  # Viewbox for plot of behavior
        self.behavior_pw.addItem(self.behavior_plot)
        self.mean_selection_plot = pg.ViewBox()  # ViewBox for plot of single df signal
        self.behavior_pw.addItem(self.mean_selection_plot)  # add ViexBox to PW
        self.behavior_pw.getAxis('left').linkToView(self.behavior_plot)
        self.behavior_pw.getAxis('left').linkToView(self.stimulus_plot)
        self.behavior_pw.getAxis('right').linkToView(self.mean_selection_plot)  # connect axis to VB
        self.behavior_plot.setXLink(self.behavior_pw)
        self.stimulus_plot.setXLink(self.behavior_pw)
        self.mean_selection_plot.setXLink(self.behavior_pw)  # DF and stim/behav link by time

        self.behavior_pw.sigRangeChanged.connect(self.update_behavior_plot)  # connect so that VB scales with PW

        self.beh_time = pg.InfiniteLine(0, movable=True)
        self.beh_time.sigPositionChangeFinished.connect(self.plot_time_changed)
        self.behavior_pw.addItem(self.beh_time)
        self.beh_dock.addWidget(self.behavior_pw)
        self.beh_stim_text = pg.TextItem(text='Behavior (white), Stimulus (red)', color=[255, 255, 255], anchor=(0, 0))
        self.beh_stim_text.setPos(0, 3)
        self.behavior_pw.addItem(self.beh_stim_text)
        self.behavior_pw.invertY(True)  # It is still not so clear to me why this is needed, but it does the trick.!

        # Activity plot
        self.activity_pw = pg.PlotWidget()
        self.activity_plots = [pg.PlotDataItem(), pg.PlotDataItem()]  # raw calcium signal
        for iP in [0, 1]:
            self.activity_pw.addItem(self.activity_plots[iP])
        self.activity_plots[0].setPen(pg.mkPen('w'))
        self.activity_plots[1].setPen(pg.mkPen('r'))
        self.activity_plot_spikes = [pg.PlotDataItem(), pg.PlotDataItem()]  # deconvolved spikes
        for iP in [0, 1]:
            self.activity_pw.addItem(self.activity_plot_spikes[iP])
        self.activity_plot_spikes[0].setPen(pg.mkPen('y'))
        self.activity_plot_spikes[1].setPen(pg.mkPen('b'))
        self.activity_plot_calcfit = [pg.PlotDataItem(), pg.PlotDataItem()]  # calcium fit by decovolved spikes
        for iP in [0, 1]:
            self.activity_pw.addItem(self.activity_plot_calcfit[iP])
        self.activity_plot_calcfit[0].setPen(pg.mkPen('y'))  # same color as corresponding spikes
        self.activity_plot_calcfit[1].setPen(pg.mkPen('b'))
        self.activity_pw.setYRange(0.5, 3.2)
        self.activity_time = pg.InfiniteLine(0, movable=True)
        self.activity_time.sigPositionChangeFinished.connect(self.plot_time_changed)
        self.activity_pw.addItem(self.activity_time)
        self.activity_dock.addWidget(self.activity_pw)
        self.activity_text = pg.TextItem(text='Neuron', color=[255, 255, 255], anchor=(0, 0))
        self.activity_text.setPos(0, 3)
        self.activity_pw.addItem(self.activity_text)

        # LAYOUT
        # Splitter
        self.v_splitter.addWidget(self.view3D)
        self.v_splitter.addWidget(self.view2D)
        self.v_splitter.addWidget(self.area_plot)
        self.v_splitter.setStretchFactor(10, 1)
        # 3D Widget
        self.v_layout_center.addWidget(self.v_splitter)  # ,ViewerY,ViewerX,ViewerH,ViewerW)
        # Time scroll bar
        self.v_layout_center.addWidget(self.frame_sb)  # ,ViewerY+ViewerH,1,1,DivX-2)
        # Space range selector
        self.v_layout_left.addWidget(self.limit_stop_le)  # ,0,0,1,1)
        self.v_layout_left.addWidget(self.limit_stop_sb)  # ,1,0,2,1)
        self.v_layout_left.addWidget(self.limit_start_sb)  # ,3,0,2,1)
        self.v_layout_left.addWidget(self.limit_start_le)  # ,ViewerY+ViewerH-1,0,1,1)
        self.v_layout_left.addWidget(self.datasets_scrollbar)
        # Play button
        self.v_layout_left.addWidget(self.play_btn)  # ,ViewerY+ViewerH,0,1,1)
        # Structural data
        self.v_layout_center_right.addWidget(self.structural_data_le)
        self.v_layout_center_right.addWidget(self.structural_data_sb)
        # Color bar
        # self.v_layout_right.addWidget(self.toggle_limiter_description_label)
        self.v_layout_right.addWidget(self.toggle_manual_auto_limit_cb)
        self.v_layout_right.addWidget(self.limit_color_stop_sb)  # ,0,DivX-1,1,1)
        self.v_layout_right.addWidget(self.rate_sel_cp)  # ,1,DivX-1,ViewerH-alpha_h-2,1)
        self.v_layout_right.addWidget(self.limit_color_start_sb)  # ,ViewerH-alpha_h-1,DivX-1,1,1)
        self.v_layout_right.addWidget(self.log_colorscale_cb)
        self.v_layout_right.addWidget(self.alpha_sel_cp)  # ,ViewerH-alpha_h,DivX-1,alpha_h,1)
        # Time text display
        self.v_layout_right.addWidget(self.frame_spb)  # ,ViewerY+ViewerH,DivX-1,1,1)
        # Options
        self.v_layout_main.addWidget(self.options_box)  # ,ViewerY+ViewerH+1,0,1,DivX)

        # Other dialog boxes
        self.select_cell_dialog = SelectCellDialog(self)
        self.static_data_dialog = StaticDataDialog(self)
        self.compute_correlation_dialog = CorrelationDialog(self)
        self.shortcuts_dialog = ShortcutsDialog(self)
        # Jupyter kernel for external console
        self.setCentralWidget(self.main_widget)
        self.kernel = InternalIPKernel()
        self.kernel.init_ipkernel('qt')
        self.qt_console = None
        # Video writer for export
        self.video_writer = None
        self._export_video = False

    def closeEvent(self, a0: QtGui.QCloseEvent) -> None:
        """closeEvent of Viewer.

        Calls leaving() to close kernel and then closes the UI"""
        self.leaving()
        a0.accept()

    def leaving(self):
        """Close the iPython console."""
        if self.qt_console is not None:
            self.qt_console.kill()
        self.kernel.ipkernel.exit()

    def console(self):
        """Create an iPython console"""
        if self.qt_console is not None:
            return self.qt_console
        self.kernel.namespace['self'] = self
        self.qt_console = self.kernel.new_qt_console()

    def cluster_color(self):
        pass

    @property
    def color_normalizer(self):
        return self._color_normalizer

    @color_normalizer.setter
    def color_normalizer(self, value):
        self._color_normalizer = value
        self.make_cmap()

    def make_cmap(self):
        """Create a color map"""
        # color = np.array([(0, 0, 255, self.alpha*255), (128, 0, 128, self.alpha*255),
        #                   (255, 0, 0, self.alpha*255),(255, 255, 255, self.alpha*255)],
        #                  dtype=np.ubyte)  # Define the color at the stop points, here for a fire like colormap
        # color = np.array(plt.cm.inferno.colors)*255

        if self._cmap_name == 'GFP':
            all_colors = np.zeros((256, 3))
            all_colors[:, 1] = np.arange(0, 256)
        else:
            all_colors = colors.get_cmap(self._cmap_name)
            # colors = np.array(plt.get_cmap(self._cmap_name).colors) * 255

        # FIXME: Start and Stop colors not taken into account yet
        # colors = np.ubyte(colors)
        # pos = np.linspace(self.start_color, self.stop_color,
        #                   colors.shape[0])  # Define stop points on the color map, on the [0, 1] range
        cmap_steps = all_colors.shape[0]
        cmap_steps_res = (1 - (self.start_color + (1-self.stop_color)))/cmap_steps
        cmap_presteps = int(np.round(self.start_color/cmap_steps_res))
        cmap_poststeps = int(np.round((1-self.stop_color)/cmap_steps_res))
        cmap_steps_new = cmap_steps + cmap_presteps + cmap_poststeps
        if cmap_presteps > 0:
            pre_color = np.tile(all_colors[0,:],(cmap_presteps,1))
            all_colors = np.concatenate((pre_color, all_colors),axis=0)

        if cmap_poststeps > 0:
            post_color = np.tile(all_colors[-1,:],(cmap_poststeps,1))
            all_colors = np.concatenate((all_colors,post_color), axis=0)

        # Include alpha value as the 4th column in all_colors
        all_colors = np.vstack((all_colors.transpose(), [self.alpha] * all_colors.shape[0])).transpose()


        if self.data_set:  # defined as None when Fishualizer initiates
            if not self.swap_color_data[self.data_set]: # normal, color axis not reversed
                # self.cmap = pg.ColorMap(pos, colors[::1])  # normal
                self.cmap = colors.linear_cmap(all_colors, self.color_normalizer)
            elif self.swap_color_data[self.data_set]:  # if swap == False, reverse color axis
                # self.cmap = pg.ColorMap(pos, colors[::-1])  # reverse (what we were used to)
                self.cmap = colors.linear_cmap(all_colors[::-1, :], self.color_normalizer)
        else:
            self.cmap = colors.linear_cmap(all_colors, self.color_normalizer)

    def make_alphamap(self):
        """ Create a transparency map"""
        pos = np.array([1., 0.])  # Define stop points on the color map, on the [0, 1] range
        alphas = np.array([(0, 0, 0, 1), (1, 1, 1, 1)])  # Define the color at the stop points
        all_alphas = colors.interp_colors(pos, alphas)
        # self.alpha_map = pg.ColorMap(pos, color)
        self.alpha_map = colors.linear_cmap(all_alphas)

    def reset_view(self):
        pass

    def copy_clipboard(self):
        pass

    def set_perspective_3d(self, view=None):
        """Set perspective of self.view3D to a 3D view (default)."""
        if view is None:
            view = self.last_viewbox[0]
        view.opts['distance'] = 1
        view.opts['fov'] = 60
        if self.perspective_3d_action.isChecked() is False and view == self.view3D:  # if this function was called without pressing the button, then also correclty set it  #NB: technically this function is called again upon setting the button, could be fixed
            self.perspective_3d_action.setChecked(True)
        self._logger.info(f'viewbox {view} is set to 3D')

    def set_perspective_2d(self, view=None):
        """Set perspective of self.view3D to a 2D view. This is done by effectively
        moving out (increase distance) and zooming in (decrease fov).
        The smaller self.view3D.opts['fov'] is, the closer one comes to a 2D view,
        but rendering might be slower for very small values of fov."""
        if view is None:
            view = self.last_viewbox[0]
        view.opts['distance'] = 150
        view.opts['fov'] = 0.5  # hard-coded here, could be changed manually in console.
        if self.perspective_2d_action.isChecked() is False and view == self.view3D:  # if this function was called without pressing the button, then also correclty set it  #NB: technically this function is called again upon setting the button, could be fixed
            self.perspective_2d_action.setChecked(True)
        self._logger.info(f'viewbox {view} is set to 2D')
        self.view2D.update()
        self.view3D.update()

    def log_color_scale(self):
        pass

    def switch_alpha_mode(self, mode):
        pass

    def edit_cmap(self):
        pass

    def edit_hide(self):
        pass

    def compute_correlation_gui(self):
        pass

    def compute_mean_activity(self):
        pass

    def help_shortcuts(self):
        """Show the shortcut dialog pop up."""
        self.shortcuts_dialog.show()

    def help_log(self):
        """Show the logger."""
        self.log_win.show()

    def add_static_gui(self):
        pass

    def show_select_cell_dialog(self):
        """Show the select cell dialog."""
        self.select_cell_dialog.show()

    def plot_time_changed(self, marker):
        pass

    def time_scroll(self):
        pass

    def time_set(self, value):
        pass

    def hu_view_mode(self):
        pass

    def activity_view_mode(self):
        pass

    def select_data_file(self):
        pass

    def update_plot(self):
        pass

    def layers_scroll_top(self):
        pass

    def layers_scroll_bottom(self):
        pass

    def play(self, event):
        pass

    def video_export(self, event):
        pass

    def grab_plot(self):
        pass

    def toggle_manual_auto_limit(self, text):
        pass

    def color_top_limit(self, pos):
        pass

    def color_bottom_limit(self, pos):
        pass

    def regionchange(self, text):
        pass

    def regionconfirmed(self):
        pass

    def selection_change(self, text):
        pass

    def add_data(self):
        pass

    def load_density(self):
        pass

    def fast_deconvolv(self, n):
        pass

    def toggle_calcium_fit(self):
        pass

    def show_region_current_cell(self):
        pass

    def toggle_hideneuropil(self):
        pass

    def toggle_hidegrid(self):
        pass

    def toggle_somas(self):
        pass

    def toggle_nanview(self):
        pass

    def toggle_randomzfill(self):
        pass

    def increase_dot_size(self):
        pass

    def decrease_dot_size(self):
        pass

    def switch_topside(self):
        pass

    def swap_color_axis(self):
        pass

    def update_behavior_plot(self):
        pass

    def call_reverse_z_coords(self):
        pass

    def call_switch_xy_coords(self):
        pass

    def inverse_x_axis(self):
        pass

    def inverse_y_axis(self):
        pass

    def inverse_z_axis(self):
        pass

    def toggle_xy_outline(self):
        pass

    def toggle_xz_outline(self):
        pass

    def toggle_yz_outline(self):
        pass

    def draw_xy_outline(self):
        pass

    def draw_xz_outline(self):
        pass

    def draw_yz_outline(self):
        pass

    def show_top_side_outlines(self):
        pass

    def choose_coordinates(self):
        pass

    def limitx(self):
        pass

    def limity(self):
        pass

    def limitz(self):
        pass

    def change_region_view(self):
        pass

    def change_playtimer_interval(self):
        pass

    def save_checking(self, action, check=True):
        """set action to check, but block signals while doing so (to prevent
        recurrently calling switch_alpha_mode())."""
        action.blockSignals(True)
        action.setChecked(check)
        action.blockSignals(False)

    @property
    def alpha(self):
        return self._alpha

    @property
    def start_color(self):
        return self._start_color

    @property
    def stop_color(self):
        return self._stop_color

class Fishualizer(Viewer):
    def __init__(self, argv):
        """
        Actual class defining the GUI behavior. This class will be instantiated to create the main window.
        """
        # create and initialize window
        super(Fishualizer, self).__init__()

        # parse input arguments
        parser = argparse.ArgumentParser(description='Fishualizer')

        parser.add_argument('datafile',type=str,nargs='?',default=None,
                            help='Path to h5 datafile')
        parser.add_argument('--loadram', type=bool, nargs=1, default=True,
                            help='Load ram Option')
        parser.add_argument('--visstyle', type=str, nargs=1, default='transparent',
                            help='Visualization Option')
        parser.add_argument('--ignorelags', action='store_true',
                            help='Ignore the time lags provided across layers')
        parser.add_argument('--forceinterpolation', action='store_true',
                            help='Force redoing the interpolation (useful for debugging)')
        parser.add_argument('--ignoreunknowndata', action='store_true',
                            help='Ignore data sets that are not recognised automatically')

        self.args = parser.parse_args()


        # Parameters and constants
        self._nb_times = 1  # Number of time points in the data
        self._current_time = 0  # Current time point viewed
        self._max_time = 0  # Maximum length of the video
        self._current_frame = 0  # Current Frame
        self._limit_start = [-10, -10, -10]
        self._limit_stop = [10, 10, 10]
        self._copy_clipboard = True

        self._last_selection = ['df', 'df']
        self.activity_plot_ind = 0
        self.disp_start = {
            'df': 1}  # this is overwritten in load_data(), but needed for pre data load selection_change() (when switching between data sets) in load_data()
        self.disp_stop = {'df': 1.5}
        self.disp_change = {'df': False}  # whether limits should be recalculated (avoids unnecesssary calcuations i.e. runtime)
        self.fix_colorscale = False
        self.swap_color_data = {'df': False}  # default to swap (reverse color scale)
        self.bool_reverse_x = False   # reverse x coords ?
        self.bool_reverse_y = False   # reverse y coords ?
        self.bool_reverse_z = False   # reverse z coords ?
        self.data_set = 'df'  # set default data_set (should be same as default QComboBox Button)
        self.current_selected_cell = 0
        self._selection_updated = True
        self._data_loaded = 0
        self._animate = False
        self._dynamics_on_play = True
        self._rotate_on_play = False
        self._labelled_only = False
        self.labelnames = None
        self._labels_available = False
        self._use_zbrainatlas_coords = False
        self.reference_x_axis = None  # coordinate system axes, are initiated in update_coordinate_system()
        self.reference_y_axis = None
        self.reference_z_axis = None
        self.bool_show_ref_axes = False
        self.scale_bar = None
        self.bool_topside_outline = False
        self._data_path = ''  # Path to the data file
        self.Plot3D = None  # Pointer to the 3D plot.
        self._data = None  # Loaded data
        self.hulls = None  # ROI hull
        self.density_plot = None
        self.outline_plot = {0: None, 1:None, 2:None}
        self.region_plot = {x: {'left': {}, 'right': {}, 'both': {}} for x in range(3)}
        self.c_roi = 1  # Current ROI indexf
        self._alpha_mode = 'additive'  # Current color blending mode
        self.dot_size = 6
        self.z_random_shift = False
        self.z_coords_unique = None

        self.brightness_multiplier = 8
        self.structural_black_th = 10
        self._rec = Zecording('', **{'ignorelags': self.args.ignorelags,
                                     'forceinterpolation': self.args.forceinterpolation,
                                     'ignoreunknowndata': self.args.ignoreunknowndata,
                                     'parent': self})  # Zecording object
        # Initialize a QTimer used to play the movie.
        # Here the time interval is fixed but could be made into a setting in the GUI
        self.frame_rate = 10
        self._playtimer = QtCore.QTimer()
        self._playtimer.setInterval((1000 / self.frame_rate))  # Interval between frames in ms
        self._playtimer.timeout.connect(self.animate)  # Connect the Timer to a callback function called every x ms
        self.animate_mode = 'default'  # mode used in self.animate()
        self.default_animate_azimuth_step = 1
        # Initialize a QTimer used to export a movie
        self._export_timer = QtCore.QTimer()
        self._export_timer.setInterval(1000 // 30)  # Interval between frames in ms (60 frames per sec)
        self._export_timer.timeout.connect(
            self.video_recording)  # Connect the Timer to a callback function called every x ms

        self.setWindowTitle("Fishualizer")
        self.plotopts = {'centered': True}  # 'centered' if True: center at median of coords, if False center at (0,0,0)
        self.resize_scrollbar()
        self.region_indices = None
        self.neurons_set()
        self.hull_color = np.array([1.0, .4, 0.0, 0.1])  # TODO why do we need two hull colors?
        self.hull_color_dict = {}
        self.region_spec_color = {}
        self.current_hull_color = np.array([.82, .82, .82, 0.1])  # grey color
        self.main_zbrain_divisions = np.array([0, 93, 113, 259, 274])
        self.panstep = 0.005
        self.data_names = {0: 'df'}  # , 1: 'spikes'} # predefined datasets
        self.i_dataset = 0  # number of predefined data sets (with zero-indexing)
        self.hide_neuropil = False  # if 'on', only the neurons in neurons_show are well visible, if False regular Fishualizer (all neurons)
        self.hide_grid = False # if 'on' the grid is shown
        self.hide_nans = True # choose whether to show the NaN values in gray or make them transparent
        self.show_somas = False

        self.plot_calcium_fit = 1
        self.cl_colors = {}
        self.reg_colors = {}
        # Logging
        self._logger = logging.getLogger('Fishlog')
        self._logger.setLevel(logging.DEBUG)
        self._log_handler = RotatingFileHandler(os.path.join(fishualizer_folder, 'Content/Logs/Fishualizer.log'), maxBytes=1e6, backupCount=1)
        formatter = logging.Formatter(
            '%(asctime)s :: %(filename)s :: %(funcName)s :: line %(lineno)d :: %(levelname)s :: %(message)s',
            datefmt='%Y-%m-%d:%H:%M:%S')
        self._log_handler.setFormatter(formatter)
        self._logger.addHandler(self._log_handler)
        self.log_win = LogWindow(self)
        self.log_win.log_handler.setFormatter(formatter)
        self._logger.info('Init is over')
        self.statusBar().showMessage('Initialization is finished')
        sys.excepthook = handle_exception
        self._region_selected_prior = [False, 0]
        self.setWindowIcon(QtGui.QIcon(os.path.join(fishualizer_folder, 'Content/icon.png')))
        self.user_params = load_config()
        self._selected_inds = 0
        self.filter_dict = {}
        self.export_dir = os.getcwd()

        if self.args.datafile is not None:
            self._data_path = os.path.expanduser(self.args.datafile)
            self.load_data()

    def closeEvent(self, a0: QtGui.QCloseEvent):
        """closeEvent of Fishualizer.

        This function is called automatically if UI is closed.
        Closes rec class, informs logger and calls Viewer.closeEvent() which
        further handles closing.
        """
        if self.rec.path is not None:
            self.rec.close()
            self._logger.info('File closed')
        super().closeEvent(a0)

    def select_data_file(self):
        """
        Callback function called to open a data file
        """
        # Open a dialog box in the current directory and filter a few possible extensions
        cwd = self.user_params['paths']['data_path']  # os.getcwd()
        # file_dialog = QtWidgets.QFileDialog(self, 'Choose a data file', cwd, 'Data (*.h5)')
        file_dialog = OpenDataDialog(self, 'Choose a data file', cwd, 'Data (*.h5)')
        file_dialog.loadram_cbx.setChecked(self.user_params.get('loadram', False))
        dpath, _filter, ignorelags, forceinterp, ignoreunknowndata, loadram = file_dialog.getOpenFileName(self,
                                                                                        "Choose a data file", cwd, 'Data (*.h5 *.hdf5)')
        self.args.ignorelags = ignorelags
        self.args.forceinterpolation = forceinterp
        self.args.ignoreunknowndata = ignoreunknowndata
        self.args.loadram = loadram
        if dpath != '':
            # Save the data path folder for later
            self.user_params['paths']['datapath'] = os.path.dirname(dpath)
            # Set the new data path. The data_path setter will take care of the rest
            self.data_path = dpath

    def log_color_scale(self):
        if self.log_colorscale_cb.isChecked():
            self.color_normalizer = colors.log_normalize
        else:
            self.color_normalizer = colors.normalize
        self.update_plot()

    @property
    def alpha_mode(self):
        return self._alpha_mode

    @alpha_mode.setter
    def alpha_mode(self, value):
        """Property that is the current transparency modes.

        The setter function checks if self.Plot3D exists, changes its GLOption and
        calls update_plot().

        Parameter:
        -----------
            value: str (pre-defined)
                new alpha mode
        """
        self._alpha_mode = value
        if value is 'maximum':
            value = {'glBlendFunc': (OpenGL.GL.GL_ONE, OpenGL.GL.GL_ONE),
                     'glBlendEquation': (OpenGL.GL.EXT.blend_minmax.GL_MAX_EXT, ), OpenGL.GL.GL_BLEND: True}
        if self.Plot3D is not None:
            self.Plot3D.setGLOptions(value)
        for vb in [self.view2D, self.view3D]:
            vb.update()

    def save_checking(self, action, check=True):
        """set action to check, but block signals while doing so (to prevent
        recurrently calling switch_alpha_mode())."""
        action.blockSignals(True)
        action.setChecked(check)
        action.blockSignals(False)

    def switch_alpha_mode(self, mode='additive'):
        """Change the transparency mode via property self.alpha_mode

        Parameter:
        ----------
            mode: str, (predefined)
                'additive'
                'translucent'
                'opaque'
        """
        self.alpha_mode = mode
        for ac in [self.a_action, self.t_action, self.o_action, self.max_action]:
            self.save_checking(self.a_action, False)  # uncheck all options

        if mode == 'additive':  # check the current option
            self.save_checking(self.a_action, True)
        elif mode == 'translucent':
            self.save_checking(self.t_action, True)
        elif mode == 'opaque':
            self.save_checking(self.o_action, True)
        elif mode == 'maximum':
            self.save_checking(self.max_action, True)
        for vb in [self.view2D, self.view3D]:
            vb.update()

    def time_scroll(self):
        """ Callback function updating the plot when scroll bar is moved."""
        super().time_scroll()
        self.current_frame = self.frame_sb.value()
        # self.current_time = float(self.current_frame)/self.rec.sampling_rate

    def toggle_manual_auto_limit(self, selection):
        """ Callback function toggling between manual and auto color limiting."""
        # could also be implemented by a simple boolean switch, but this architecture allows possible future extensions
        print(selection)
        if selection == 'Auto limits':
            self.fix_colorscale = False
        elif selection == 'Manual limits':
            self.fix_colorscale = True

    def limit_color_start(self, pos):
        """
        Callback handling the setting of the bottom_limit of the color map

        Parameters
        ----------
        pos: float
            Relative position of the click (between 0 and 1)
        """
        pos = 1-pos

        if pos > self.stop_color:
            tmp = self.start_color
            self.stop_color = pos
            pos = tmp

        self.start_color = pos
        self.rate_sel_cp.cmap = self.cmap


    def limit_color_stop(self, pos):
        """
        Callback handling the setting of the top_limit of the color map

        Parameters
        ----------
        pos: float
            Relative position of the click (between 0 and 1)
        """
        pos = 1-pos

        if pos < self.start_color:
            tmp = self.stop_color
            self.start_color = pos
            pos = tmp

        self.stop_color = pos  # New stop position. Will update the color map automagically
        self.rate_sel_cp.cmap = self.cmap  # Set the colormap in the colorpicker


    def limit_colorbar_start(self, value):
        """Function to change the value of self.disp_start of current data set

        This function is called by valueChanged of limit_color_start_sb (user numerical input QSpinBox)
        It also updates the cmap and the plot.

        Parameter:
        -----------
            value: float
                new value of disp_start
        """
        # text = self.LimitColorStartE.text()
        # if len(text)>0:
        #     if self.LimitColorStartE.isModified():
        #         self.disp_start[self.data_set] = float(text)
        #         self.update_plot()
        self.disp_start[self.data_set] = value
        self.limit_color_stop_sb.setMinimum(value)
        # self.make_cmap()
        self.update_plot()

    def limit_colorbar_stop(self, value):
        """Function to change the value of self.disp_stop

        This function is called by valueChanged of limit_color_stop_sb (user numerical input QSpinBox)
        It also updates the cmap and the plot.

        Parameter:
        -----------
            value: float
                new value of disp_stop
        """
        # text = self.LimitColorStopE.text()
        # if len(text)>0:
        #     if self.LimitColorStopE.isModified():
        #         self.disp_stop[self.data_set] = float(text)
        #         self.update_plot()
        self.disp_stop[self.data_set] = value
        self.limit_color_start_sb.setMaximum(value)
        # self.make_cmap()
        self.update_plot()

    def time_set(self, box_value):
        """Callback function for frame_spb() that updates the plot when a time is entered."""
        self.current_time = box_value.value()

    def step_fwd(self):
        """Step one frame forwards in time"""
        self.current_time = self.current_time + 1 / self.rec.sampling_rate

    def step_bwd(self):
        """Step one frame backwards in time"""
        self.current_time = self.current_time - 1 / self.rec.sampling_rate

    def alpha_setter(self, pos):
        """
        Callback handling the setting of a new alpha value through the alpha selector

        Parameters
        ----------
        pos: float
            Relative position of the click (between 0 and 1)
        """
        self.alpha = pos

    def make_cmap(self):
        """
        Update the color map in response to a change in start/stop/alpha value and update the plot accordingly
        """
        super().make_cmap()
        self.update_plot()

    @property
    def start_color(self):
        return self._start_color

    @start_color.setter
    def start_color(self, value):
        """Start color value of variable color range bar. Also update cmap in setter."""
        self._start_color = value
        self.make_cmap()  # update the colormap to reflect the new start color

    @property
    def stop_color(self):
        return self._stop_color

    @stop_color.setter
    def stop_color(self, value):
        """Stop color value of variable color range bar. Also update cmap in setter."""
        self._stop_color = value
        self.make_cmap()  # update the colormap to reflect the new stop color

    @property
    def alpha(self):
        return self._alpha

    @alpha.setter
    def alpha(self, value):
        """Property with the value of alpha (transparency)

        The setter updates the value and also calls self.make_cmap() to update

        Paramter:
            value: float
                new value for alpha (min=0, max=1, but values outside this range
                are also handled)
        """
        self._alpha = value
        self.make_cmap()  # update the colormap to reflect the new alpha value

    @property
    def current_time(self):
        return self._current_time

    @current_time.setter
    def current_time(self, value):
        """
        Set the new current time. Values are clipped between 0 and the maximal time.
        Current frame is automatically updated.

        Parameters
        ----------
        value: float
        """
        self._current_time = np.clip(value, 0, self._max_time)
        self.current_frame = int(self.current_time * self.rec.sampling_rate)

    @property
    def current_frame(self):
        return self._current_frame

    @current_frame.setter
    def current_frame(self, value):
        """
        Set the new current frame. Values are clipped between 0 and the number of frames
        Current time is automatically updated.
        Scrollbar position and frame number text are updated
        Plot is refreshed

        Parameters
        ----------
        value: int

        """
        self._current_frame = np.clip(value, 0, self.rec.n_times - 1)
        self.frame_sb.blockSignals(True)
        self.frame_spb.blockSignals(True)
        self.frame_sb.setValue(self._current_frame)
        # self.frame_le.setText(str(self._current_time))
        self._current_time = float(self.current_frame) / self.rec.sampling_rate
        self.frame_spb.setValue(self._current_time)
        self.beh_time.setValue(self._current_time)
        self.activity_time.setValue(self._current_time)

        self.frame_sb.blockSignals(False)
        self.frame_spb.blockSignals(False)
        self.update_plot()

    def selected_inds_filters(self):
        """ This function controls the indices made by all filters

        self.filter_dict contains the names of the filters as keys
        and contains the indices that have to be shown for every filter

        The dictionary will be empty upon loading and after "view all neurons" has been clicked
        """
        index_skip = True
        for filter in self.filter_dict.values():
            if index_skip == False:
                cells = np.intersect1d(cells, filter)
            if index_skip:
                cells = filter
                index_skip = False
        self._selected_inds = cells
        self._selection_updated = True  # Consequently update_plot() will update the shown cells
        self.update_plot()

    def load_data(self):
        """Open the data file and load the data

        Set the self.rec attribute whose setter will take care of the rest
        Plot the behavior/stimulus plot
        Add labels to menu if labels are available
        Set current self.view3D=None and let update_plot() make a new self.view3D

        """
        if self.rec.path is not None:
            self.rec.close()
            self.region_check_cb.setChecked(False)
        self._logger.info('Loading data...')
        # self.statusBar().showMessage('Loading data ...')
        self.statusBar().showMessage(f'RAM used: {str(ps.virtual_memory().percent)}%')
        # Get the extension
        basedir = os.path.dirname(self.data_path)
        # set back to default,
        # because if data is changed while running from sampledata_spike to sampledata it would crash otherwise
        self.data_sel_cb.setCurrentIndex(0)  # change button
        # remove all options other than 'df' from dropdown menu  (convenient when changing between datasets)
        while self.i_dataset >= 1:
            self.data_sel_cb.removeItem(self.i_dataset)  # remove last (remaining) item
            self.i_dataset -= 1  # decrease until only self.iDataSet = 0 is left
        self.data_names = {0: 'df'}
        self.data_set = 'df'  # set default data_set (should be same as default QComboBox Button)

        # Open the recording
        self.rec = Zecording(self.data_path, **{'ignorelags': self.args.ignorelags,
                                                'forceinterpolation': self.args.forceinterpolation,
                                                'ignoreunknowndata': self.args.ignoreunknowndata,
                                                'parent': self,
                                                'loadram': self.args.loadram})

        self._selected_inds = np.arange(self.rec.n_cells)
        self.setWindowTitle(f'Fishualizer - {self.data_path}')
        self.choose_coordinates(bool_update_plot=False)
        if len([k for k in self.rec.available_data if 'coord' in k]) > 1:  # if multiple coord systems are present, enable switching via menu UI
            self.choose_coordinates_action.setEnabled(True)
        else:
            self.choose_coordinates_action.setEnabled(False)
        self.behavior_plot.clear()
        self.stimulus_plot.clear()
        self.behavior_pw.removeItem(self.beh_stim_text)  # clear previous legen

        if 'behavior' in self.rec.available_data:
            """Arrange plotting of behavior and/or stimulus
            """
            self.behavior_plot.addItem(pg.PlotDataItem(x=self.rec.times, y=self.rec.behavior, pen='y'))
            if 'stimulus' in self.rec.available_data:  # stimulus and behavior
                self.stimulus_plot.addItem(pg.PlotDataItem(x=self.rec.times, y=self.rec.stimulus, pen='r'))
                self.behavior_pw.setYRange(np.min(np.minimum(self.rec.behavior, self.rec.stimulus)),
                                           np.max(np.maximum(self.rec.behavior, self.rec.stimulus)) + 6)  # lay out
                self.beh_stim_text = pg.TextItem(text='Behavior (yellow), Stimulus (red)',
                                                 color=[255, 255, 255], anchor=(0, 0))
                self.beh_stim_text.setPos(0, np.max(np.maximum(self.rec.behavior,
                                                               self.rec.stimulus)) + 3)  # move because stimulus and behavior can vary greatly in magnitude (dependent on physical unit)
            else:  # only behavior
                self.behavior_pw.setYRange(np.min(self.rec.behavior), np.max(self.rec.behavior) + 6)  # lay out
                self.beh_stim_text = pg.TextItem(text='Behavior (yellow)', color=[255, 255, 255], anchor=(0, 0))
                self.beh_stim_text.setPos(0, np.max(self.rec.behavior) + 3)

        elif 'stimulus' in self.rec.available_data:  # only stimulus
            self.stimulus_plot.addItem(pg.PlotDataItem(x=self.rec.times, y=self.rec.stimulus, pen='r'))
            self.behavior_pw.setYRange(np.min(self.rec.stimulus), np.max(self.rec.stimulus) + 6)  # lay out
            self.beh_stim_text = pg.TextItem(text='Stimulus (red)', color=[255, 255, 255], anchor=(0, 0))
            self.beh_stim_text.setPos(0, np.max(self.rec.stimulus) + 3)

        else:  # no stimulus or behavior
            self.beh_stim_text = pg.TextItem(text='No behavior or stimulus to display',
                                             color=[255, 255, 255], anchor=(0, 0))
            self.beh_stim_text.setPos(0, 1)
        self.behavior_pw.setXRange(0, self.rec.times[-1])
        self.behavior_pw.addItem(self.beh_stim_text)  # add now instead of at the beginning
        self.mean_selection_plot.clear()  # clear mean trace (if there is one left from preivous data set)

        if 'spikes' in self.rec.available_data:
            # define like this so it does not crash when SampleData without spikes is loaded
            # Add 'spikes' option to dropdown menu and data_names dictionary
            # so if you do not load spikes, it does not appear as option
            self.data_sel_cb.addItem('spikes')  # add option to dropdown menu
            self.datasets_scrollbar.setRange(0, 1)
            self.datasets_scrollbar.setValue(0)  # still df
            self.i_dataset += 1  # increase index
            self.data_names[self.i_dataset] = 'spikes'  # add dataname to overview of data names

        if 'labels' in self.rec.available_data:
            self._labels_available = True
            self.region_check_cb.setEnabled(True)
            self.neuron_check_cb.setEnabled(True)
            self.region_sel_cb.setEnabled(True)
            self.labelled_cb.setEnabled(True)
            self.color_region_action.setEnabled(True)
        else:  # no labels in hdf5 file
            self._logger.warning('No region labels Found at in hdf5 file')
            print('No region labels Found in hdf5 file')
            self._labels_available = False
            self.region_check_cb.setEnabled(False)
            self.neuron_check_cb.setEnabled(False)
            self.region_sel_cb.setEnabled(False)
            self.labelled_cb.setEnabled(False)

        if 'structural_data' in self.rec.available_data:
            self.structural_data = self.rec.structural_data
            self.structural_data_sb.setMaximum(self.structural_data.shape[0]-1)
            self.structural_data_check_cb.setEnabled(True)

        if 'structural_data' not in self.rec.available_data:
            self.structural_data_check_cb.setEnabled(False)

        # Open association between labels and indices
        if self._labels_available:
            with open(os.path.join(fishualizer_folder, 'Content/RegionLabels.txt')) as F2:  # load in labelnames
                content = F2.readlines()
            self.labelnames = [x.strip() for x in content]
            self.region_sel_cb.blockSignals(True)  # block signals while adding new items to the qcombobox in regions_add() because this will call region_change() everytime
            self.region_sel_cb.clear()  # clear previous entries (if any)
            self.regions_add()
            self.region_sel_cb.blockSignals(False)
            self._labelled_inds = np.where(np.sum(self.rec.labels, axis=1) > 0)

        if '_additional_static_data' in self.rec._data.keys():  # if the h5 data set contained non-recognised static data sets, these are stored here
            for s_data_name, s_data_set in self.rec._data['_additional_static_data'].items():  # unpack and add to data (and fishualizer menu)
                self.add_static(c_data=s_data_set, dataname=s_data_name)
                self._logger.debug(f'Static dataset {s_data_name} added.')

        self.data_sel_cb.setCurrentText('df')  # default to df activity
        self._logger.info("Data loading done.")
        self.statusBar().showMessage('Data loading is done. Creating the plot, please wait')
        print('... done.')

        # Remove old plot
        if self.Plot3D is not None:
            self.view3D.removeItem(self.Plot3D)
            self.view2D.removeItem(self.Plot3D)
            self.Plot3D = None

        # Assign Times
        self._max_time = len(self.rec.times) - 1

        if np.mean(self.rec.df[:, np.unique(np.sort(np.random.choice(self.rec.n_times, 100)))]) > 1:  # mean of some random time points
            """
            T: We talked about not calcuting full statistics (e.g. percentile)
            because it takes long to calculate. Now the mean of 5 time points,
            all neurons is used to disambiguate between calcium and deconvolved
            """
            self.disp_start['df'] = 1
            self.disp_stop['df'] = 1.5
        else:
            self.disp_start['df'] = 0
            self.disp_stop['df'] = 1.5

        self.limit_color_start_sb.setValue(self.disp_start[self.data_set])
        self.limit_color_stop_sb.setValue(self.disp_stop[self.data_set])

        # add other keys as well so they can be plotted as well (needs finetuning):
        if 'spikes' in self.rec.available_data:
            self.disp_start['spikes'] = 0
            self.disp_stop['spikes'] = 0.4
            self.disp_change['spikes'] = False
            self.swap_color_data['spikes'] = False

        self._data_loaded = 1
        self.hide_action_min.setEnabled(True)
        self.hide_action_max.setEnabled(True)
        self.reset_view_action.setEnabled(True)
        self.color_cluster_action.setEnabled(True)

        # Initialize Random Shifts in Z to visually Fill plot
        self.z_coords_unique = np.unique(self.rec.sel_frame[:, 2])
        dz = np.mean(np.diff(self.z_coords_unique))
        self.z_random_shifts = np.random.uniform(-0.95 * dz / 2, 0.95 * dz / 2, self.rec.sel_frame.shape[0])

        self.reset_scrollbars()
        self.neurons_set()
        self.plot_trace()
        self.center_view(update=False)

        # Set maximum of frame_sb spinbox (defaults otherwise to 99)
        self.frame_spb.setMaximum(self.rec.n_times)  # suffices while Sampling Frequency > 1Hz
        self.viewyz(self.view2D)  # switch 2D plot to side view
        self.set_perspective_2d(view=self.view2D)  # Change to 2D perspective for view2D
        self.set_window_sizes(window_sizes=[600, 300, 40])  # change sizes of view3D : view2D : lineplot (i.e. top to bottom)

        self.alpha_menu.setEnabled(True)
        # self.recomputeBackdrops()
        self.statusBar().showMessage("Plot created.")

    def choose_coordinates(self, bool_update_plot=True):
        """
        Choose coordinate system.

        Parameters:
        --------------
        bool_update_plot: boolean
            if True, self.update_plot() is called (preferred when changed in UI), if False it is not (preferred during data loading)
        """
        recognized_coord_names = [k for k in self.rec.available_data if 'coord' in k]

        if len(recognized_coord_names) > 0:  # check if multiple data sets are present named "something something coord"
            choices = [self.rec.data_names[k] for k in self.rec.available_data if
                       ('coord' in k) and (k != 'zbrainatlas_coordinates')]  # possible input names containing 'coord'
            if len(choices) > 1:  #multiple coordinate systems, let user choose
                choice_input_name, ok_pressed = QtWidgets.QInputDialog.getItem(self, "Coordinate choice",
                                                                               "What coordinate system should be used?",
                                                                               choices, 0, False)
            elif len(choices) == 1:  # only 1 coordinate system
                choice_input_name = choices[0]
            for default_name, input_name in self.rec.data_names.items():  # loop through mappings to find corresponding default name
                if input_name == choice_input_name:
                    self.rec.sel_frame = self.rec.datasets[default_name]
                    self.reset_scrollbars()
                    self._logger.info(f'Using {choice_input_name} data as coordinate system.')
                    self.statusBar().showMessage(f'Using {choice_input_name} data as coordinate system.')

                    break
            if bool_update_plot:
                self.update_plot()
        else:
            self._logger.error('No coordinates present in the data.')  # in principle this should not occur because utilities.open_h5_data() checks this as well

    def set_limit_slider_pos(self):
        """Set x,y,z sliders to default values of [0,0,0] - [99,99,30]"""
        self._limit_slider_start_pos = [0, 0, 0]
        self._limit_slider_stop_pos = [99, 99, 30]

    def limitx(self):
        """Change axis interval slider to x axis."""
        self._dimsel = 0
        self.x_action.setChecked(True)
        self.set_limit_gui()

    def limity(self):
        """Change axis interval slider to y axis."""
        self._dimsel = 1
        self.y_action.setChecked(True)
        self.set_limit_gui()

    def limitz(self):
        """Change axis interval slider to z axis."""
        self._dimsel = 2
        self.z_action.setChecked(True)
        self.set_limit_gui()

    def viewyz(self,view=None):
        """Change camera position to yz plane."""
        if view is None:
            view = self.last_viewbox[0]
        view.setCameraPosition(elevation=0, azimuth=0)

    def viewyz_left(self, view=None):
        """Change camera position to yz plane."""
        if view is None:
            view = self.last_viewbox[0]
        view.setCameraPosition(elevation=0, azimuth=180)

    def viewxz(self,view=None):
        """Change camera position to xz plane."""
        if view is None:
            view = self.last_viewbox[0]
        view.setCameraPosition(elevation=0, azimuth=90)

    def viewxy(self,view=None):
        """Change camera position to xy plane."""
        if view is None:
            view = self.last_viewbox[0]
        view.setCameraPosition(elevation=90, azimuth=0)

    def set_limit_gui(self):
        """Function that configures the axis interval scrollbar of dimension self._dimsel."""
        self.limit_start_le.setText(str(np.around(self._limit_start[self._dimsel], 3)))  # set limit start line editor to minimum
        self.limit_start_sb.setRange(0, self._limit_slider_nsteps[self._dimsel] - 1)  # set number of intervals
        tmp = np.argmin(np.absolute(self._limit_start[self._dimsel] - self._limit_slider_values[self._dimsel]))
        self._limit_slider_start_pos[self._dimsel] = tmp  # save start value
        self._logger.debug('Slider Start Pos : {}'.format(self._limit_slider_start_pos[self._dimsel]))
        # print('Slider Start Pos : {}'.format(self._limit_slider_start_pos[self._dimsel]))
        self.limit_start_sb.setValue(self._limit_slider_start_pos[self._dimsel])  # set start value

        self.limit_stop_le.setText(str(np.around(self._limit_stop[self._dimsel], 3)))
        self.limit_stop_sb.setRange(0, self._limit_slider_nsteps[self._dimsel] - 1)
        tmp = np.argmin(np.absolute(self._limit_stop[self._dimsel] - self._limit_slider_values[self._dimsel]))
        self._limit_slider_stop_pos[self._dimsel] = tmp
        self._logger.debug('Slider Stop Pos : {}'.format(self._limit_slider_stop_pos[self._dimsel]))
        # print('Slider Stop Pos : {}'.format(self._limit_slider_stop_pos[self._dimsel]))
        self.limit_stop_sb.setValue(self._limit_slider_stop_pos[self._dimsel])

    def limit_start_scroll(self):
        """This function is called if the scroll bar self.limit_start_sb is moved.

        It puts the (new) current value of the scroll bar in self._limit_start, matches
        the line text in self.limit_start_le and calls neurons_set() which takes care of further updating.
        """
        if self._data_loaded:
            ind = self.limit_start_sb.value()
            self._limit_start[self._dimsel] = self._limit_slider_values[self._dimsel][ind]
            self.limit_start_le.setText(str(np.around(self._limit_start[self._dimsel], 3)))
            self.neurons_set()

    def limit_stop_scroll(self):
        """This function is called if the scroll bar self.limit_stop_sb is moved.

        It puts the (new) current value of the scroll bar in self._limit_stop, matches
        the line text in self.limit_stop_le and calls neurons_set() which takes care of further updating.
        """
        if self._data_loaded:
            ind = self.limit_stop_sb.value()
            self._limit_stop[self._dimsel] = self._limit_slider_values[self._dimsel][ind]
            self.limit_stop_le.setText(str(np.around(self._limit_stop[self._dimsel], 3)))
            self.neurons_set()

    def limit_start(self):
        """This function is called if the line edit self.limit_start_le has been edited.

        It puts the (new) current value of the line text in self._limit_start and
        calls neurons_set() which takes care of further updating.
        """
        if self.limit_start_le.isModified():
            self._limit_start[self._dimsel] = float(self.limit_start_le.text())
            self.neurons_set()

    def limit_stop(self):
        """This function is called if the line edit self.limit_stop_le has been edited.

        It puts the (new) current value of the line text in self._limit_stop and
        calls neurons_set() which takes care of further updating.
        """
        if self.limit_stop_le.isModified():
            self._limit_stop[self._dimsel] = float(self.limit_stop_le.text())
            self.neurons_set()

    def plot_time_changed(self, marker):
        """This function updates self.current_time if the time in one of the plots
        has been changed by the user.

        Parameter:
        ----------
            marker: pg.InfiniteLine class
                The vertical time line in the plots are of this instance.
                Their attribute .value() is called to get the new time.
        """
        self.current_time = int(marker.value())

    def neurons_set(self):
        """This functions selects the neurons currently visible with the xyz limit bars
        and hides the others.

        It loops through the three dimensions xyz and finds all coords that are
        between the self._limit_start and self._limit_stop. Subsequently self._selected_inds
        is updated and update_plot() is called, which only plots neurons in self._selected_inds
        """
        if self._data_loaded:
            coords = self.rec.sel_frame
            coord = self.rec.sel_frame[:, 0]
            # Hack to get a boolean variable of matches shape
            remain_ind = np.ones(coord.shape, dtype = bool)
            for cInd in [0, 1, 2]:
                scaled_coord = coords[:, cInd]
                scaling_factor = self.rec.spatial_scale_factor
                inv_scaling_factor = 1 / scaling_factor[cInd]  # scale back to raw coordinates (to avoid sign changes which could be confusing)
                coord = inv_scaling_factor * scaled_coord
                coord_ind = np.logical_and(coord >= self._limit_start[cInd], coord <= self._limit_stop[cInd])
                remain_ind = np.logical_and(remain_ind, coord_ind)

            cells = np.where(remain_ind > 0)[0]  # indices of cells that remain within all 3 axis intervals
            self.filter_dict['axis_filter'] = cells

            if self._labelled_only:
                self.filter_dict['labelled_only'] = self._labelled_inds[0]
            else:
                self.filter_dict['labelled_only'] = np.arange(self.rec.n_cells)

            self.selected_inds_filters()

    def select_cell(self, pos_click=None, alt=0):
        """Visually select a neuron based on the camera position and the mouse click position.

        Parameters:
        ------------
            pos_click:
                clicked mouse position
            alt: 0 or 1
                Whether alt is pressed, to know whether to move the alt=0 or alt=1 selection ball
        """

        if self._data_loaded:
            # get center of current view
            centerQT = self.view3D.opts['center']
            center = np.array([centerQT.x(), centerQT.y(), centerQT.z()])

            # get vector to camera
            vectorQT = self.view3D.cameraPosition()
            vector_camera = np.array([vectorQT.x(), vectorQT.y(), vectorQT.z()])

            # relate points and camera vector to camera position
            coords_centered = self.rec.sel_frame[self._selected_inds] - vector_camera
            vector_center = center - vector_camera

            if pos_click is not None:
                # for mouse click on, find the deviation of the
                fov = self.view3D.opts['fov']  # field of view in degrees
                viewsize = self.view3D.size()  # size of the display in pixels

                deg_per_pixel = fov / viewsize.width()
                angle = np.zeros((2, 1))
                angle[0] = deg_per_pixel * (pos_click.x() - viewsize.width() / 2) / 360 * 2 * np.pi
                angle[1] = -deg_per_pixel * (pos_click.y() - viewsize.height() / 2) / 360 * 2 * np.pi

                # rotate a vector in the default coordinate system accordingly
                rotated_vector = [1.1 * np.sin(angle[0]), 1.1 * np.sin(angle[1]), 1]

                # apply this vector to the cameras coordinate system
                BV0 = [vector_center[1], -vector_center[0], 0]
                BV0 = BV0 / np.linalg.norm(BV0)
                BV2 = vector_center / np.linalg.norm(vector_center)
                BV1 = np.cross(BV0, BV2)
                BV1 = BV1 / np.linalg.norm(BV1)
                vector_target = rotated_vector[0] * BV0 + rotated_vector[1] * BV1 + rotated_vector[2] * BV2
            else:
                vector_target = vector_center

            # Project the coordinates onto the view axis and find closest point in the orthogonal direction
            scalar_proj = np.dot(coords_centered, vector_target)  # Nx1
            scalar_proj = np.divide(scalar_proj, np.linalg.norm(vector_target) * np.linalg.norm(vector_target))
            aligned_vectors = np.outer(vector_target, np.transpose(scalar_proj))
            orthogonal_vectors = coords_centered - np.transpose(aligned_vectors)  # 3xN
            distances = np.linalg.norm(orthogonal_vectors, axis=1)

            # compute weights based on current distribution of values. Weights are supposed to aid the selection, but preferring relatively large activities
            current_intensities = np.sum(self.colors[self._selected_inds], 1) / 3
            weights = 1 - 1 / (1 + np.exp(current_intensities - 0.5))
            weights[distances > 0.01] = 1  # consider only weights within 10 microns
            index = np.argmin(np.multiply(weights, distances))
            index = self._selected_inds[index]
            self.current_selected_cell = index
            self.show_cell(index, alt)
            coords_current_str = str(np.round(self.rec.sel_frame[index, :], 3))
            if self._copy_clipboard:
                Qt.QApplication.clipboard().setText(coords_current_str)

    def show_cell(self, index=-1, alt=0):
        """Show white ball around selected cell and plot its trace

        Parameters:
            index: int, default=-1
                cell index
            alt, 0 or 1, default=0
                parameter indicating whether the alt=0 or alt=1 cell is considered
        """
        if index is False:
            index = int(self.select_cell_dialog.cellnumber.text())
            self.current_selected_cell = index

        # Indicate the selected neuron, by moving the selector onto it.
        self.select_3D.setData(pos=self.rec.sel_frame[index])
        self.select_3D.show()
        self._logger.info('Selected Neuron : {}'.format(index))
        print('Selected Neuron : {}'.format(index))
        c_region = 'not assigned'
        if self._labels_available:
            c_region_index = scipy.sparse.find(self.rec.labels[index, :])
            if c_region_index[1].size > 0 and self.labelnames is not None:
                c_region = [self.labelnames[x + 1] for x in c_region_index[1]]  # +1 because it starts with main divisions
        print(f'Connected to region(s): {c_region}')
        self.plot_trace(index, alt)
        # in case the connections are displayed
        if self.data_set == 'input' or self.data_set == 'output':
            self.update_plot()

    def recompute_backdrops(self):
        """
        Compute common backdrops, like average activity, additional ones can be loaded from outside analysis
        """
        # self.backdrops['average'] = np.mean(self._data['values'])
        pass

    def resize_scrollbar(self):
        """
        Resize the scroll bar according to the number of time points the data have
        """
        self.frame_sb.setMinimum(0)
        self.frame_sb.setMaximum(max(0, self._nb_times - 1))

    @staticmethod
    def frame_to_array(frame):
        frame_bits = frame.bits()
        str_frame = frame_bits.asstring(frame.width() * frame.height() * frame.depth() // 8)
        frame_arr = np.fromstring(str_frame, dtype=np.uint8).reshape(
            (frame.height(), frame.width(), frame.depth() // 8))
        frame_arr = frame_arr[:, :, (2, 1, 0, 3)]
        return frame_arr

    def grab_plot(self, filename=None, include_view3D=True, include_view2D=True):
        """Make a screenshot of self.view3D and save via QFileDialog."""
        if include_view3D:
            frameTop = self.view3D.grabFrameBuffer()
            frameTop = frameTop.convertToFormat(QtGui.QImage.Format_ARGB32)
            frameTop_arr = self.frame_to_array(frameTop)
        if include_view2D:
            frameBottom = self.view2D.grabFrameBuffer()
            frameBottom = frameBottom.convertToFormat(QtGui.QImage.Format_ARGB32)
            frameBottom_arr = self.frame_to_array(frameBottom)
        if include_view2D and include_view3D:
            frame_arr = np.concatenate((frameTop_arr,frameBottom_arr), axis=0)
        elif include_view2D and not include_view3D:
            frame_arr = frameBottom_arr
        elif include_view3D and not include_view2D:
            frame_arr = frameTop_arr

        if filename is None or type(filename) != str:
            im_path, _filter = QtWidgets.QFileDialog.getSaveFileName(self, "Saving figure to...",
                                                                        self.export_dir, 'Image (*.png)')
        else:
            im_path = filename
        self.export_dir = im_path
        if im_path != '':
            if not im_path.endswith('.png'):
                im_path += '.png'
            imwrite(im_path, frame_arr)

    def selection_change_screenshot(self, foldername='', prefix='screenshot_',
                                    dataset='something', sleep_time=0.3):
        '''Change selection to dataset, and take a screenshot saved under
        foldername + prefix + dataset .png '''
        assert dataset in list(self.data_names.values())
        self.selection_change(selection=dataset)  # change selection
        pg.QtGui.QApplication.processEvents()  # update graphics so that screenshot can be made
        time.sleep(sleep_time) # sleep for some time to process the above before taking screenshot
        im_name = foldername + prefix + dataset + '.png'
        self.grab_plot(filename=im_name)  # take screenshot

    def video_export(self, event=None):
        self._export_video = not self._export_video
        if self._export_video:
            cwd = os.getcwd()
            video_path, _filter = QtWidgets.QFileDialog.getSaveFileName(self, "Saving video to...",
                                                                        cwd, 'Video (*.mp4 )')
            if video_path != '':
                # Saving
                if not video_path.endswith('mp4'):
                    video_path += '.mp4'
                self.video_writer = get_writer(video_path, mode='I', quality=10)
                self._export_timer.start()
            else:
                self._export_video = False
                self.video_action.setChecked(False)
        else:
            self._export_timer.stop()
            self.video_writer.close()

    def video_recording(self):
        """Save frame in video_writer."""
        frame = self.view3D.grabFrameBuffer()
        frame = frame.convertToFormat(QtGui.QImage.Format_RGB32)
        frame = frame.scaledToWidth(1080)
        frame = frame.copy(0, 0, 1080, 720)
        frame_arr = self.frame_to_array(frame)
        self.video_writer.append_data(frame_arr)

    def play(self, event=None):
        """
        Callback handling the play button, to pley the movie

        Parameters
        ----------
        event
        """
        # Event can be none if triggered through the keyboard shorcut
        # In that case we click the button and then check its state to know if we start or stop the animation
        if event is None:
            self.play_btn.click()
            event = self.play_btn.isChecked()
        if event:
            # Starting to play by starting the Timer
            self._playtimer.start()
            self.c_pbc = 0
        else:
            # Stopping
            self._playtimer.stop()
            self.statusBar().showMessage('')

    def save_class(self, filename=None):
        '''Function that saves class (ie all the attributes) to a text file'''
        if filename is None:
            filename = 'Fishualizer_class'
        dt = datetime.datetime.now()
        timestamp = str(dt.date()) + '-' + str(dt.hour).zfill(2) + str(dt.minute).zfill(2)  # add time stamp to this state
        file_ts = filename + '_' + timestamp + '.txt'
        attr_dict = vars(self)  # save all attributes as dict (long arrays are shortened as if they were printed)
        with open(file_ts, 'w') as f:
            print(attr_dict, file=f)  # save file

        return timestamp

    def tidy_up_viewboxes(self, distance=180, somata=False, reverse_x=False,
                          draw_outlines=True, draw_crosses=True, draw_scale=True):

        ## get rid of grids, selection, axes, set perspectives
        if self.hide_grid is False:
            self.toggle_hidegrid()
        self.switch_topside(add_scalebar=False)

        ## zbrain regions, draw outlines in 3D and 2D
        if self._use_zbrainatlas_coords is False:
            self.change_region_view()
        if draw_outlines:
            self.draw_xy_outline(view=self.view3D)
            self.draw_yz_outline(view=self.view2D)

        ## select neurons
        if self._labelled_only is False:
            self.labelled_cb.setChecked(True)  # only used labelled neurons
        if reverse_x:
            self.call_reverse_x_coords()

        ## visuals
        if self.show_somas is not somata:
            self.toggle_somas()

        # add cross and scale bar, add , sb_obj=1 to add new ones instead of replace
        if draw_crosses:
            self.add_scale_bar(start=[0.08, 0.12, 0], end=[0.08, 0.18, 0], bar_width=3, view=self.view3D, sb_obj=1)  # horizontal, view3D
            self.add_scale_bar(start=[0.05, 0.15, 0], end=[0.11, 0.15, 0], bar_width=3, view=self.view3D, sb_obj=1)  # vertical, view3D

            self.add_scale_bar(start=[0, 0.12, 0.28], end=[0, 0.18, 0.28], bar_width=3, view=self.view2D, sb_obj=1)  # horizontal, view2D
            self.add_scale_bar(start=[0, 0.15, 0.25], end=[0, 0.15, 0.31], bar_width=3, view=self.view2D, sb_obj=1)  # vertical, view2D

        if draw_scale:
            self.add_scale_bar(start=[0, 1, 0.05], end=[0, 1.1, 0.05], bar_width=5, view=self.view2D, sb_obj=1)  # scale bar

        self.v_splitter.setSizes([200, 155, 0])  # customise window sizes
        for vb in [self.view2D, self.view3D]:
            vb.opts['distance'] = distance  # zoom in/out
            vb.update()

    def hu_view_mode(self):
        if self._data_loaded:
            self.tidy_up_viewboxes(somata=False, reverse_x=False,
                                    draw_outlines=True, draw_crosses=False, draw_scale=False)

            self.toggle_manual_auto_limit_cb.setCurrentIndex(1)  # switch to manual
            self._cmap_name = 'turqoiseblackred_30'  # set new colormap & repaint
            self.make_cmap()
            self.rate_sel_cp.repaint()

            self.disp_start[self.data_set] = -0.3
            self.disp_stop[self.data_set] = 0.3
            self.limit_color_start_sb.setValue(self.disp_start[self.data_set])
            self.limit_color_stop_sb.setValue(self.disp_stop[self.data_set])

    def activity_view_mode(self):
        if self._data_loaded:
            self.data_sel_cb.setCurrentText('df')

            self.toggle_manual_auto_limit_cb.setCurrentIndex(0)  # switch to auto
            self._cmap_name = 'inferno'  # set new colormap & repaint
            self.make_cmap()
            self.rate_sel_cp.repaint()

            self.disp_start[self.data_set] = 0
            self.disp_stop[self.data_set] = 1.5
            self.limit_color_start_sb.setValue(self.disp_start[self.data_set])
            self.limit_color_stop_sb.setValue(self.disp_stop[self.data_set])

    def select_next_dataset(self):
        curr_sel = self.data_sel_cb.currentIndex()
        max_index = len(self.data_names) - 1
        if curr_sel < max_index:
            self.data_sel_cb.setCurrentIndex(curr_sel + 1)
        else:
            self._logger.warning('already at max data set')

    def select_previous_dataset(self):
        curr_sel = self.data_sel_cb.currentIndex()
        if curr_sel > 0:  # the minimum index
            self.data_sel_cb.setCurrentIndex(curr_sel - 1)
        else:
            self._logger.warning('already at min data set')

    def select_index_dataset(self, curr_sel=0):
        curr_sel = int(curr_sel)
        max_index = len(self.data_names) - 1
        if curr_sel >= 0 and curr_sel <= max_index:  # the minimum index
            self.data_sel_cb.setCurrentIndex(curr_sel)
        else:
            self._logger.warning('already at min or max  data set')

    def select_index_dataset_scrollbar(self):
        ind = self.datasets_scrollbar.value()
        self.select_index_dataset(curr_sel=ind)

    def save_all_static_datasets(self, foldername=None, base_ds_name='hu_',
                                 index_start=0, index_stop=200, index_zfill=3):
        if foldername is None:
            print('Define folder!')
            return None
        foldername = foldername.rstrip('/')

        timestamp = self.save_class(filename=foldername + '/Fishualizer_class')  # save current state

        for it in range(index_start, index_stop):  # loop through ds
            str_ind = str(it).zfill(index_zfill)  # convert to str format
            ds_name = base_ds_name + str_ind
            self.selection_change_screenshot(foldername=foldername, prefix='/screenshot_',
                                             dataset=ds_name, sleep_time=0.3)

    def switch_rotation(self):
        """Toggle rotation during play animation"""
        self._rotate_on_play = not self._rotate_on_play
        self.check_turn_off_animation()  # if both are off, stop animation if need

    def toggle_dynamics(self):
        '''Toggle dynamics during play animation'''
        self._dynamics_on_play = not self._dynamics_on_play
        self.check_turn_off_animation()  # if both are off, stop animation if needed

    def toggle_animation(self):
        '''Toggle animation  '''
        self._animate = not self._animate
        self.save_checking(action=self.anim_action, check=self._animate)
        self.play(event=self._animate)

    def check_turn_off_animation(self):
        '''If both dynamics and rotation are turned off, stop animation'''
        if self._rotate_on_play is False and self._dynamics_on_play is False:
            self._animate = False
            self.save_checking(action=self.anim_action, check=self._animate)
            self.play(event=False)

    def set_labelled(self):
        self._labelled_only = self.labelled_cb.isChecked()
        if self._labelled_only:
            self._logger.info('Only showing labelled neurons')
        else:
            self._logger.info('Showing labelled and unlabelled neurons')
        self.neurons_set()
        self.update_plot()

    def animate(self):
        """
        Callback function called by the animation Timer
        Dependent on the setting self.animate_mode either a normal rotation is
        performed (self.animate_mode=='default' and if self._rotate_on_play==1) or
        a preprogrammed (in this function) camera trajectory is executed (self.animate_mode=='fancy_movie')
        """
        if self.animate_mode == 'default':
            if self._rotate_on_play == 1:  # azimuth rotation only
                self.view3D.orbit(0, self.default_animate_azimuth_step)  # increase azimuth

        elif self.animate_mode == 'fancy_movie':  # customize movement
            magnification = 0.98
            frame_offset = 1000
            if (self.current_frame  > 100 + frame_offset) & (self.current_frame <= 160 + frame_offset):  # decrease elevation
                self.view3D.orbit(-0.8, -1)
            elif (self.current_frame > 200 + frame_offset) & (self.current_frame <= 225 + frame_offset ):  # zoom in
                self.view3D.opts['distance'] = magnification * self.view3D.opts['distance']
            elif (self.current_frame > 300 + frame_offset) & (self.current_frame <= 360 + frame_offset):  # increase elevation
                self.view3D.orbit(0.8, 1)
            elif (self.current_frame > 400 + frame_offset) & (self.current_frame <= 425 + frame_offset):  # zoom out
                self.view3D.opts['distance'] = (1 / magnification) * self.view3D.opts['distance']

        if self._dynamics_on_play:
            if self.current_frame < self.nb_times - 1:
                self.current_frame += 1  # Increment the current frame number, the setter take care of the rest
            else:
                self.current_frame = 0  # Reset current frame once we reach the end
        # pbc = '-\|/'
        pbc = '               ><(((°> '
        self.c_pbc = (self.c_pbc + 1) % len(pbc)
        self.statusBar().showMessage(f'Playing {pbc[-self.c_pbc:]}')

    def selection_change(self, selection):
        """
        Callback method for the combobox allowing to select the displayed data
        It can also be used programmatically to reset the selection

        Parameters
        ----------
        selection: int or str
            Either the index or the name of the data set to select

        """
        if isinstance(selection, str):  # evoked through code
            data_set = selection
        else:  # evoked through GUI
            data_set = self.data_names[selection]

        self.data_sel_cb.blockSignals(True)
        self.data_sel_cb.setCurrentText(data_set)
        self.data_sel_cb.blockSignals(False)

        self.datasets_scrollbar.blockSignals(True)
        self.datasets_scrollbar.setValue(selection)
        self.datasets_scrollbar.blockSignals(False)

        self._last_selection[1] = self._last_selection[0]
        self._last_selection[0] = data_set

        if self.fix_colorscale:  # if one wants to fix color scales, assign to previous color scales (which still allows tuning)
            self.disp_start[data_set] = self.disp_start[self.data_set]
            self.disp_stop[data_set] = self.disp_stop[self.data_set]
            self.disp_change[data_set] = True  # later, if fix_colorscale == False, it will be recomputed.

        elif not self.fix_colorscale and self.disp_change[data_set]:
            if data_set == 'df':
                self.disp_start[data_set] = 0
                self.disp_stop[data_set] = 1.5
            elif data_set == 'spikes':
                self.disp_start[data_set] = 0
                self.disp_stop[data_set] = 0.4
            else:
                self.disp_start[data_set] = np.percentile(getattr(self.rec, data_set), 1)
                self.disp_stop[data_set] = np.percentile(getattr(self.rec, data_set), 99)
            self.disp_change[data_set] = False

        self.data_set = data_set
        self.frame_sb.setEnabled(True)
        if data_set == 'df':
            # self.disp_start[data_set] = 1.1 # NB: this would override the settings in load_data() !
            # self.disp_stop[data_set] = 1.5  # change default start color for visibility
            self.alpha = 0.6  # change default alpha, not as important as start_color, but a little
        elif data_set == 'spikes':
            # self.disp_start[data_set] = 0
            # self.disp_stop[data_set] = 0.5
            self.alpha = 0.8
        else:  # Static data sets
            self.frame_sb.setEnabled(False)
            self.alpha = 0.8

        # Update color limits
        self.limit_color_start_sb.setValue(self.disp_start[data_set])
        self.limit_color_stop_sb.setValue(self.disp_stop[data_set])

    def selection_toggle(self):
        self.data_sel_cb.setCurrentText(self._last_selection[1])

    def export_selected(self):
        """Export the data corresponding the current selection to .npz file"""
        c_ind = self._selected_inds
        c_data = self.rec.df[c_ind]  #TODO: Should we change this to current data set (self.data_set)?
        self._logger.info('Exporting Cell Numbers and Calcium Data to ExportSelectedData.npz')
        print('Exporting Cell Numbers and Calcium Data to ExportSelectedData.npz')
        np.savez('ExportSelectedData.npz', c_ind, c_data)
        self.statusBar().showMessage('Exported Cell Numbers and Calcium Data to ExportSelectedData.npz')

    def leaving(self):
        """If the Fishualizer is closed, call Viewer.leaving() to close the kernel
        and then stop video exporting if still recording"""
        super().leaving()
        if self._export_video:
            self.video_writer.close()
            self._export_timer.stop()

    @property
    def data_path(self):
        return self._data_path

    @data_path.setter
    def data_path(self, value):
        """
        When the data path changes, open the corresponding data file

        Parameters
        ----------
        value: str
            Path to the data as selected from the dialog box
        """
        self._data_path = value
        self.load_data()

    @property
    def rec(self):
        return self._rec

    @rec.setter
    def rec(self, value):
        self._rec = value
        self.nb_times = self.rec.n_times

    @property
    def nb_times(self):
        return self._nb_times

    @nb_times.setter
    def nb_times(self, value):
        """
        When the number of time points in the data changes, resize the scrollbar accordingly

        Parameters
        ----------
        value: int
            Number of time points in the data
        """
        self._nb_times = value
        self.resize_scrollbar()

    def get_current_values(self):
        """ Return the values of all neurons of the current data set and frame

        Returns:
        ---------
            current_values: np.array
                The numerical values of the current data set and current frame
        """
        if self.data_set == 'df' or self.data_set == 'spikes':
            current_values = np.float32(self.rec.datasets[self.data_set][:, self.current_frame])  # Select relevant data
        elif self.data_set == 'input' or self.data_set == 'output':
            if self.data_set == 'input':
                tmp, inds, weights = scipy.sparse.find(self.rec.input[self.current_selected_cell, :])
            else:
                inds, tmp, weights = scipy.sparse.find(self.rec.output[:, self.current_selected_cell])
            n_cells = self.rec.n_cells
            current_values = np.zeros(n_cells)
            self._logger.debug(f'Number of Connections {len(inds)}')
            if len(inds) != 0:
                current_values[inds] = weights  # Assign connection values for estimated subset (spiking)
        else:
            current_values = self.rec.__getattribute__(self.data_set)
        return current_values

    def update_plot(self):
        """ Update the plot, this function is called if something in the data has changed
        (e.g. different frame or different data set)

        This function clips the current values to the interval [self.disp_start,
        self.disp_stop]. Values are mapped to colors via self.cmap. If no instance
        exists in self.Plot3D, initialization features are created (e.g. camera position,
        GLScatterPlotItem, reference axes)."""
        global disp_start
        if self._data_loaded:
            # DATA SELECTION
            super().update_plot()
            current_values = self.get_current_values().copy()

            coords = self.rec.sel_frame # get current coordinates
            if self.z_random_shift:
                coords[:, 2] = coords[:, 2] + self.z_random_shifts

            # COLORMAPPING
            disp_start = self.disp_start[self.data_set]
            disp_stop = self.disp_stop[self.data_set]

            # Handle NaNs
            idx_nan = np.isnan(current_values)
            current_values[idx_nan] = disp_start

            # Assign Colors via color map
            c_colors = self.cmap(current_values, vmin=disp_start, vmax=disp_stop)

            # Replace NaNs after color selections
            if self.hide_nans:
                c_colors[idx_nan,3] = 0 # make all nans transparent / invisible
            else:
                c_colors[idx_nan,0:3] = [0.1, 0.1, 0.1] # show all nan's as translucent gray

            # Show Neuron Location as gray backdrop for low activity neurons
            if self.show_somas:
                gray = np.array([0.5, 0.5, 0.5, c_colors[0, 3]])
                current_values_norm = (current_values - disp_start) / (disp_stop - disp_start)
                current_values_norm[current_values_norm < 0] = 0
                current_values_norm[current_values_norm > 1] = 1
                c_colors = (1 - current_values_norm[:, None]) * gray.T[None, :] + current_values_norm[:, None] * c_colors

            self.rate_sel_cp.cmap = self.cmap  # update color scale bar
            self.colors = c_colors # recall colors

            self.update_behavior_plot()  # update behavior plot, because scaling can be off after zoom or new data addition

            # If no 3D plot ever created, make one and plot
            if self.Plot3D is None or self._selection_updated:
                if self.Plot3D is None:
                    self.Plot3D = gl.GLScatterPlotItem(pos=coords[self._selected_inds],
                                                       color=c_colors[self._selected_inds],size=self.dot_size)
                    self.switch_alpha_mode('maximum')  # Set maximum intensity projection as the default
                    for vb in [self.view2D, self.view3D]:
                        vb.setCameraPosition(distance=1)
                        vb.addItem(self.Plot3D)
                    self.update_coordinate_system()
                    #
                    ref_pos = np.array([0, 0, 0])
                    self.select_3D = gl.GLScatterPlotItem(pos=ref_pos, color=[1, 1, 1, 0.5], size=20)
                    for vb in [self.view2D, self.view3D]:  # if only added to 1 vb, it hides all other elements except Plot3D to the other vb
                        vb.addItem(self.select_3D)

                else:
                    self.Plot3D.setData(pos=coords[self._selected_inds], color=c_colors[self._selected_inds])
                    self._selection_updated = False
                    self.view2D.update()

            else:
                # reuse previous plot, just change the data (positions and/or colors)
                self.Plot3D.setData(pos=coords[self._selected_inds], color=c_colors[self._selected_inds],size=self.dot_size)
                for vb in [self.view2D, self.view3D]:
                    vb.update()

    def structural_data_input(self):
        """ This function is called when the structural data checkbox is checked or unchecked

        when the structural data checkbox is checked:
            - Shows the scrollbar
            - Changes the transparency mode to translucent
            - Calls plot_structural_data to plot the data
        when the structural data checkbox is unchecked:
            - Hides the scrollbar
            - Changes the transparency mode to additive
            - Removes the structural data plot
        """
        if self.structural_data_check_cb.isChecked():
            self.structural_data_le.show()
            self.structural_data_sb.show()
            # self.switch_alpha_mode('translucent')
            self.plot_structural_data()
        if not self.structural_data_check_cb.isChecked():
            self.structural_data_le.hide()
            self.structural_data_sb.hide()
            # self.switch_alpha_mode('additive')
            self.view3D.removeItem(self.structural_data_plot)

    def plot_structural_data(self):
        """ This function gets called when the structural data checkbox is checked throught structural_data_input

        Based on the value of the scrollbar plots the structural data
        Sets the selected neurons and scrollbars according to the selected layer of structural data
        """
        levels = (np.min(self.structural_data), np.max(self.structural_data))

        self.structural_data_RGBA = pg.makeRGBA(self.structural_data[self.structural_data_sb.value()], levels=levels)[0]
        self.structural_data_RGBA = self.structural_data_RGBA.astype('float32')
        tmp_x, tmp_y = np.where(np.sum(self.structural_data_RGBA[:, :, :3], axis=2) < self.structural_black_th)
        self.structural_data_RGBA[tmp_x, tmp_y, 3] = 0
        self.structural_data_RGBA[:, :, :] *= self.brightness_multiplier
        self.structural_data_RGBA = np.clip(self.structural_data_RGBA, a_min=0, a_max=255)
        self.structural_data_RGBA[:, :, 3] = np.clip(self.structural_data_RGBA[:, :, 3], a_min=0, a_max=255)

        self.structural_data_RGBA = self.structural_data_RGBA.astype('uint8')
        self.structural_data_plot = gl.GLImageItem(self.structural_data_RGBA)

        # tmp = self.structural_data_RGBA[:, :, :3].mean(axis=2)
        # local_disk = skimage_disk(20)
        # local_mean = skimage_rank.mean(tmp / tmp.max(), local_disk)
        # local_mean = local_mean.astype('float32')
        # image_show = np.clip(tmp / (local_mean + 5), a_min=0, a_max=3)
        # image_norm = (image_show / image_show.max() * 255).astype('uint8')
        # self.structural_data_RGBA[:, :, :3] = np.repeat(image_norm[:, :, np.newaxis], 3, axis=2)

        self.structural_data_plot = gl.GLImageItem(self.structural_data_RGBA)
        self.view3D.addItem(self.structural_data_plot)
        self.structural_data_plot.translate(-0.005, -0.01, 0)  #FIXME hard coded & 3 different translation options
        n_x = self.structural_data_plot.data.shape[0]
        n_y = self.structural_data_plot.data.shape[1]
        self.structural_data_plot.translate(1 - n_x / n_x, 1 - n_y / n_y, 0)
        self.structural_data_plot.scale(0.001,0.001,0.001)

        # works based on the scrollbars maybe we have to make it standalone
        layer = self._limit_slider_values[2][self.structural_data_sb.value()] # to get the z layer based on the scrollbar
        cells = np.where(self.rec.sel_frame[:,2] == layer)
        # self.limit_start_sb.setValue(self.structural_data_sb.value())
        # self.limit_stop_sb.setValue(self.limit_start_sb.value() + 0.5)
        self.structural_data_plot.translate(0,0,float(layer))  # translate image to current z layer

        self.filter_dict['structural_data'] = cells  # add current layer selection to filter procedure
        self.selected_inds_filters()

    def pan_left(self):
        """Pan leftwards."""
        self.view3D.pan(0, self.panstep, 0)

    def pan_right(self):
        """Pan rightwards."""
        self.view3D.pan(0,-self.panstep, 0)

    def pan_front(self):
        """Pan forwards."""
        self.view3D.pan(self.panstep, 0, 0)

    def pan_back(self):
        """Pan backwards."""
        self.view3D.pan( -self.panstep, 0, 0)

    def center_view(self, update=True, vb_list=None, transpose_arr=[0, 0, 0]):
        """Reset the center of the view. If self.plotopts['centered'] is True,
        center at median of coords, if False center at (0,0,0).

        Parameters:
            update: bool, default: True
                Determines whether to call Fishualizer.update_plot()
            vb_list; List of viewboxes to update
            transpose_arr: list or arr
                transpose from median
        """
        if self._data_loaded:
            median_coords = np.median(self.rec.sel_frame, axis=0)
            if vb_list is None:
                vb_list = [self.view3D, self.view2D]
            for vb in vb_list:
                if self.plotopts['centered']:
                    vb.opts['center'] = QtGui.QVector3D(median_coords[0] + transpose_arr[0],
                                                        median_coords[1] + transpose_arr[1],
                                                        median_coords[2] + transpose_arr[2])
                else:
                    vb.opts['center'] = QtGui.QVector3D(0, 0, 0)
            if update:
                self.update_plot()

    def change_playtimer_interval(self):
        """Change the frame rate of the playtimer. This functions prompts a
        QtInputDialog whose output sets the new frame rate."""
        if self._data_loaded:
            frequency, ok_pressed = QtWidgets.QInputDialog.getDouble(self, "Playing speed",
                                                                     "Please choose a new frame rate (Hz) for the dynamic activity",
                                                                     self.frame_rate, 0.2, 500, 2)
            if ok_pressed:
                self.frame_rate = frequency
                self._playtimer.setInterval((1000 / self.frame_rate))  # convert to interval in ms
                self._logger.info(f'Frame rate changed to {frequency} Hz')
                self.statusBar().showMessage(f'Frame rate changed to {frequency} Hz')

    def regions_add(self):
        """Add region names to qcombobox in the GUI.

        This function is called by load_data() if labels are availabel in self.rec"""
        self.region_indices = [[] for _ in range(len(self.labelnames) + 1)]

        # Add the set of all regions at the beginning to the dropdown
        n_cells = self.rec.n_cells  # data['df'].shape[0]
        c_string = 'All Main Divisions' + ' (N=' + str(n_cells) + ')'
        self.region_sel_cb.addItem(c_string)
        self.regionString = (c_string + ";")
        self.region_indices[0] = self.main_zbrain_divisions
        # Add all other regions to the dropdown
        for i_pos in range(len(self.labelnames)):
            c_cell_inds = scipy.sparse.find(self.rec.labels[:, i_pos])
            n_cells = c_cell_inds[0].shape[0]
            c_string = self.labelnames[i_pos] + ' (N=' + str(n_cells) + ')'
            self.region_sel_cb.addItem(c_string)
            self.regionString += (c_string + ";")
            self.region_indices[i_pos + 1] = np.array([i_pos])
        self.labelnames.insert(0, 'AllMainDivisions')
        self.selected_regions_arr = np.zeros(len(self.labelnames))

    def draw_hull(self, coords, iH, color):
        """Draw a region hull.

        This function is called by region_change().
        Separate all neurons in two groups by kmeans. (For disjoint regions like habenula).
        Split z coords in max 100 bin, compute convex hull per bin. Then create the
        Delaunay triangulation of all neurons in the all convex hulls. Add this as
        a GLMeshItem to self.hulls, which in turn is added to self.view3D

        Parameters:
        -----------
            coords: np.array [n, 3]
                coordinates of all neurons inside the region.
            iH: int
                index of current hull. If multiple regions are drawn then iH
                keeps track of which region that is.
            color: np.array
                array of colors in RGBA format

        """
        try:
            book, _ = kmeans(coords[:, :2], 2)
            code, _ = vq(coords[:, :2], book)
            zs = set(coords[:, 2])
            hull_points = {cl: np.zeros((0, 3)) for cl in range(2)}
            if len(zs) < 100:  # for max 100 horizontal z-stacks
                horizontal_alignment = True
            else:  # z-stacks not horizontally aligned (and thus many zs values)
                horizontal_alignment = False
                n_intervals = 50  # number of new z planes to evaluate
                min_zs, max_zs = min(zs), max(zs)
                zs = np.linspace(min_zs, max_zs, n_intervals)  # create linear array of zs values
                stepsize_zs = zs[1] - zs[0]
                zs = set(zs)

            for z in zs:
                for cl in range(2):
                    c_coords = coords[code == cl]
                    """
                    ## This commented section defines a hard-coded midline cut
                    # of the Mes. and Tel. if all main divisions are plotted. Caveat:
                    # only works in zbrain coordintes (because of hard coded midline coords).!
                    if iH == 4 or iH == 1:
                        if iH == 1:
                            self._logger.debug('Mesencephalic midline cut')
                        if iH == 4:
                            self._logger.debug('Telencephalic midline cut')
                        if cl == 0:
                            c_coords = coords[np.where(coords[:, 0] <= 0.248)[0], :]
                        elif cl == 1:
                            c_coords = coords[np.where(coords[:, 0] > 0.248)[0], :]
                    """
                    if horizontal_alignment:
                        c_slice = c_coords[np.isclose(c_coords[:, 2], z), :]
                    else:  # change absolute tolerance(=stepsize_zs) to include all neurons in z-vicinity
                        c_slice = c_coords[np.isclose(c_coords[:, 2], z, atol=stepsize_zs), :]
                    if c_slice.shape[0] < 4:
                        hull_points[cl] = np.vstack((hull_points[cl], c_slice))
                    else:
                        hull = ConvexHull(c_slice[:, :2])
                        hull_points[cl] = np.vstack((hull_points[cl], c_slice[hull.vertices, :]))

            for cl in range(2):
                c_coords = hull_points[cl]
                d = scipy.spatial.Delaunay(c_coords)
                print(color)
                # if self.hulls[cl] is None:
                self.hulls[iH][cl] = gl.GLMeshItem(vertexes=c_coords, faces=d.convex_hull, color=color, smooth=True,
                                                   drawEdges=False, drawFaces=True)
                self.hulls[iH][cl].setGLOptions('additive')
                for vb in [self.view3D, self.view2D]:
                    vb.addItem(self.hulls[iH][cl])

        except scipy.spatial.qhull.QhullError:  # this error can otherwise crash the app
            self._logger.error("QHull error while computing hulls")
            print("QHull error while computing hulls")

    def region_changed_by_combobox(self):
        """Function that removes all plotted regions (via color plotting) except for currently selected regions via the combobox."""
        self.region_check_cb.setChecked(True)
        self.selected_regions_arr = np.zeros(len(self.selected_regions_arr)) #clears it for other uses of the function
        self.selected_regions_arr[self.region_sel_cb.currentIndex()] = 1
        self.region_change()

    def region_change(self):
        """Prepare a new region draw of the currently selected region.

        This function is called if the region combobox is changed, or by other functions
        if a new region needs to be drawn.
        This function removes previous hulls and selects coordinates to be used.
        Then draw_hull() is called for the new region draw.
        """
        if self._data_loaded:
            selected_region_indices = np.where(self.selected_regions_arr)[0] # indices in self.region_sel
            region_indices = []
            for index in selected_region_indices:
                region_indices = np.concatenate([region_indices, self.region_indices[index]])  # needed because of potential inclusion of main divisions array
            region_indices = [int(item) for item in region_indices]
            if self.hulls is not None: # in these loops and ifs there can be an IndexError because it will always loop and therefore it will get in to a ERROR
                for iH in range(len(self.hulls)):
                    for iL in range(len(self.hulls[iH])):
                        for vb in [self.view3D, self.view2D]:
                            if self.hulls[iH][iL] is not None and self.hulls[iH][iL] in vb.items:
                                    vb.removeItem(self.hulls[iH][iL])

            self.hulls = [[] for _ in region_indices]
            if self.neuron_check_cb.isChecked() == True:
                cells = scipy.sparse.find(self.rec.labels[:,region_indices])[0]  # indices of all neurons in current selection of regions
                self.filter_dict['region_filter'] = cells  # set region_filter to this selection of cells
                self.selected_inds_filters()  # update combined selection
            for iH, i_region in enumerate(region_indices):
                if i_region in self.region_spec_color.keys():# or self.region_sel_cb.currentIndex() in self.region_spec_color.keys():
                    color = self.region_spec_color.get(i_region) # selected_region_indices[iH] gets the correct index cause region_indices are the index - 1
                else:
                    color = self.current_hull_color
                self.hulls[iH] = [None, None]
                c_region_index = region_indices[iH]
                c_cell_inds = scipy.sparse.find(self.rec.labels[:, c_region_index])
                c_cell_inds = c_cell_inds[0]
                if self._use_zbrainatlas_coords is False:  # use coordinates of cells in region to create the region hull
                    c_coordinates = self.rec.sel_frame[c_cell_inds, :]
                elif self._use_zbrainatlas_coords:  # use coordinates of grid points in zbrain atlas to create the region hull (with the advantage that the hull shows the full region, rather than only the volume occupied by cells in the region)
                    grid_unit_inds = np.where(self.rec.zbrainatlas_regions == c_region_index)[0]
                    c_coordinates = self.rec.zbrainatlas_coordinates[grid_unit_inds, :]
                self._logger.info(f'Drawing index: {c_region_index} and name: {self.labelnames[c_region_index + 1]}')
                print('Drawing index: {} and name: {}'.format(c_region_index, self.labelnames[c_region_index + 1]))
                if c_coordinates.shape[0] > 0:
                    self.draw_hull(c_coordinates, iH, color)
                else:
                    self._logger.warning('No Neurons assigned to this region.')
                    self.statusBar().showMessage('No Neurons assigned to this region.')
                    print('No Neurons assigned to this region.')

    def region_show(self):
        """Enable region hulls if region checkbox is checked.

        Set visibility of the different regions. Call self.region_change() to draw the hull
        unless self.hulls exist (when region is unchanged but region_check_cb has been unchecked/checked)."""
        show_hull = self.region_check_cb.isChecked()
        # if show_hull:  # not necessary to print color I reckon?
        #     print('Current Color: {}'.format(self.current_hull_color)) #only print if checked True

        #TODO: What does the section below do? It changes the FaceColors but does not redraw right? Doesn't seem to have an effect when redrawn via changing selection back and forth..
        # if self.hulls is not None:
        #     for iH in range(len(self.hulls)):
        #         for iL in range(len(self.hulls[iH])):
        #             self.hulls[iH][iL].opts['meshdata'].setFaceColors(self.current_hull_color)

        if show_hull:
            self.region_changed_by_combobox()
        else:
            self.remove_regions()

    def neuron_show(self):
        if self.neuron_check_cb.isChecked():
            self.region_check_cb.setChecked(True)
            self.region_change()
        if not self.neuron_check_cb.isChecked():
            self.reset_view()

    def add_static_gui(self):
        """ Call UI to add static data."""
        if self._data_loaded:
            self.static_data_dialog.show()
        else:  # main data is not loaded, this would a error later (self.data not defined, no coordinates etc.)
            self._logger.warning("Please load main data file first before adding static data sets")

    def compute_mean_activity(self):
        if self._data_loaded:
            result = np.mean(self.rec.datasets[self.data_set],1)
            self.add_static(result, 'Mean Activity')
            self.statusBar().showMessage("Mean activity computed.")

    def compute_correlation_gui(self):
        """ Call UI to compute correlation w.r.t. stimulus or behavior."""
        if self._data_loaded:
            enough_mem = False  # default to False
            if ("h5py" in str(type(self.rec.df))):  # if df is memory mapped:
                size_data_set = os.path.getsize(self.data_path)
                available_ram_memory = ps.virtual_memory().available
                if size_data_set > available_ram_memory:
                    WarningDialog(self, str(ps.virtual_memory().available)).show() # Message that gives detailed report concerning memory availability.
                    self._logger.warning("The correlation analyis was cancelled, due to insufficient RAM memory.")
                else:
                    enough_mem = True
            if ("numpy.ndarray" in str(type(self.rec.df)) or enough_mem == True):  # If enough memory is available, or df is already in RAM (i.e. it is a np.array)
                for x in self.rec.single_data:  #  add single data sets to correlation gui (if necessary).
                    if x in self.rec.available_data:
                        if x not in [self.compute_correlation_dialog.correlation_single.itemText(i) for i in range(
                                self.compute_correlation_dialog.correlation_single.count())]:  # if not already in QComboBox, alternatively one could reset all QComboBox options
                            self.compute_correlation_dialog.correlation_single.addItem(x)

                for x in self.rec.multi_data:  #  add multi (i.e. dynamic) data sets to correlation gui (if necessary).
                    if x in self.rec.available_data:
                        if x not in [self.compute_correlation_dialog.correlation_multi.itemText(i) for i in
                                     range(self.compute_correlation_dialog.correlation_multi.count())]:
                            self.compute_correlation_dialog.correlation_multi.addItem(x)
                self.compute_correlation_dialog.show()  # call dialog
            if ("h5py" not in str(type(self.rec.df)) and "numpy.ndarray" not in str(type(self.rec.df))):
                #TODO: This should not be possible?
                self._logger.warning("The data is of the incorrect file type to be used for correlation analysis.")
        else:  # main data is not loaded
            self._logger.warning("Please load main data file first before computing correlations")

    def status_bar_loader(self, progress):
        """ Update the statusBar to show the progress in percentages.

        Parameter:
        -----------
            - progress, int, float,
                The current percentage to show in the statusBar
        """
        self.statusBar().showMessage(f"Loading: {progress}%")
        if progress == 100:
            self.statusBar().showMessage("Loading Done")

    def compute_correlation_function(self):
        """ Compute correlation between single time trace data and multi time traces data.

        Uses current settings of QComboBoxes in compute_correlation_dialog to fetch data sets
        Adds correlation to static data sets and add to self.rec via add_static()
        """
        name_single = self.compute_correlation_dialog.correlation_single.currentText()  # get current selections
        name_multi = self.compute_correlation_dialog.correlation_multi.currentText()
        kwargs_fun = {'sig_1D': name_single, 'sigs_2D': name_multi}
        self.statusBar().showMessage(f'Computing correlation between {name_multi} and {name_single}. Please wait')
        result, name_result = self.rec.compute_analysis(self, correlation_1D2D, **kwargs_fun)
        name_result = name_result + '_' + name_single
        self.add_static(result, name_result)
        self.statusBar().showMessage("Correlation analysis done.")

    def select_file_and_load(self, option='static'):
        """UI to load datasets by browsing through directories.

        Parameter:
        -----------
            - option, str, optional
                if 'connection' a sparse connectivity matrix is loaded
                else a single static data set must be added.

        Returns:
        -----------
            c_data: multiple types possible (dependent on file type)
                loaded data set
            import_data_method: str, is defined in this function
                Because multiple data types can be returned, this parameter gives
                'instructions' how to handle the data to the function calling select_file_and_load()

        """
        # FIXME: Rémi: I don't really like this function: complex with multiple return types
        # .npy & .npz: single datasets, .h5: pd matrix of multiple sets.
        # cwd = os.getcwd()
        cwd = self.user_params['paths']['static_path']
        dpath, _filter = QtWidgets.QFileDialog.getOpenFileName(self, "Choose a data file", cwd,
                                                               'Data (*.npy *.npz *.h5 *.db)')  # get datapath, like main open_data etc.

        if dpath == '':
            return

        # Set the new data path.
        self._data_path = dpath  # do not set self.data_path, because that function opens load_data() at the moment. I think we should avoid altering that sequence (for now)
        self.user_params['paths']['static_path'] = os.path.dirname(dpath)  # Save it for next time
        ext = os.path.splitext(self._data_path)[1]  # Get the extension
        #  load the data according to the method related to the extension

        if option == 'connection':
            #TODO: hard require that now only a .npz file can be loaded? Move the UI getOpenFileName inside if statement?
            c_data = scipy.sparse.load_npz(self._data_path)
            import_data_method = 'connectivity'
        elif option == 'density':
            c_data = self._data_path
            import_data_method = 'density'
        elif option == 'static':
            try:
                if ext == '.npy':  # load regular files
                    c_data = np.squeeze(np.load(self._data_path))  # add file to self.data dictionary
                    import_data_method = 'single'
                elif ext == '.npz':  # load sparse files, squeeze to ensure proper shape
                    c_data = np.squeeze(scipy.sparse.load_npz(self._data_path).A)
                    import_data_method = 'single'
                elif ext == '.h5':  # load pd matrix
                    c_data = pd.read_hdf(self._data_path)
                    import_data_method = 'multi'
                elif ext == '.db':  # Load shelve
                    import_data_method = 'shelve'
                    s_win = ShelveWindow(self, dpath)
                    c_data = s_win.get_imported_datasets()
            except KeyError:
                self._logger.error(f'Invalid static data file {dpath}')
                # if some invalid file is added, it crashes when the option is data in the dropdown menu (KeyError)
                return

        return c_data, import_data_method

    def load_static(self, threshold_params=None):
        """Load static data set by calling UI function to select and call add_static() to add."""
        c_data, import_data_method = self.select_file_and_load()
        self._logger.debug(f'threshold_params set to {threshold_params}')
        if import_data_method == 'single':
            dataname = self.static_data_dialog.load_dataname.text()
            if len(dataname) == 0:
                dataname = os.path.basename(self._data_path)
            self.add_static(c_data, dataname, threshold_params=threshold_params)

        elif import_data_method == 'multi':
            column_names = list(c_data.columns.values)  # get column names
            n_neurons = len(self.rec.sel_frame)
            ind_neurons = c_data.index.tolist()  # indices of neurons in df
            for c_name in column_names:  # loop through columns
                # Create new data variables (with column names) in self.data, as np.zeros with column inserted?
                c_data_ind = np.zeros(n_neurons)  # create empty array (not-specificied neuron thus get default-value 0)
                c_data_ind[ind_neurons] = c_data[c_name]  # put in values of c_name column
                self.add_static(c_data_ind, c_name, threshold_params=threshold_params)
        elif import_data_method == 'shelve':
            for c_name, c_data in c_data.items():
                # TODO: Very crude check that the shapes may align
                if len(c_data) == self.rec.n_cells:
                    self.add_static(c_data, c_name, threshold_params=threshold_params)

    def load_density(self):
        dpath, _ = self.select_file_and_load(option='density')
        self.draw_density_map(datapath=dpath, maptype='density_map', cut_off_threshold=1,
                              density_scale_factor=1, slice_density=5)
        self.show_density = True
        self.toggle_density_action.setEnabled(True)
        self.save_checking(action=self.toggle_density_action, check=True)

    def add_static(self, c_data, dataname, threshold_params=None):
        """Add a static data set to Zecording and set appropriate internal parameters

        Parameters:
        ----------
        c_data: np array/list of len n_cells x 1
             dataset to add
        dataname: str
             dataname to use, # DONE: check if already exists, else alter to avoid overwriting?
        """
        suffix = 1
        start = True
        c_data = np.squeeze(c_data)
        assert len(c_data) == self.rec.n_cells and c_data.ndim == 1
        if threshold_params is not None:
            if threshold_params['binarise']:  # binaraise the input data
                new_c_data = np.zeros_like(c_data)
                new_c_data[np.abs(c_data) >= threshold_params['threshold']] = 1  # if absolute value geq threshold, 1, else 0
                c_data = new_c_data
        while dataname in self.rec.available_data:
            if start:
                dataname = f'{dataname}_{suffix}'
            else:
                previous_suffix_str = f'_{suffix-1}'
                dataname = dataname.rstrip(previous_suffix_str)  # does not break while loop because dataname must be in
                dataname = f'{dataname}_{suffix}'
            start = False
            suffix += 1

        if dataname == 'output':
            self.rec.output = self.rec.input  # because they are the same, input is added first in add_connectivity(), better for memory allocation(?)
        elif dataname not in self.rec.available_data:
            setattr(self.rec, dataname, c_data)
            self.rec.available_data.add(dataname)
        if dataname == 'output' or dataname == 'input':
            self.disp_start[dataname] = scipy.sparse.csr_matrix.min(self.rec.datasets[dataname])
            self.disp_stop[dataname] = scipy.sparse.csr_matrix.max(self.rec.datasets[dataname])
        else:
            self.disp_start[dataname] = np.percentile(self.rec.datasets[dataname], 1)
            self.disp_stop[dataname] = np.percentile(self.rec.datasets[dataname], 99)
        self.disp_change[dataname] = False
        # if almost all neurons are 0, the 99 percentile also is 0 (and the fish is not drawn)
        if self.disp_stop[dataname] == 0:
            self.disp_stop[dataname] = 1
        self.swap_color_data[dataname] = False
        self.data_sel_cb.addItem(dataname)  # add option to dropdown menu
        self.i_dataset += 1  # increase index
        self.data_names[self.i_dataset] = dataname
        self.datasets_scrollbar.setRange(0, len(self.data_names) - 1)
        self.datasets_scrollbar.setValue(len(self.data_names) - 1)
        self.data_sel_cb.setCurrentText(dataname)

    def compute_static(self):
        """Compute static data set based on user input formula."""
        formula = self.static_data_dialog.compute_formula.text()
        c_data = eval('np.apply_along_axis(' + formula + ',1,self.rec.datasets[self.data_set])')
        self.add_static(c_data, formula)

    def add_connectivity(self):
        """Add sparse connectivity matrix to two static data sets (input and output)."""
        connections = self.select_file_and_load('connection')[0]
        self.add_static(connections, 'input')
        self.add_static(connections, 'output')

    def plot_trace(self, index=0, alt=0):
        """Plot trace of selected neuron in self.activity_pw

        Calcium trace is deconvolved online, spikes are also plotted.

        Parameters:
        ----------
            index, optional (default = 0)
                index of the selected neuron
            alt, option (default = 0)
                Different neurons can be plotted simultaneously (currently 2 neurons)
                the alt parameter tracks which plot is affected.
        """
        self.activity_plot_ind = alt

        c_region = 'not assigned'
        if self._labels_available:
            c_region_index = scipy.sparse.find(self.rec.labels[index, :])
            if c_region_index[1].size > 0 and self.labelnames is not None:
                c_region = self.labelnames[c_region_index[1][-1] + 1]  # Account for all region

        self.activity_plots[self.activity_plot_ind].setData(x=self.rec.times, y=self.rec.df[index, :])
        if 'spikes' in self.rec.available_data:
            self.activity_plot_spikes[self.activity_plot_ind].setData(x=self.rec.times,
                                                                      y=self.rec.spikes[index, :] + 1)
        elif 'oasis' in sys.modules:  # only if oasis module is loaded, spikes can be deconvolved
            single_spike_train = self.fast_deconvolv(n=index)  # deconvolve online (just 1 neuron)
            self.activity_plot_spikes[self.activity_plot_ind].setData(x=self.rec.times,
                                                                      y=single_spike_train[1] + single_spike_train[
                                                                          2])  # spike train + individual baseline
            if self.plot_calcium_fit == 1:
                self.activity_plot_calcfit[self.activity_plot_ind].setData(x=self.rec.times,
                                                                           y=single_spike_train[0] + single_spike_train[
                                                                               2])  # calcium fit + individual baseline
            elif self.plot_calcium_fit == 0:
                self.activity_plot_calcfit[
                    self.activity_plot_ind].clear()  # remove remaining plot (from before the toggle)
        self.activity_text.setPlainText(f'Neuron {index} ({c_region}). Coordinates: {np.round(self.rec.sel_frame[index, :], 3)}')
        self.current_neuron_plot_ind = (index, alt)  # to feed back when calcium fit is toggled on/off

    def edit_cmap(self):
        """ Create UI to choose a colormap."""
        super().edit_cmap()
        items = colors.get_all_cmaps()
        # items = [x for x in plt.colormaps() if not x.endswith('_r')]
        # items = [x for x in items if 'Vega' not in x]
        # items = [x for x in items if x != 'spectral']
        # items = [x for x in items if type(plt.get_cmap(x)) is matplotlib.colors.ListedColormap]
        # items.append('GFP')
        item, ok_pressed = ColorMapDialog(self, "Choose colormap", "Name:", items).get_colormap()
        if ok_pressed and item:
            self._cmap_name = item
            self.make_cmap()
            self.rate_sel_cp.repaint()

    def edit_hide(self, threshold='min'):
        """ Create UI to hide a selection of neurons below or above threshold

        Plot mean trace of visible neurons in behavior_pw widget

        Parameter
        ----------
            threshold, 'min' or 'max':
                'min' minimum threshold, hide neurons below
                'max' maximum threshold, hide neurons above
        """
        super().edit_hide()
        current_values = self.get_current_values()
        question_str = 'Choose ' + threshold + 'imum threshold value'
        if threshold == 'min':
            threshold_limit = current_values.min()
        if threshold == 'max':
            threshold_limit = current_values.max()
        value, ok_pressed = QtWidgets.QInputDialog.getDouble(self, question_str, "Threshold:",
                                                             threshold_limit, decimals=3)

        if ok_pressed:
            if threshold == 'min':
                cells = np.where(current_values > value)[0]
                self.filter_dict['min_threshold'] = cells
            elif threshold == 'max':
                cells = np.where(current_values < value)[0]
                self.filter_dict['max_threshold'] = cells
            if len(self._selected_inds) != self.rec.n_cells:  # only compute if a subset of all cells is selected (if all cells are selected, it is usually to reset the view, and computing their mean, then a byproduct, takes time)
                weights = current_values[self._selected_inds] / sum(current_values[self._selected_inds])  # weighted mean of visible cells
                weighted_mean = np.dot(weights, self.rec.df[self._selected_inds, :])
                # all_weights = current_values / sum(current_values)  # weighted mean of all cells (including invisible ones)
                # all_weighted_mean = np.dot(all_weights, self.rec.df)
                # self.plot_single_df_trace(y_trace=weighted_mean)
                self.plot_single_df_trace(y_trace=self.rec.df[self._selected_inds, :].mean(axis=0))  # unweighted mean of visible cells
            else:
                self.mean_selection_plot.clear()
        self.selected_inds_filters()

    def plot_single_df_trace(self, y_trace):
        """Plot single df/f trace in behavior_pw.

        This can be any single trace, but with x_trace=self.rec.times (because of the XLink)

        Parameters:
        ----------
            - x_trace
            - y_trace
        """
        x_trace=self.rec.times
        self.mean_selection_plot.clear()
        self.mean_selection_plot.addItem(pg.PlotDataItem(x=x_trace, y=y_trace))  # plot the mean of the selection

    def update_behavior_plot(self):
        """ Update the scaling of the self.mean_selection_plot (w.r.t. self.behavior_pw)."""
        self.stimulus_plot.setGeometry(self.behavior_pw.viewRect())
        self.behavior_plot.setGeometry(self.behavior_pw.viewRect())
        self.mean_selection_plot.setGeometry(self.behavior_pw.viewRect())

    def fast_deconvolv(self, n):  # calcium_trace = 0): # deconvolve neuron n
        """ Use OASIS method to deconolve selected cell online.

        ----------
        Parameters:
        - n: int, index of neuron

        ---------
        Returns:
        - c, calcium fit (from s)
        - s, inferred spike trace
        - sig_deconv[2], baseline constant of inference
        """

        ytrace = self.rec.df[n, :]  # get calcium trace
        ytrace = ytrace.astype(np.float64)  # oasis package only takes float64 as input (in C conversion)
        # deconvolve for warm start, optimize gamma (= decay time constant)
        sig_deconv = oasis.functions.deconvolve(ytrace, penalty=0, optimize_g=5)
        c, s = oasisAR1(ytrace - sig_deconv[2], sig_deconv[3],
                        s_min=0.3)  # deconvolve again, now set minimal spike height s_min (free parameter!)
        return c, s, sig_deconv[2]  # return (fitted calcium trace, spike train, baseline)

    def toggle_calcium_fit(self):
        """ Toggle whether to plot the calcium fit in the neuronal plot.

        Calcium fit is calculated (together with inferred spike train) by the OASIS method.
        """
        if self.plot_calcium_fit == 1:
            self.plot_calcium_fit = 0
        elif self.plot_calcium_fit == 0:
            self.plot_calcium_fit = 1
        self.plot_trace(index=self.current_neuron_plot_ind[0], alt=self.current_neuron_plot_ind[1])  # refresh plot

    def show_region_current_cell(self):
        """Draw region of currently selected cell.

        If regions are loaded and a cell is selected, the hull is drawn
        of the corresponding cell. Cells usually have multiple reason, the smallest one is drawn.
        Other regions are printed.
        """
        if self._labels_available and self.region_check_cb.isChecked():
            index = self.current_selected_cell  # take selected neuron
            c_region_index = scipy.sparse.find(self.rec.labels[index, :])
            c_region_print = {}
            if c_region_index[1].size > 0 and self.labelnames is not None:
                n_cells_region = np.zeros(len(c_region_index[1]))
                for iLoop, iR in enumerate(c_region_index[1]):
                    c_cell_inds = scipy.sparse.find(self.rec.labels[:, iR])
                    n_cells_region[iLoop] = c_cell_inds[0].shape[0]  # number of cells in this region
                    c_region_print[iR] = self.labelnames[iR + 1]
                for iLoop, iR in enumerate(c_region_index[1]):
                    if iLoop == np.argmin(n_cells_region):  # if smallest region, save index for drawing:
                        i_region_plotted = iR
                        print(f'Neuron {index} is in region {iR}; {c_region_print[iR]}, plotted')
                    else:  # not drawn
                        print(f'Neuron {index} is in region {iR}; {c_region_print[iR]}, not plotted')
            # Currently: show region with smallest N? Toggle this?
            self._region_selected_prior[0] = True  # use this region for drawing
            self._region_selected_prior[1] = i_region_plotted + 1  # add plus one, because region_change() expects this shift (due to zero-default and dropwown selection)
            # self.region_sel_cb.setCurrentIndex(i_region_plotted + 1)  # add plus for dropdown selection
            self.selected_regions_arr[i_region_plotted] = 1
            self.region_change()

    def toggle_hideneuropil(self):
        """Toggle visibilty of neuropil on or off.

        Neuropil data is retrieved from self.rec, and should thus be provided in
        the main hdf5 data file.
        """
        self.hide_neuropil = not self.hide_neuropil  # toggle
        self.update_plot()
        if 'not_neuropil' in self.rec.available_data:
            if self.hide_neuropil:
                n_neuropil_cells = int(len(self.rec.not_neuropil) - sum(self.rec.not_neuropil))
                self._logger.info(f'Neuropil is not plotted ({n_neuropil_cells} neuropil cells)')
            elif not self.hide_neuropil:
                self._logger.info('Neuropil is plotted')
        else:
            self._logger.warning('Neuropil information is not loaded in the data')

    def toggle_hidegrid(self):
        """Toggle visibilty of grid on or off. """
        self.hide_grid = not self.hide_grid  # toggle
        for iG in range(3):
            if self.hide_grid is False:
                self.Grids[iG].show()
            else:
                self.Grids[iG].hide()
        self.view2D.update()
        self.view3D.update()

    def toggle_somas(self):
        """Toggle visibilty of somata. """
        self.show_somas = not self.show_somas  # toggle
        self.save_checking(action=self.somas_action, check=self.show_somas)
        self.update_plot()

    def toggle_nanview(self):
        """Toggle visibilty of NaN values. """
        self.hide_nans = not self.hide_nans  # toggle
        self.save_checking(action=self.nan_action, check=self.hide_nans)
        self.update_plot()

    def toggle_randomzfill(self):
        """Toggle visibilty of NaN values. """
        self.z_random_shift = not self.z_random_shift  # toggle
        self.save_checking(action=self.randomz_action, check=self.z_random_shift)
        self.update_plot()

    def set_window_sizes(self, window_sizes=[200, 100, 0]):
        '''Set window sizes of the three windows'''
        if (window_sizes is not None) and (window_sizes is not False):
            self.v_splitter.setSizes(window_sizes)  # change sizes of view3D : view2D : lineplot (i.e. top to bottom)

    def switch_topside(self, add_scalebar=True):
        """Switch 3D and 2D plot to standard top-side view for publications"""
        self.center_view(vb_list=[self.view3D])  # center camera position
        self.center_view(vb_list=[self.view2D], transpose_arr=[0, 0, -0.03])
        self.viewxy(self.view3D)  # switch 3D plot to top view
        self.viewyz(self.view2D)  # switch 2D plot to side view
        if add_scalebar:
            self.add_scale_bar()  # Add default scale bar
        if self.bool_show_ref_axes:  # if ref axes are currently shown, hide them:
            self.toggle_reference_axes()
        if hasattr(self,'select_3D'):
            self.select_3D.hide()  # hide cell selection
        self.set_window_sizes(window_sizes=[200, 100, 0])
        for vb in [self.view3D, self.view2D]:
            self.set_perspective_2d(view=vb)  # Change to 2D perspective for both
        for ax in [self.reference_x_axis, self.reference_y_axis, self.reference_z_axis]:
            ax.hide()  # Hide the cross
        self.view2D.update()
        self.view3D.update()

    def increase_dot_size(self):
        self.dot_size += 1
        self.update_plot()

    def decrease_dot_size(self):
        self.dot_size -= 1
        if self.dot_size < 1:
            self.dot_size = 1
        self.update_plot()

    def swap_color_axis(self):
        """ Change color axis orientation of current data set (self.data_set).

        Currently hard-coded: also change the disp_start and disp_stop. This functionality
        is currently used to view either postive or negative extremes (of e.g.
        correlations), it saves some scrolling to reset the disp_start and disp_stop.
        """
        self.swap_color_data[self.data_set] = not self.swap_color_data[self.data_set]  # swap color axis
        if self.swap_color_data[self.data_set]:
            self.disp_start[self.data_set] = 0
            self.disp_stop[self.data_set] = np.percentile(self.rec.datasets[self.data_set], 99)
        else:
            self.disp_start[self.data_set] = np.percentile(self.rec.datasets[self.data_set], 1)
            self.disp_stop[self.data_set] = 0
        self.limit_color_start_sb.setValue(self.disp_start[self.data_set])
        self.limit_color_stop_sb.setValue(self.disp_stop[self.data_set])
        self.make_cmap()  # update the colormap
        self.update_plot()

    def add_supplementary_traces(self):
        """ Add some supplementary single traces to the data set.

        Calling this function adds some pre-defined functions (related to the stimulus) to Zecording class
        This function can currently only be accessed from the console.
        """
        pos_only = np.zeros_like(self.rec.stimulus)
        neg_only = np.zeros_like(self.rec.stimulus)
        pos_only[np.where(self.rec.stimulus > 0)[0]] = self.rec.stimulus[np.where(self.rec.stimulus > 0)[0]]  # positive stimulus only
        neg_only[np.where(self.rec.stimulus < 0)[0]] = self.rec.stimulus[np.where(self.rec.stimulus < 0)[0]]  # negative stimulus only

        supp_data = {'deriv_stim': np.gradient(self.rec.stimulus),
                     'abs_deriv_stim': np.abs(np.gradient(self.rec.stimulus)),
                     'pos_only_stim': pos_only,
                      'abs_neg_only_stim': np.abs(neg_only)}  # dictionary of supplementary data to add

        for data_name in supp_data:  # put all supp. data in rec
            self.rec.add_supp_single_data(s_name=data_name, s_data=supp_data[data_name])

        pos_deriv = self.rec.deriv_stim.copy()
        pos_deriv[pos_deriv < 0] = 0
        neg_deriv = self.rec.deriv_stim.copy()
        neg_deriv[neg_deriv > 0] = 0
        self.rec.add_supp_single_data('pos_deriv_stim', pos_deriv)  # positive derivative only
        self.rec.add_supp_single_data('neg_deriv_stim', neg_deriv)  # postiive derivative only

        def exp_smooth(trace, time_constant = (2.6 * 2.5)):
            """Exponential smoothing

            Defined inside add_supplementary_traces()

            Parameters
            ----------
                trace: np.array,
                    trace to smooth
                time_constant: float, optional (=2.6 default, GCaMP6s line of Migault et al., submitted)

            Returns
            ----------
                conv_trace:, float
                    convolved trace
            """
            alpha_test = (1 - np.exp(-1/ time_constant)) # exponential decay time constant from Migault et al., 2018, biorxiv paper supp. methods (vestibular)
            k = lambda tau: alpha_test*(1-alpha_test)**tau
            k_len = len(trace)//10
            kernel = np.hstack((np.zeros(k_len), k(np.arange(k_len))))
            conv_trace = np.convolve(trace, kernel, mode='same') / np.sum(kernel)
            return conv_trace

        # Add exponentially smoothed functions. Especially useful for fast derivatives.
        self.rec.add_supp_single_data('conv_stim', exp_smooth(self.rec.stimulus))
        self.rec.add_supp_single_data('conv_pos_stim', exp_smooth(self.rec.pos_only_stim))
        self.rec.add_supp_single_data('conv_neg_stim', exp_smooth(self.rec.abs_neg_only_stim))
        self.rec.add_supp_single_data('conv_pos_deriv_stim', exp_smooth(self.rec.pos_deriv_stim))
        self.rec.add_supp_single_data('conv_neg_deriv_stim', exp_smooth(self.rec.neg_deriv_stim))
        self.rec.add_supp_single_data('abs_conv_neg_deriv_stim', np.abs(exp_smooth(self.rec.neg_deriv_stim)))
        self.rec.add_supp_single_data('abs_conv_all_deriv_stim', np.abs(exp_smooth(self.rec.deriv_stim)))

    def call_reverse_x_coords(self):
        """Function to call the Zecording.reverse_coords() in self.rec

        # TODO: swap x legend (Red one)
        self.rec.reverse_coords() reverses the x coordinates by:
        x_new = max(x_old) - (x_old - min(x_old))"""
        if self._data_loaded:
            if 'zbrainatlas_coordinates' not in self.rec.available_data:  # Load to make sure everything is revsered simultaneously
                self.add_zbrainatlas()  # load using default path (in ~/fishualizer/data/ZBrainAtlas folder)
            self.rec.reverse_coords(dim=0)
            self.bool_reverse_x = not self.bool_reverse_x
            self.update_plot()
            self._logger.info('x coordinates have been reversed')
        else:
            pass

    def call_reverse_y_coords(self):
        """Function to call the Zecording.reverse_coords() in self.rec

        # TODO: swap y legend (Green one)
        self.rec.reverse_coords() reverses the y coordinates by:
        y_new = max(y_old) - (y_old - min(y_old))"""
        if self._data_loaded:
            if 'zbrainatlas_coordinates' not in self.rec.available_data:  # Load to make sure everything is revsered simultaneously
                self.add_zbrainatlas()  # load using default path (in ~/fishualizer/data/ZBrainAtlas folder)
            self.rec.reverse_coords(dim=1)
            self.bool_reverse_y = not self.bool_reverse_y
            self._logger.info('y coordinates have been reversed')
            self.update_plot()
        else:
            pass

    def call_reverse_z_coords(self):
        """Function to call the Zecording.reverse_coords() in self.rec

        # TODO: swap z legend (Blue one)
        self.rec.reverse_z_coords() reverses the z coordinates by:
        z_new = max(z_old) - (z_old - min(z_old))"""
        if self._data_loaded:
            if 'zbrainatlas_coordinates' not in self.rec.available_data:  # Load to make sure everything is revsered simultaneously
                self.add_zbrainatlas()  # load using default path (in ~/fishualizer/data/ZBrainAtlas folder)
            self.rec.reverse_coords(dim=2)
            self.bool_reverse_z = not self.bool_reverse_z
            self._logger.info('z coordinates have been reversed')
            self.update_plot()
        else:
            pass


    def call_switch_xy_coords(self):
        """Function to call the Zecording.switch_xy_coords() in self.rec
        """
        if self._data_loaded:
            self.rec.switch_xy_coords()
            self.update_plot()
        else:
            pass


    def update_coordinate_system(self):
        """Plot coordinate system for reference of front and scale.

        Parameters (length etc.) are currently hard coded in this function.
        Coordinate system is created and added to self.view3D
        """
        axes = [self.reference_x_axis, self.reference_y_axis, self.reference_z_axis]
        for ax in axes:
            if ax is not None:
                self.view3D.removeItem(ax)  # if previously initiated, remove axes.

        scale_factor = self.rec.spatial_scale_factor
        axis_width = 5
        ref_length = 0.1
        opposite_site_sign = -1 * np.sign(scale_factor)  # used to set coordinate system in opposite quadrant (so that it is not visible in orthogonal views)
        #ref_pos = opposite_site_sign * np.array([0.25, .5, 0])
        ref_pos = np.array([0, 0, 0])
        x_axis = np.array([ref_pos + [0, 0, 0], ref_pos + [(scale_factor[0] * ref_length), 0, 0]])
        x_color = np.array([[1, 0, 0, 1], [1, 0, 0, 1]])
        self.reference_x_axis = gl.GLLinePlotItem(pos=x_axis, color=x_color, width=axis_width)
        self.view3D.addItem(self.reference_x_axis)

        y_axis = np.array([ref_pos + [0, 0, 0], ref_pos + [0, (scale_factor[1] * ref_length), 0]])
        y_color = np.array([[0, 1, 0, 1], [0, 1, 0, 1]])
        self.reference_y_axis = gl.GLLinePlotItem(pos=y_axis, color=y_color, width=axis_width)
        self.view3D.addItem(self.reference_y_axis)

        z_axis = np.array([ref_pos + [0, 0, 0], ref_pos + [0, 0, (scale_factor[2] * ref_length)]])
        z_color = np.array([[0, 0, 1, 1], [0, 0, 1, 1]])
        self.reference_z_axis = gl.GLLinePlotItem(pos=z_axis, color=z_color, width=axis_width)
        self.view3D.addItem(self.reference_z_axis)
        self.bool_show_ref_axes = True

    def scale_coordinates(self, xscale=1, yscale=1, zscale=1):
        """Scale all coordinates in visualization.

        This function does not overwrite the hdf5 data file. It sets
        self.rec.spatial_scale_factor to the given scaling and updates the plot.

        Parameters:
        ------------
            xscale: float
                new scaling of x coordinates
            yscale: float
                new scaling of y coordinates
            zscale: float
                new scaling of z coordinates
        """
        if self._data_loaded:
            self.rec.spatial_scale_factor = (xscale, yscale, zscale)  # set scaler
            if self.region_check_cb.isChecked():
                self.region_change()  # redraw region because scaling factors and thus coordinate plotting has changed.
            self.update_coordinate_system()
            #TODO save to hdf5 file as attribute (do this inside self.rec.spatial_scale_factor?)
            self.update_plot()
        else:
            pass

    def inverse_x_axis(self):
        """Function that inverts the x axis of all coordinate data sets. It calls
        Fishualizer.scale_coordinates() with updated scale factors."""
        current_scaling = self.rec.spatial_scale_factor
        self.scale_coordinates(xscale=(-1*current_scaling[0]), yscale=current_scaling[1], zscale=current_scaling[2])

    def inverse_y_axis(self):
        """Function that inverts the y axis of all coordinate data sets. It calls
        Fishualizer.scale_coordinates() with updated scale factors."""
        current_scaling = self.rec.spatial_scale_factor
        self.scale_coordinates(xscale=current_scaling[0], yscale=(-1*current_scaling[1]), zscale=current_scaling[2])

    def inverse_z_axis(self):
        """Function that inverts the z axis of all coordinate data sets. It calls
        Fishualizer.scale_coordinates() with updated scale factors."""
        current_scaling = self.rec.spatial_scale_factor
        self.scale_coordinates(xscale=current_scaling[0], yscale=current_scaling[1], zscale=(-1*current_scaling[2]))

    def cluster_color(self):
        """Calls the Controls.ColoringDialog class that allows the user to color different clusters."""
        if len(self.cl_colors.keys()) == 0:
            colors = []
        else:  # add existing cluster color settings to ColoringDialog
            colors = [{'Dataset': v[0], 'Color': v[1], 'Alpha': v[3], 'Selected': v[2]} for k, v in self.cl_colors.items()]
        cl_win = ColoringDialog(self, sorted(list(self.rec.available_data)), data=colors, version='Cluster')
        #TODO: constrain self.rec.available_data in above line to 1D data sets
        if cl_win.exec_():
            self.draw_clusters(**cl_win.colors)

    def region_color(self):
        """Calls the Controls.ColoringDialog class that allows the user to color different regions."""
        region_list = [item for item in self.regionString.split(";") if item != ""]
        if len(self.reg_colors.keys()) == 0:
            region_cl = []
        else:
            region_cl = [{'Dataset': v[0], 'Color': v[1], 'Alpha': v[3], 'Selected': v[2]} for k, v in self.reg_colors.items()]
        region_colors =  ColoringDialog(self, region_list, data=region_cl, version='Region')
        if region_colors.exec_():
            self.hull_color_dict = region_colors.colors
            self.color_regions(**region_colors.colors)

    def color_regions(self, **kwargs):
        """"Function that colors the regions returned by region_color().

        Parameters:
        ------------
            **kwargs: dict containing two-tuples (name, hex_code) as values
                dict containing {cl_id: (name, hex_code)}
                    name: str
                        Should be the name of a region.
                    hex_code: str or None
                        Should be hex color code (e.g. '#17becdf')
                        Use None if you do not want to specify the color.
                    alpha: float
                        Transparency value
        """
        self.region_check_cb.setChecked(True)
        self.selected_regions_arr = np.zeros(len(self.selected_regions_arr)) # Clears it for other uses of the function

        for region_cl_id, region_cl_items in enumerate(kwargs.keys()):
            name = kwargs[region_cl_items][0]
            index = self.region_sel_cb.findText(name)
            if kwargs[region_cl_items][1] == None and (index - 1) not in self.region_spec_color.keys():  # if None is returned and index is specified already:
                region_color = list(self.current_hull_color[0:3])
            else:
                region_color = self.unhex(kwargs[region_cl_items])
            if index == 0:
                for region_index in self.region_indices[index]:  # color all division of 'Main divisions' individually
                    self.region_spec_color[region_index] = region_color
            else:
                self.region_spec_color[index - 1] = region_color
            self.reg_colors = kwargs  # save settings for when ColoringDialog is called again
            region_color.append(kwargs[region_cl_items][3])
            if int(kwargs[region_cl_items][2]) == 1:
                self.selected_regions_arr[index] = int(kwargs[region_cl_items][2]) # sets the value to 1 if selected

        if np.where(self.selected_regions_arr)[0].size == 0:  # if no regions were selected
            self._logger.info(f'No regions were selected, using {self.region_sel_cb.currentIndex()}.')
            self.selected_regions_arr[self.region_sel_cb.currentIndex()] = 1
        self.region_change()


    def draw_clusters(self, overlap='uniform', verbose=0, **kwargs):  # use **kwargs because unspecified number of clusters
        """Function to color code different clusters in same plot

        # TODO:
        Create legend (use dict names)

        kwargs: {'Dataset': v[0], 'Color': v[1], 'Alpha': v[3], 'Selected': v[2]}

        Parameters:
        ------------
            overlap: str
                How to handle overlapping clusters:
                    uniform: uniformly randomly choose
                    max: choose one with max connection
                    max_absolute: choose on with max absolute connection
            **kwargs: dict containing two-tuples (name, hex_code) as values
                dict containing {cl_id: (name, hex_code)}
                    name: str
                        Should be the name of a single data set available
                        All nonzero elements are used in cluster plot
                    hex_code: str or None
                        Should be hex color code (e.g. '#17becdf')
                        Use None if you do not want to specify the color.
                    alpha: float
                        Transparency value
        """
        ## check for double entries
        cluster_values = pd.DataFrame({})
        for cl_id, cl_name in enumerate(kwargs.keys()):
            name = kwargs[cl_name][0]
            if len(kwargs[cl_name]) >= 3:
                if kwargs[cl_name][2] == 0:  # if not selected
                    continue  # skip
            cluster_values[name] = getattr(self.rec, name)
        n_neurons_per_cluster_pre = (cluster_values != 0).sum(axis=0)

        ## Selection of cluster :
        if overlap == 'max' or overlap == 'max_absolute':
            if overlap == 'max':
                max_clusters = cluster_values.idxmax(axis=1)  # arg max per row
            elif overlap == 'max_absolute':
                max_clusters = cluster_values.abs().idxmax(axis=1)  # arg max per row of absolute values
            for i_neuron in range(len(cluster_values)):
                cluster_values.loc[i_neuron, cluster_values.columns != max_clusters[i_neuron]] = 0  # set all non-max clusters to 0, per neuron
        elif overlap == 'uniform':
            for i_neuron in range(len(cluster_values)):
                nz_clusters = cluster_values.iloc[i_neuron].nonzero()[0]  # relevant cluster (w/ nonzero value )
                if len(nz_clusters) > 0:  # at least 1 cluster
                    random_cl = int(np.random.choice(a=nz_clusters, size=1))  # select random cluster
                    cluster_values.loc[i_neuron, cluster_values.columns != cluster_values.columns[random_cl]] = 0  # set all other clustesrs to 0-
        assert len(np.where(cluster_values)[0]) == len(np.unique(np.where(cluster_values)[0])), 'ERROR: some neurons have been labelled twice in clusters '

        n_neurons_per_cluster_post = (cluster_values != 0).sum(axis=0)
        if verbose > 0:
            for ii in range(len(n_neurons_per_cluster_pre)):
                print(n_neurons_per_cluster_pre[ii], n_neurons_per_cluster_post[ii])

            print(f'Sum pre: {np.sum(n_neurons_per_cluster_pre)}, sum post: {np.sum(n_neurons_per_cluster_post)}')
        ## create colors to plot
        current_selection = set()  # use set to update sequentially in function
        c_colors = self.cmap(np.ones(self.rec.n_cells))
        for cl_id, cl_name in enumerate(kwargs.keys()):
            name = kwargs[cl_name][0]  # data set
            if len(kwargs[cl_name]) >= 3:
                if kwargs[cl_name][2] == 0:  # if not selected
                    continue  # skip
            if len(kwargs[cl_name]) == 4: # alpha set
                try:
                    alpha = kwargs[cl_name][3]  # alpha value
                    self._logger.debug(f'Alpha value: {alpha}')
                except ValueError:
                    alpha = 0.8
            else:  # default
                alpha = 0.8
            if kwargs[cl_name][1] is None:  # if no color is specified
                # color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color'] # create default color cycle
                # kwargs[cl_name][1] = color_cycle[cl_id]
                kwargs[cl_name][1] = "#%06x" % random.randint(0, 0xFFFFFF)  # random colour hex
                # hexcode = color_cycle[cl_id].lstrip('#')  # use cl_id-th element of default cycle
            # else:
            # neurons = np.where(getattr(self.rec, name))[0]  # find nonzero elements of static data sets
            neurons = np.where(cluster_values[name])[0]  # find nonzero of prepared cluster values
            current_selection.update(set(neurons))  # add to selected inds
            color_cl = self.unhex(kwargs[cl_name])
            color_cl.append(alpha)  # add transparency
            c_colors[neurons] = np.array(color_cl)  # set color for all neurons in cluster

        # one cannot call update_plot() here because that would override colors, so update explicitly:
        self.colors = c_colors
        self.cl_colors = kwargs
        # self._selected_inds = np.arange(self.rec.n_cells)
        self._selected_inds = np.array(list(current_selection))  # to np array for slicing
        self.update_behavior_plot()
        coords = self.rec.sel_frame
        self.Plot3D.setData(pos=coords[self._selected_inds], color=c_colors[self._selected_inds])  # only plot self._selected_inds
        self.switch_alpha_mode('translucent')  # set to translucent for better visibility (as long as not too many neurons are plotted)
        # self.hull_color[3] = 0.1
        # self.current_hull_color = [0.82, 0.82, 0.82, 0]

    def draw_all_hus(self, cluster_dict=None, overlap='max'):
        '''quick function to draw clusters of all 200 HUs'''
        if cluster_dict is None:
            cluster_dict = {}
            for ii in range(200):
                cl_name = f'hu_{str(ii).zfill(3)}'
                cluster_dict[cl_name] = [cl_name, "#%06x" % random.randint(0, 0xFFFFFF), 1, 0.8]
        else:
            for k, v in cluster_dict.items():  # check if all cluster names are in (static) data
                assert k in self.rec.available_data, f'cluster_dict not in correct format, {k} not in available data'
                assert len(v) <= 4, 'cluster_dict is not in correct format, each entry should have length <= 4 (see draw_cluster())'
        self.hu_colours = cluster_dict  # for saving
        self.draw_clusters(**cluster_dict, overlap=overlap, verbose=1)

    def draw_all_hus_from_path(self, cluster_dict_path='Content/',
                               cluster_dict_file='colours_2021-01-22.pkl',
                               overlap='max'):
        full_cluster_path = os.path.join(cluster_dict_path, cluster_dict_file)
        with open(full_cluster_path, 'rb') as f:
            cluster_dict = pickle.load(f)
        self._logger.debug(f'Cluster dictionary loaded from {full_cluster_path}')
        self.draw_all_hus(cluster_dict=cluster_dict, overlap=overlap)

    def draw_phase_map(self, amp_thresh=0, phase=None, amplitude=None):
        """Draw a phase map given phase & amplitude data. The phase is colored with
        the hsv cmap (same as Migault et al. pub), with the brightness scaled with
        the amplitude.
        If no phase or amplitude are given as input (i.e. there are None), the function
        will try to get them from self.rec

        Parameters:
        ------------
            amp_thresh: float, int
                amplitude threshold, only points with an amplitude greater than amp_thresh
                are selected for plotting (self._selected_inds is updated).
            phase: None or np.array with dims (n_cells,)
                phases of data.
            amplitude: None or np.array with dims (n_cells,)
                amplitudes of data.
        """
        if phase is None and 'phase_map' in self.rec.available_data:
            phase = self.rec.phase_map[:, 2]
            phase = np.mod(phase, 2 * np.pi)  # to ensure color coding is 0 to 2pi
        if amplitude is None and 'phase_map' in self.rec.available_data:
            amplitude = self.rec.phase_map[:, 1]
        if phase is None or amplitude is None:
            self._logger.warning('Phase map could not be drawn, because no phase or amplitude was found.')
            return None
        else:
            color_map = cm.ScalarMappable(cmap=cm.hsv)
            colors = color_map.to_rgba(phase)
            colors[:, 3] = amplitude / amplitude.max() * 3

            # colors = colors[:, :3]
            self._selected_inds = np.where(amplitude > amp_thresh)[0]

            self.update_behavior_plot()
            coords = self.rec.sel_frame
            self.colors = colors
            self.Plot3D.setData(pos=coords[self._selected_inds], color=colors[self._selected_inds])  # only plot self._selected_inds
            self.switch_alpha_mode('translucent')

    def draw_density_map(self, datapath=None, maptype='density_map', cut_off_threshold=None,
                         density_scale_factor=5, slice_density=1):
        """Add density map to self.view3D, the plot is created via utilities.create_density_map()

        # TODO: add to add_static_gui ?
        Parameters:
        ------------
            datapath: str (default to None)
                datapath where the density map (in .h5 file) is located. If None,
                it defaults to hard-coded directory.
            maptype: str; options: ('density_map', 'hard_threshold')
                type of plot
            cut_off_threshold: float or None
                threshold for cut-off of density map. If None, it defaults to 0 for map_type == 'density_map'
                and to 0,0005 for map_type == 'hard_threshold'
            density_scale_factor: float, int
                if maptype == 'density_map', the density is normalized to the max value, and
                the intensity (alpha) value is subsequently linearly scaled with the normalized
                density, multiplied by density_scale_factor
        """
        if datapath is None:
            cwd = os.getcwd()
            datapath = cwd + '/Volume_Example/grid_PE_6runs_minimum_density.h5'
        dp, resolution, min_coords = utilities.create_density_map(gridfile=datapath, map_type=maptype,
                                            den_threshold=cut_off_threshold, den_scale=density_scale_factor)

        # account for additive superposition as more slices are plotted
        dp[:,:,:,3] = dp[:,:,:,3]/slice_density

        #gl_value = {'glBlendFunc': (OpenGL.GL.GL_SRC_ALPHA, OpenGL.GL.GL_ONE_MINUS_SRC_ALPHA),
        #         'glBlendEquation': (OpenGL.GL.EXT.blend_minmax.GL_FUNC_ADD_EXT, ), OpenGL.GL.GL_BLEND: True}  # allows to view volume item from both lateral sides
        for vb in [self.view3D, self.view2D]:
            if self.density_plot in vb.items:
                vb.removeItem(self.density_plot)
        self.density_plot = gl.GLVolumeItem(dp, sliceDensity=slice_density,smooth=True)
        self.density_plot.scale(resolution[0], resolution[1], resolution[2])  # scale with resolution (because the voxels default to resolution (1,1,1))
        self.density_plot.translate(min_coords[0], min_coords[1], min_coords[2])  # translate to minimum of grid
        #self.density_plot.setGLOptions(gl_value)
        for vb in [self.view3D, self.view2D]:
            vb.addItem(self.density_plot)

    def toggle_density_plot(self):
        """Toggle show/hide of density plot."""
        self.show_density = not self.show_density
        self.save_checking(action=self.toggle_density_action, check=self.show_density)
        for vb in [self.view3D, self.view2D]:
            if self.density_plot in vb.items:
                if self.show_density:
                    self.density_plot.show()
                else:
                    self.density_plot.hide()
            vb.update()

    def add_zbrainatlas(self, filepath=None):
        """Load full ZBrainAtlas regions.
        Parameter:
        -----------
            filepath: str
                filepath where the ZBrainAtlas outlines are located.
        """
        if self._data_loaded: # because it is added to self.rec
            success = utilities.load_zbrain_regions(recording=self.rec, zbrainfile=filepath)
            if success:
                self.change_region_view_action.setEnabled(True)  # enable menu function to toggle between views
        else:
            self._logger.warning('Please load the main data before loading the ZBrainAtlas.')

    def change_region_view(self):
        """Toggle function to change region view between local labels and ZBrainAtlas labels."""
        if 'zbrainatlas_coordinates' not in self.rec.available_data:  # if zbrainatlas is not yet loaded
            self.add_zbrainatlas()  # load using default path (in ~/fishualizer/data/ZBrainAtlas folder)
        self._use_zbrainatlas_coords = not self._use_zbrainatlas_coords  # toggle
        if self.region_check_cb.isChecked():
            self.region_change()  # update

        self.statusBar().showMessage("Changed region view")

    def toggle_xy_outline(self):
        '''Toggle xy outline box'''
        if self._data_loaded:
            if self.xy_outline_action.isChecked(): # if selected to True; draw new outline
                self.draw_xy_outline()
            else:  # if deselected (to False); remove current outline
                if self.outline_plot[2] in self.view3D.items:
                    self.view3D.removeItem(self.outline_plot[2])  # the outline is indexed by dim_other, remove previous outline

    def draw_xy_outline(self, view=None):
        """Draw outline of xy plane (by calling draw_outline())"""
        if self._data_loaded:
            if view is None:
                view = self.view3D
            coords = self.rec.sel_frame
            len_coords = coords.max(axis=0) - coords.min(axis=0)
            if len_coords[1] > len_coords[0]:  # match long and short dimensions
                self.draw_outline(dim_short=0, dim_long=1, view=view)
            else:
                self.draw_outline(dim_short=1, dim_long=0, view=view)

    def toggle_xz_outline(self):
        '''Toggle xz outline box'''
        if self._data_loaded:
            if self.xz_outline_action.isChecked():
                self.draw_xz_outline()
            else:
                if self.outline_plot[1] in self.view3D.items:
                    self.view3D.removeItem(self.outline_plot[1])  # the outline is indexed by dim_other, remove previous outline

    def draw_xz_outline(self, view=None):
        """Draw outline of xz plane (by calling draw_outline())"""
        if self._data_loaded:
            if view is None:
                view = self.view3D
            coords = self.rec.sel_frame
            len_coords = coords.max(axis=0) - coords.min(axis=0)
            if len_coords[2] > len_coords[0]:
                self.draw_outline(dim_short=0, dim_long=2, view=view)
            else:
                self.draw_outline(dim_short=2, dim_long=0, view=view)

    def toggle_yz_outline(self):
        '''Toggle yz outline box'''
        if self._data_loaded:
            if self.yz_outline_action.isChecked():
                self.draw_yz_outline()
            else:
                if self.outline_plot[0] in self.view3D.items:
                    self.view3D.removeItem(self.outline_plot[0])  # the outline is indexed by dim_other, remove previous outline

    def draw_yz_outline(self, view=None):
        """Draw outline of yz plane (by calling draw_outline())"""
        if self._data_loaded:
            if view is None:
                view = self.view3D
            coords = self.rec.sel_frame
            len_coords = coords.max(axis=0) - coords.min(axis=0)
            if len_coords[2] > len_coords[1]:
                self.draw_outline(dim_short=1, dim_long=2, view=view)
            else:
                self.draw_outline(dim_short=2, dim_long=1, view=view)

    def draw_xy_region_outline(self, view=None, zbrain_region_ind=0, split_into_two=False, side='left'):
        """Draw outline of xy plane (by calling draw_outline())"""
        if self._use_zbrainatlas_coords is False:
            print('Please select zbrain coordinates to use this function (draw_main_region_outlines())')
            self._logger.warning('Please select zbrain coordinates to use this function (draw_main_region_outlines())')
            return

        assert type(zbrain_region_ind) == int
        zbrain_region_ind = np.array([zbrain_region_ind])

        if self._data_loaded:
            if view is None:
                view = self.view3D
            coords = self.rec.sel_frame
            len_coords = coords.max(axis=0) - coords.min(axis=0)
            if len_coords[1] > len_coords[0]:  # match long and short dimensions
                self.draw_outline(dim_short=0, dim_long=1, view=view, zbrain_region_inds=zbrain_region_ind,
                                  split_into_two=split_into_two, lateral_side=side)
            else:
                self.draw_outline(dim_short=1, dim_long=0, view=view, zbrain_region_inds=zbrain_region_ind,
                                  split_into_two=split_into_two, lateral_side=side)

    def draw_xz_region_outline(self, view=None, zbrain_region_ind=0, split_into_two=False, side='left'):
        """Draw outline of xz plane (by calling draw_outline())"""
        if self._use_zbrainatlas_coords is False:
            print('Please select zbrain coordinates to use this function (draw_main_region_outlines())')
            self._logger.warning('Please select zbrain coordinates to use this function (draw_main_region_outlines())')
            return

        assert type(zbrain_region_ind) == int
        zbrain_region_ind = np.array([zbrain_region_ind])

        if self._data_loaded:
            if view is None:
                view = self.view3D
            coords = self.rec.sel_frame
            len_coords = coords.max(axis=0) - coords.min(axis=0)
            if len_coords[2] > len_coords[0]:
                self.draw_outline(dim_short=0, dim_long=2, view=view, zbrain_region_inds=zbrain_region_ind,
                                  split_into_two=split_into_two, lateral_side=side)
            else:
                self.draw_outline(dim_short=2, dim_long=0, view=view, zbrain_region_inds=zbrain_region_ind,
                                  split_into_two=split_into_two, lateral_side=side)

    def draw_yz_region_outline(self, view=None, zbrain_region_ind=0, split_into_two=False, side='left'):
        """Draw outline of yz plane (by calling draw_outline())"""
        if self._use_zbrainatlas_coords is False:
            print('Please select zbrain coordinates to use this function (draw_main_region_outlines())')
            self._logger.warning('Please select zbrain coordinates to use this function (draw_main_region_outlines())')
            return

        assert type(zbrain_region_ind) == int
        zbrain_region_ind = np.array([zbrain_region_ind])

        if self._data_loaded:
            if view is None:
                view = self.view3D
            coords = self.rec.sel_frame
            len_coords = coords.max(axis=0) - coords.min(axis=0)
            if len_coords[2] > len_coords[1]:
                self.draw_outline(dim_short=1, dim_long=2, view=view, zbrain_region_inds=zbrain_region_ind,
                                  split_into_two=split_into_two, lateral_side=side)
            else:
                self.draw_outline(dim_short=2, dim_long=1, view=view, zbrain_region_inds=zbrain_region_ind,
                                  split_into_two=split_into_two, lateral_side=side)

    def draw_main_region_outlines(self, main_regions_inds=None, remove_all_other_regions=True):
        '''Plot main regions, if main_regions_inds == None, then plot self.main_zbrain_divisions
        assume xy view in view3D and yz view in view2D'''
        if self._use_zbrainatlas_coords is False:
            print('Please select zbrain coordinates to use this function (draw_main_region_outlines())')
            self._logger.warning('Please select zbrain coordinates to use this function (draw_main_region_outlines())')
            return
        if main_regions_inds is None:
            main_regions_inds = self.main_zbrain_divisions
        if remove_all_other_regions:
            self.remove_all_region_outlines()

        for reg_ind in main_regions_inds:
            for side in ['left', 'right']:
                self.draw_xy_region_outline(view=self.view3D, zbrain_region_ind=int(reg_ind),
                                            split_into_two=True, side=side)
                self.draw_yz_region_outline(view=self.view2D, zbrain_region_ind=int(reg_ind),
                                            split_into_two=True, side=side)

    def remove_region_outline(self, view=None, remove_regions_ind=0, dim_other=2):
        '''Remove 1 particular region outline.
        dim_other = dimsension that is not used (0=x, 1=y, 2=z).'''
        assert type(remove_regions_ind) == int
        if view is None:  # default is to remove region outline in both vbs
            view_list = [self.view2D, self.view3D]
        else:
            view_list = [view]  # make a list for compatibility with loop
        for side in ['left', 'right', 'both']:
            if remove_regions_ind in self.region_plot[dim_other][side].keys():
                curr_reg_outline = self.region_plot[dim_other][side][remove_regions_ind]
                for vb in view_list:
                    if curr_reg_outline in vb.items:
                        vb.removeItem(curr_reg_outline)
                if curr_reg_outline not in self.view2D.items and curr_reg_outline not in self.view3D.items:  # if not present anymore in either vb, delete
                    del self.region_plot[dim_other][side][remove_regions_ind]

    def remove_all_region_outlines(self, view=None):
        '''Remove all region outlines (not full data outlines). if view == None, remove from both vbs'''
        for dim_other in [0, 1, 2]:  # all 3 perspectives
            list_present_regions = set(sum([list(self.region_plot[dim_other][side].keys()) for side in ['left', 'right', 'both']], []))
            for present_reg in list_present_regions: # loop through currently present regions
                self.remove_region_outline(remove_regions_ind=int(present_reg), dim_other=dim_other)

    def show_top_side_outlines(self):
        '''Draw xy outline for top view & yz outline for side view '''
        if self.bool_topside_outline is False:  # if currently not drawn: (or something else draw)n
            if self.xy_outline_action.isChecked() is False: # if not yet drawn, draw new outline
                self.xy_outline_action.setChecked(True)
                self.toggle_xy_outline()  # this uses the current check
        elif self.xy_outline_action.isChecked():  # if currently drawn, remove:
                self.xy_outline_action.setChecked(False)
                self.toggle_xy_outline()
        if self.outline_plot[0] in self.view2D.items:
            self.view2D.removeItem(self.outline_plot[0])
        else:
            self.draw_yz_outline(view=self.view2D)
        self.bool_topside_outline = not self.bool_topside_outline

    def draw_outline(self, dim_short=0, dim_long=1, line_width=2, moving_average=True,
                     select_method='minmax', param_alpha=0.005, view=None,
                     zbrain_region_inds=None, split_into_two=False, lateral_side='left'):
        """Draw outline of the fish. Use current coordinates or zbrainatlas_coordinates,
        depending on self._use_zbrainatlas_coords
        The outline is drawn for the two dimensional plane (dim_short, dim_long),
        and is saved by the complementary dimension index (dim_other) in self.outline_plot[dim_other].

        Two methods can be used to create the outlines; minmax or alpha_shapeself.
        minmax is faster, but less accurate. The default (by menu access) is currently
        set to 'minmax' for speed reasons, the 'alpha_shape' (for better figures)
        can be accessed from the console.
        Alternatively, one could also skip every n coordinates to speed up the process,
        but this is currently not implemented because the alpha_shaping will not be
        instantaneous anyway. This might however be nice for less powerful computers. (# TODO)

        Parameters:
        ------------
            dim_short: int (0, 1, 2)
                short dimension of outline
            dim_long: int (0, 1, 2).remove(dim_short)
                long dimension of outline (this split up is better for min/max outlining)
            line_width: float/int?
                width of line to draw
            moving_average: bool
                whether to compute the moving average (per 3 points) of the line (for smoothing)
            select_method: str ('minmax', 'alpha_shape')
                which method to use for selecting the points that make the outline of the fish
            param_alpha: float
                alpha value to use in case select_method == 'alpha_shape'
            view: viewbox
                which viewbox to draw in
            zbrain_region_inds: np array, or list
                which regions to plot (MUST be a np array or list)
        """
        if view is None:
            view = self.view3D
        if self._use_zbrainatlas_coords:  # use full zbrainatlas
            if zbrain_region_inds is None:
                zbrain_region_inds = self.main_zbrain_divisions  # default = show full outline
                bool_outline = True
            else:
                zbrain_region_inds = np.array(zbrain_region_inds)
                assert len(zbrain_region_inds) == 1, 'you can only plot 1 region the time, except for full outline'
                bool_outline = False
            neurons_draw = np.array([])
            for iloop, ir in enumerate(zbrain_region_inds):  # use outlines from main divisions
                tmp_sel = np.where(self.rec.zbrainatlas_regions == ir)[0]
                neurons_draw = np.concatenate((neurons_draw, tmp_sel.copy()), axis=0)  # add to neurons_draw
            neurons_draw = neurons_draw.astype('int')
            coordinates = self.rec.zbrainatlas_coordinates[neurons_draw, :]  # get coordinates
            mid_x = np.median(self.rec.zbrainatlas_coordinates[:, 0])
        else:
            coordinates = self.rec.sel_frame[self._selected_inds, :]  # get coordinates of current coordinate selection
            mid_x = np.median(self.rec.sel_frame[:, 0])
            bool_outline = True
        if split_into_two:
            if lateral_side == 'left':
                left_selection = np.where(coordinates[:, 0] >= mid_x)[0]
                coordinates = coordinates[left_selection, :]
            elif lateral_side == 'right':
                right_selection = np.where(coordinates[:, 0] < mid_x)[0]
                coordinates = coordinates[right_selection, :]
        else:
            lateral_side = 'both'
        all_dims = {0, 1, 2}
        all_dims.remove(dim_short)
        all_dims.remove(dim_long)
        dim_other = int(list(all_dims)[0])  # the dim that is not dim_short or dim_long
        if bool_outline:  # remove old outlines
            if self.outline_plot[dim_other] in view.items:
                view.removeItem(self.outline_plot[dim_other])  # the outline is indexed by dim_other, remove previous outline
        else:  # remove old region outlines
            if zbrain_region_inds[0] in self.region_plot[dim_other][lateral_side].keys():  # if this combination of regions was plot already, remove
                view.removeItem(self.region_plot[dim_other][lateral_side][zbrain_region_inds[0]])

        if select_method == 'minmax':
            dlong_arr = np.linspace(coordinates[:, dim_long].min(), coordinates[:, dim_long].max(), 100)  # array to create outlines with
            shortmin = np.zeros((len(dlong_arr), 3)) - 1  # assuming -1 is not possible
            shortmax = np.zeros((len(dlong_arr), 3)) - 1
            prev_dlong = 0
            for iloop, dlong in enumerate(dlong_arr):  # loop through array of long dimension values
                current_sel =np.where(np.array([coordinates[:, dim_long] < dlong]) & np.array([coordinates[:, dim_long] > prev_dlong]))[1]
                if len(current_sel) > 0:
                    shortmin[iloop, dim_short] = coordinates[current_sel, dim_short].min(axis=0)  # set to min
                    shortmin[iloop, dim_long] = dlong  # set to current element of long dimension value array
                    shortmax[iloop, dim_short] = coordinates[current_sel, dim_short].max(axis=0)
                    shortmax[iloop, dim_long] = dlong
                    prev_dlong = dlong
            shortmin[:, dim_other] = np.zeros(shortmin.shape[0])  # set non-considered dim to 0 (is translated later)
            shortmax[:, dim_other] = np.zeros(shortmax.shape[0])

            k = 0
            while k <= shortmin.shape[0]-1:  # loop to remove all non-assigned points of outline
                if shortmin[k, dim_short] == -1:
                    shortmin = np.delete(shortmin, k, 0)
                    shortmax = np.delete(shortmax, k, 0)
                    continue
                else:
                    k += 1
            if moving_average:  # compute moving average per 3 points to smooth
                for ii in range(1, shortmin.shape[0]-1):
                    shortmin[ii, dim_short] = (1 / 3) * (shortmin[ii - 1, dim_short] + shortmin[ii, dim_short] + shortmin[ii + 1, dim_short])
                for jj in range(1, shortmax.shape[0]-1):
                    shortmax[jj, dim_short] = (1 / 3) * (shortmax[jj - 1, dim_short] + shortmax[jj, dim_short] + shortmax[jj + 1, dim_short])
            line_points = np.concatenate((shortmin, np.flipud(shortmax)), axis=0)  # importantly, line_points need to be sequentially sorted (hence the np.flip)

        elif select_method == 'alpha_shape':
            dim_used = np.array(list({dim_short, dim_long}))  # sorted(!) np array of used dimensions
            outline_test = alpha_shape.alpha_fish_new(points=coordinates[:, dim_used], alpha=param_alpha)  # get 2D alpha shape from alpha_shape.py file
            outline_points = outline_test[0]  # outline points
            for kk, vv in outline_test.items():  # unpack outline points
                if kk > 0:
                    outline_points = np.concatenate((outline_points, vv), axis=0)

            center_test = coordinates[:, dim_used].mean(axis=0)  # center of fish in 2D plane
            ph_test = np.arctan((center_test[1] - outline_points[:, 1]) /( center_test[0] - outline_points[:, 0]))  # determine phase of every point w.r.t. center in 2D plane
            ph_test[np.where(outline_points[:,0] < center_test[0])[0]] += np.pi  # add pi to left half of points (because arctan maps everything to half the phase interval)
            order = np.argsort(ph_test)  # argsort the phases
            tmp_line_points = outline_points[order, :]  # order all outline points (because a line is draw sequentially)
            # Note: sorting by phase works quite well, but is not guaranteed to be perfect..
            if dim_other == 0:  # three cases to add back third dimension
                line_points = np.concatenate((np.zeros((tmp_line_points.shape[0], 1)), tmp_line_points), axis=1)
            if dim_other == 1:
                line_points = np.concatenate((tmp_line_points[:, 0, np.newaxis], np.zeros((tmp_line_points.shape[0], 1))), axis=1)
                line_points = np.concatenate((line_points, tmp_line_points[:, 1, np.newaxis]), axis=1)
            if dim_other == 2:
                line_points = np.concatenate((tmp_line_points, np.zeros((tmp_line_points.shape[0], 1))), axis=1)
            if moving_average:  # compute moving average per 3 points to smooth
                for ii in range(1, line_points.shape[0]-1):
                    line_points[ii, :] = (1 / 3) * (line_points[ii - 1, :] + line_points[ii, :] + line_points[ii + 1, :])

        line_points = np.vstack((line_points, line_points[0, :]))  # connect first and last point to make full circumvention
        # line_colors = np.zeros((len(line_points), 4)) + 255  # use for dotted line
        # line_colors[::2, 3] = 1   # make even points visible
        # line_colors[1::2, 3] = 0  # make uenven points invisible (together they create a dotted line 1)
        if bool_outline:
            self.outline_plot[dim_other] = gl.GLLinePlotItem(pos=line_points, width=line_width)#, color=line_colors,  antialias=False, mode='lines')  # create GLLinePlotItem to plot outline
        else:
            self.region_plot[dim_other][lateral_side][zbrain_region_inds[0]] = gl.GLLinePlotItem(pos=line_points, width=line_width)#, color=line_colors,  antialias=False, mode='lines')  # create GLLinePlotItem to plot region outline
        meancoords = self.rec.sel_frame.mean(axis=0)
        translate_xyz = np.zeros(3)
        translate_xyz[dim_other] = meancoords[dim_other]  # translate to mean of current coords
        if bool_outline:
            self.outline_plot[dim_other].translate(translate_xyz[0], translate_xyz[1], translate_xyz[2])
            view.addItem(self.outline_plot[dim_other])
        else:
            self.region_plot[dim_other][lateral_side][zbrain_region_inds[0]].translate(translate_xyz[0], translate_xyz[1], translate_xyz[2])
            view.addItem(self.region_plot[dim_other][lateral_side][zbrain_region_inds[0]])

    def reset_view(self):
        """Function that resets the selection of neurons."""
        if self.rec.path is not None:
            self._selected_inds = np.arange(self.rec.n_cells)  # reset to all indices
            self.neuron_check_cb.setChecked(False)
            self.labelled_cb.setChecked(False)
            self.filter_dict = {}  # delete all filters
            self.reset_scrollbars()
            self.update_plot()

    def copy_clipboard(self):
        """ Toggle the copying of cell coordinates to the clipboard, when selecting cells
        see select_cell for details
        """
        self._copy_clipboard = not self._copy_clipboard

    def reset_scrollbars(self):
        """Reset the scrollbars to their default values."""
        self._limit_start_abs = np.min(self.rec.sel_frame, axis=0)
        self._limit_stop_abs = np.max(self.rec.sel_frame, axis=0)
        self._limit_start = np.min(self.rec.sel_frame, axis=0)
        self._limit_stop = np.max(self.rec.sel_frame, axis=0)

        self._dimsel = 0
        self.x_action.setChecked(True)
        self._limit_slider_nsteps = [100, 100, 0]
        self._limit_slider_values = [0, 0, 0]
        for cInd in [0, 1]:
            self._limit_slider_values[cInd] = np.linspace(self._limit_start_abs[cInd], self._limit_stop_abs[cInd],
                                                          self._limit_slider_nsteps[cInd])

        self._limit_slider_nsteps[2] = np.minimum(np.unique(self.rec.sel_frame[:, 2]).shape[0], 100)  # set number of z intervals to number of layers, maxed at 100
        if self._limit_slider_nsteps[2] == np.unique(self.rec.sel_frame[:, 2]).shape[0]:  # if nsteps equal to number of layers, use layer values
            self._limit_slider_values[2] = np.unique(self.rec.sel_frame[:, 2])
        else:  # else interpolate 100 values between min and max
            self._limit_slider_values[2] =  np.linspace(self._limit_start_abs[2], self._limit_stop_abs[2],
                                                          self._limit_slider_nsteps[2])
        self.set_limit_slider_pos()

        # Z slider intially selected
        self.limitz()

    def add_scale_bar(self, start=[0.05, 0.1, 0], end=[0.05, 0.2, 0], bar_width=5,
                      sb_obj=None, view=None):
        """Add a scale bar to the plot.

        Scale bar is stored in self.scale_bar and added to self.view3D. This function
        overwrites previous scale bar if it exists.

        Parameters:
        ------------
            start: list of 3 floats
                start coordinates [x, y, z] of scale bar
            end: list of 3 floats
                end coordinates [x, y, z] of scale bar
            bar_width: float/int
                width of scale bar
            sb_obj: ID of scale bar
                can be used to either replace or add new one
            view: viewbox
        """
        if view is None:
            view = self.view3D
        if sb_obj is None:
            sb_obj = self.scale_bar

        if sb_obj in view.items:
            view.removeItem(sb_obj)

        points = np.array([start, end])
        size = np.abs(np.array(end) - np.array(start))
        sb_obj = gl.GLLinePlotItem(pos=points, width=bar_width)
        self.scale_bar = sb_obj
        view.addItem(sb_obj)
        self._logger.info(f'Scale bar added. Size x {size[0]}, y {size[1]}, z {size[2]}')

    def remove_scale_bar(self, view=None):
        if view is None:
            vb_list = [self.view3D, self.view2D]
        else:
            vb_list = [view]
        for vb in vb_list:
            if self.scale_bar in vb.items:
                vb.removeItem(self.scale_bar)

    def toggle_reference_axes(self):
        """ Remove all items other than data, scale bar, outline or region hulls for
        figure.
        """
        if self.bool_show_ref_axes:  # if the ref axes are currently shown, hide:
            self.reference_x_axis.hide()
            self.reference_y_axis.hide()
            self.reference_z_axis.hide()
        else:  # if currently hidden, show:
            self.reference_x_axis.show()
            self.reference_y_axis.show()
            self.reference_z_axis.show()
        self.bool_show_ref_axes = not self.bool_show_ref_axes  # update

    def remove_regions(self):
        """Function that removes all region hulls. It is called when the region_check_cb is unchecked."""
        for regions in self.hulls:
            for region in regions:
                for vb in [self.view3D, self.view2D]:
                    if region in vb.items:
                        vb.removeItem(region)

    def reset_structural_data_plot(self):
        """ This function gets called when the value of the scrollbar is changed.
        It removes the old plot and plots the new one
        """
        self.view3D.removeItem(self.structural_data_plot)
        self.structural_data_le.setText(str(self.structural_data_sb.value()+1))
        self.plot_structural_data()

    def unhex(self, value):
        """Function that unhexes the 1-th value of a list.

        Parameters:
        ------------
            value: list, where value[1] is to be unhexed.

        returns
        -------
            RGBb: Red, Green, Blue value of the given hexcode

        """
        hexcode = value[1].lstrip('#')  # use pre-specified color code
        RGBb = list(float(int(hexcode[i:i+2], 16) / 256) for i in (0, 2 ,4))  # transform hex code to float rgb
        return RGBb

def handle_exception(exc_type, exc_value, exc_traceback):
    """Handle uncaught exceptions and print in logger."""
    logger = logging.getLogger('Fishlog')
    if issubclass(exc_type, KeyboardInterrupt):
        sys.__excepthook__(exc_type, exc_value, exc_traceback)
        return
    logger.error("Uncaught exception", exc_info=(exc_type, exc_value, exc_traceback))


if __name__ == '__main__':
    # Testing the application
    qApp = QtWidgets.QApplication(sys.argv)
    # Create a GUI instance
    window = Fishualizer(sys.argv)
    window.showMaximized()
    window.setGeometry(40, 20, 1500, 1200)
    # Wait for it to be finished
    sys.exit(window.kernel.ipkernel.start())
    sys.exit(qApp.exec_())
