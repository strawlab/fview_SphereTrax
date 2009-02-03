from __future__ import division
import pkg_resources
import os.path
import sys
import copy
import threading
import Queue
import wx
import wx.lib.plot
import wx.xrc as xrc
import numpy
import numpy.numarray as nx
import time
import matplotlib
import matplotlib.cm
import matplotlib.collections
matplotlib.use('WXAgg')
from matplotlib.backends.backend_wxagg import FigureCanvasWxAgg
from matplotlib.figure import Figure
from optic_flow import get_optic_flow
from optic_flow import get_ang_vel
from optic_flow import get_hfs_rates
from scipy.optimize import fmin
import socket  # for udp
import struct
import adskalman.adskalman as adskalman

RESFILE = pkg_resources.resource_filename(__name__,"fview_SphereTrax.xrc") # trigger extraction
RES = xrc.EmptyXmlResource()
RES.LoadFromString(open(RESFILE).read())

# IDs for frame, panel, notebook and subpanels
SphereTrax_FRAME = "SphereTrax_FRAME"
SphereTrax_PANEL = "SphereTrax_PANEL"
SphereTrax_NOTEBOOK = "SphereTrax_NOTEBOOK"
Optic_Flow_PANEL = "Optic_Flow_PANEL"
Find_Sphere_PANEL = "Find_Sphere_PANEL"
Tracking_PANEL = "Tracking_PANEL"
Closed_Loop_PANEL = "Closed_Loop_PANEL"

# IDs for opticflow panel controls
Optic_Flow_Enable_CHECKBOX = "Optic_Flow_Enable_CHECKBOX"
Num_Row_SPINCTRL = "Num_Row_SPINCTRL"
Num_Col_SPINCTRL = "Num_Col_SPINCTRL"
Window_Size_SPINCTRL = "Window_Size_SPINCTRL"
Poll_Interval_TEXTCTRL = "Poll_Interval_TEXTCTRL"
Horiz_Space_SLIDER = "Horiz_Space_SLIDER"
Horiz_Position_SLIDER = "Horiz_Position_SLIDER"
Vert_Space_SLIDER = "Vert_Space_SLIDER"
Vert_Position_SLIDER = "Vert_Position_SLIDER"

# IDs for find sphere panel controls
Find_Sphere_Image_PANEL = "Find_Sphere_Image_PANEL"
Grab_Image_BUTTON = "Grab_Image_BUTTON"
Delete_Points_BUTTON = "Delete_Points_BUTTON"
Find_Sphere_BUTTON = "Find_Sphere_BUTTON"

# IDs for tracking panel controls
Tracking_Enable_CHECKBOX = "Tracking_Enable_CHECKBOX"
Head_Rate_Plot_PANEL = "Head_Rate_Plot_PANEL"
Forw_Rate_Plot_PANEL = "Forw_Rate_Plot_PANEL"
Side_Rate_Plot_PANEL = "Side_Rate_Plot_PANEL"

# Default values for optic flow panel
OPTIC_FLOW_DEFAULTS = {
    'opticflow_enable' : False,
    'poll_int' : 0.02,
    'wnd' : 20,
    'num_row' :  2,
    'num_col' : 2,
    'horiz_space': 0.5,
    'horiz_pos' : 0.5,
    'vert_space' : 0.5,
    'vert_pos' :0.5
    }

# Defaults values for tracking panel
TRACKING_DEFAULTS = {
    'tracking_enable' : False,
    'tracking_plot_poll_int' : 500, # ms
    'tracking_plot_length' : 10, # secs
    'tracking_plot_line_color' : 'blue',
    'tracking_plot_line_width' : 1,

    }

# Sphere orientation vectors
SPHERE_U_VEC = numpy.array([0.0,-1.0, 0.0])
SPHERE_F_VEC = numpy.array([0.0, 0.0,1.0])
SPHERE_S_VEC = numpy.array([1.0, 0.0, 0.0])
SPHERE_ORIENTATION = SPHERE_U_VEC, SPHERE_F_VEC, SPHERE_S_VEC

class SphereTrax_Class:
    def __init__(self,wx_parent):
        self.wx_parent = wx_parent

        # Basic optic flow data
        self.timestamp_last = 0.0
        self.line_list = []
        self.dpix_list = []
        self.lag_buf = LagList(1)
        self.lock = threading.Lock()
        self.tracking_plot_queue = Queue.Queue(0)
        self.tracking_plot_data = []

        # Counter for updating tracking plot queue
        self.tracking_plot_cntr = 0

        # UDP settings
        self.runthread_remote_host = None
        self.remote_host_lock = threading.Lock()
        self.remote_host_changed = threading.Event()
        self.send_over_ip = threading.Event()
        self.remote_host = None
        self.sockobj = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

        # Kalman filter
        ss=6
        os=3
        A = numpy.array([[1,0,0,1,0,0],
                         [0,1,0,0,1,0],
                         [0,0,1,0,0,1],
                         [0,0,0,1,0,0],
                         [0,0,0,0,1,0],
                         [0,0,0,0,0,1]],
                        dtype=numpy.float)
        C = numpy.array([[1,0,0,0,0,0],
                         [0,1,0,0,0,0],
                         [0,0,1,0,0,0]],
                        dtype=numpy.float)
        Q = 0.25*numpy.eye(ss)
        R = 5.0*numpy.eye(os)
        init_x = numpy.array([0,0,0,0,0,0],dtype=numpy.float)
        init_P = 1.0*numpy.eye(ss)

        self.kalman = adskalman.KalmanFilter(A,C,Q,R,init_x,init_P)
        self.kalman_isinitial = 1


        # Initialize GUI
        self.main_frame_init()
        self.optic_flow_panel_init()
        self.find_sphere_panel_init()
        self.tracking_panel_init()
        self.closed_loop_panel_init()


    def main_frame_init(self):
        """
        Initializes the main frame and notebook part of the GUI
        """
        self.frame = RES.LoadFrame(self.wx_parent,SphereTrax_FRAME)
        self.panel = xrc.XRCCTRL(self.frame,SphereTrax_PANEL)
        self.notebook = xrc.XRCCTRL(self.panel,SphereTrax_NOTEBOOK)

    def optic_flow_panel_init(self):
        """
        Initializes the optic flow notebook page of the GUI
        """
        self.optic_flow_panel = xrc.XRCCTRL(self.notebook,Optic_Flow_PANEL)

        # Setup optic flow panel controls
        self.optic_flow_enable_box = xrc.XRCCTRL(self.optic_flow_panel,
                                                 Optic_Flow_Enable_CHECKBOX)
        self.num_row_spin_ctrl = xrc.XRCCTRL(self.optic_flow_panel,Num_Row_SPINCTRL)
        self.num_col_spin_ctrl = xrc.XRCCTRL(self.optic_flow_panel,Num_Col_SPINCTRL)
        self.window_size_spin_ctrl = xrc.XRCCTRL(self.optic_flow_panel,Window_Size_SPINCTRL)
        self.poll_int_text_ctrl = xrc.XRCCTRL(self.optic_flow_panel,Poll_Interval_TEXTCTRL)
        self.horiz_space_slider = xrc.XRCCTRL(self.optic_flow_panel,Horiz_Space_SLIDER)
        self.horiz_position_slider = xrc.XRCCTRL(self.optic_flow_panel,Horiz_Position_SLIDER)
        self.vert_space_slider = xrc.XRCCTRL(self.optic_flow_panel,Vert_Space_SLIDER)
        self.vert_position_slider = xrc.XRCCTRL(self.optic_flow_panel,Vert_Position_SLIDER)

        # Setup events for optic flow panel controls
        wx.EVT_CHECKBOX(self.optic_flow_enable_box, xrc.XRCID(Optic_Flow_Enable_CHECKBOX),
                        self.on_optic_flow_enable)
        wx.EVT_SPINCTRL(self.num_row_spin_ctrl, xrc.XRCID(Num_Row_SPINCTRL),
                        self.on_num_row_spin_ctrl)
        wx.EVT_SPINCTRL(self.num_col_spin_ctrl, xrc.XRCID(Num_Col_SPINCTRL),
                        self.on_num_col_spin_ctrl)
        wx.EVT_SPINCTRL(self.window_size_spin_ctrl, xrc.XRCID(Window_Size_SPINCTRL),
                        self.on_window_size_spin_ctrl)
        wx.EVT_TEXT_ENTER(self.poll_int_text_ctrl, xrc.XRCID(Poll_Interval_TEXTCTRL),
                          self.on_poll_int_text_enter)
        wx.EVT_SLIDER(self.horiz_space_slider, xrc.XRCID(Horiz_Space_SLIDER),
                      self.on_horiz_space_slider)
        wx.EVT_SLIDER(self.horiz_position_slider, xrc.XRCID(Horiz_Position_SLIDER),
                      self.on_horiz_position_slider)
        wx.EVT_SLIDER(self.vert_space_slider, xrc.XRCID(Vert_Space_SLIDER),
                      self.on_vert_space_slider)
        wx.EVT_SLIDER(self.vert_position_slider, xrc.XRCID(Vert_Position_SLIDER),
                      self.on_vert_position_slider)

        # Set default values for optic flow panel
        self.optic_flow_enable = OPTIC_FLOW_DEFAULTS['opticflow_enable']
        self.optic_flow_enable_box.SetValue(self.optic_flow_enable)

        self.poll_int = OPTIC_FLOW_DEFAULTS['poll_int']
        self.poll_int_text_ctrl.SetValue(str(self.poll_int))

        self.wnd = OPTIC_FLOW_DEFAULTS['wnd']
        self.window_size_spin_ctrl.SetValue(self.wnd)

        self.num_row = OPTIC_FLOW_DEFAULTS['num_row']
        self.num_row_spin_ctrl.SetValue(self.num_row)

        self.num_col = OPTIC_FLOW_DEFAULTS['num_col']
        self.num_col_spin_ctrl.SetValue(self.num_col)

        self.horiz_space = OPTIC_FLOW_DEFAULTS['horiz_space']
        set_slider_value(self.horiz_space_slider, self.horiz_space)

        self.horiz_pos = OPTIC_FLOW_DEFAULTS['horiz_pos']
        set_slider_value(self.horiz_position_slider, self.horiz_pos)

        self.vert_space = OPTIC_FLOW_DEFAULTS['vert_space']
        set_slider_value(self.vert_space_slider, self.vert_space)

        self.vert_pos = OPTIC_FLOW_DEFAULTS['vert_pos']
        set_slider_value(self.vert_position_slider, self.vert_pos)



    def find_sphere_panel_init(self):
        """
        Initializes find sphere notebook page of the GUI
        """
        self.find_sphere_panel = xrc.XRCCTRL(self.notebook,Find_Sphere_PANEL)

        # Set up controls for find sphere panel
        self.find_sphere_image_panel = xrc.XRCCTRL(self.find_sphere_panel,
                                                   Find_Sphere_Image_PANEL)
        self.grab_image_button = xrc.XRCCTRL(self.find_sphere_panel,Grab_Image_BUTTON)
        self.delete_points_button = xrc.XRCCTRL(self.find_sphere_panel, Delete_Points_BUTTON)
        self.find_sphere_button = xrc.XRCCTRL(self.find_sphere_panel, Find_Sphere_BUTTON)

        # Set up events
        wx.EVT_BUTTON(self.grab_image_button, xrc.XRCID(Grab_Image_BUTTON),
                      self.on_grab_image_button)
        wx.EVT_BUTTON(self.delete_points_button, xrc.XRCID(Delete_Points_BUTTON),
                      self.on_delete_points_button)
        wx.EVT_BUTTON(self.find_sphere_button, xrc.XRCID(Find_Sphere_BUTTON),
                      self.on_find_sphere_button)

        # Setup image plot panel -- copying Andrew
        sizer = wx.BoxSizer(wx.VERTICAL)
        self.plot_panel = PlotPanel(self.find_sphere_image_panel)

        # wx boilerplate
        sizer.Add(self.plot_panel, 1, wx.EXPAND )# |wx.TOP| wx.LEFT |wx.ALIGN_CENTER)
        self.find_sphere_image_panel.SetSizer(sizer)
        self.find_sphere_image_panel.Fit()

    def tracking_panel_init(self):
        """
        Initializes the tracking notebook page
        """
        self.tracking_panel = xrc.XRCCTRL(self.notebook, Tracking_PANEL)
        self.head_rate_plot_panel = xrc.XRCCTRL(self.tracking_panel, Head_Rate_Plot_PANEL)
        self.forw_rate_plot_panel = xrc.XRCCTRL(self.tracking_panel, Forw_Rate_Plot_PANEL)
        self.side_rate_plot_panel = xrc.XRCCTRL(self.tracking_panel, Side_Rate_Plot_PANEL)

        # Setup plot canvases
        self.head_rate_sizer = wx.BoxSizer(wx.VERTICAL)
        self.forw_rate_sizer = wx.BoxSizer(wx.VERTICAL)
        self.side_rate_sizer = wx.BoxSizer(wx.VERTICAL)

        self.head_rate_canvas = wx.lib.plot.PlotCanvas(self.head_rate_plot_panel)
        self.forw_rate_canvas = wx.lib.plot.PlotCanvas(self.forw_rate_plot_panel)
        self.side_rate_canvas = wx.lib.plot.PlotCanvas(self.side_rate_plot_panel)

        self.head_rate_sizer.Add(self.head_rate_canvas, 1, wx.EXPAND)
        self.forw_rate_sizer.Add(self.forw_rate_canvas, 1, wx.EXPAND)
        self.side_rate_sizer.Add(self.side_rate_canvas, 1, wx.EXPAND)

        self.head_rate_plot_panel.SetSizer(self.head_rate_sizer)
        self.forw_rate_plot_panel.SetSizer(self.forw_rate_sizer)
        self.side_rate_plot_panel.SetSizer(self.side_rate_sizer)

        # Setup Timer
        ID_Timer = wx.NewId()
        self.tracking_plot_timer = wx.Timer(self.wx_parent, ID_Timer)


        # Setup controls for tracking panel
        self.tracking_enable_box = xrc.XRCCTRL(self.tracking_panel,Tracking_Enable_CHECKBOX)

        # Setup events
        wx.EVT_CHECKBOX(self.tracking_enable_box, xrc.XRCID(Tracking_Enable_CHECKBOX),
                      self.on_tracking_enable)
        wx.EVT_TIMER(self.wx_parent, ID_Timer, self.on_tracking_plot_timer)

        # Set Defaults for tracking panel
        self.tracking_enable = TRACKING_DEFAULTS['tracking_enable']
        self.tracking_enable_box.SetValue(self.tracking_enable)
        self.tracking_plot_interval = TRACKING_DEFAULTS['tracking_plot_poll_int']
        self.tracking_plot_timer.Start(self.tracking_plot_interval)


    def closed_loop_panel_init(self):
        """
        Initializes the closed loop notebook page
        """
        self.closed_loop_panel = xrc.XRCCTRL(self.notebook,Closed_Loop_PANEL)

        ########################################
        #  copied from Flytrax

        send_to_ip_enabled_widget = xrc.XRCCTRL(self.frame,"SEND_TO_IP_ENABLED")
        send_to_ip_enabled_widget.Bind( wx.EVT_CHECKBOX,
                                        self.OnEnableSendToIP)
        if send_to_ip_enabled_widget.IsChecked():
            self.send_over_ip.set()
        else:
            self.send_over_ip.clear()

        ctrl = xrc.XRCCTRL(self.frame,"EDIT_UDP_RECEIVERS")
        ctrl.Bind( wx.EVT_BUTTON, self.OnEditUDPReceivers)
        self.edit_udp_receivers_dlg = RES.LoadDialog(self.frame,"UDP_RECEIVER_DIALOG")

        # UDP receiver diaglog init
        ctrl = xrc.XRCCTRL(self.edit_udp_receivers_dlg,"UDP_ADD")
        ctrl.Bind(wx.EVT_BUTTON, self.OnUDPAdd )

        ctrl = xrc.XRCCTRL(self.edit_udp_receivers_dlg,"UDP_EDIT")
        wx.EVT_BUTTON(ctrl,ctrl.GetId(),self.OnUDPEdit)

        ctrl = xrc.XRCCTRL(self.edit_udp_receivers_dlg,"UDP_REMOVE")
        wx.EVT_BUTTON(ctrl,ctrl.GetId(),self.OnUDPRemove)


    # Callbacks for optic flow page -----------------------------------------
    def on_optic_flow_enable(self, event):
        self.lock.acquire()
        widget = event.GetEventObject()
        if widget.IsChecked():
            self.optic_flow_enable = True
            self.timestamp_last = 0.0
            self.line_list = []
            self.dpix_list = []

        else:
            self.optic_flow_enable = False
        print "optic_flow_enable: ", self.optic_flow_enable
        self.lock.release()

    def on_num_row_spin_ctrl(self, event):
        self.lock.acquire()
        widget = event.GetEventObject()
        self.num_row = widget.GetValue()
        print 'num_row_spin_ctlr: ', self.num_row
        self.lock.release()

    def on_num_col_spin_ctrl(self, event):
        self.lock.acquire()
        widget = event.GetEventObject()
        self.num_col = widget.GetValue()
        print 'num_col_spin_ctlr: ', self.num_col
        self.lock.release()

    def on_window_size_spin_ctrl(self, event):
        self.lock.acquire()
        widget = event.GetEventObject()
        self.wnd = widget.GetValue()
        print 'window_size_spin_ctrl: ', self.wnd
        self.lock.release()

    def on_poll_int_text_enter(self, event):
        self.lock.acquire()
        widget = event.GetEventObject()
        value = widget.GetValue()
        try:
            self.poll_int = float(widget.GetValue())
        except:
            widget.SetValue(str(self.poll_int))
        print 'poll_int_text_enter: ', self.poll_int
        self.lock.release()

    def on_horiz_space_slider(self, event):
        self.lock.acquire()
        widget = event.GetEventObject()
        self.horiz_space = get_slider_value(widget)
        print 'horiz_space_slider: ', self.horiz_space
        self.lock.release()

    def on_horiz_position_slider(self, event):
        self.lock.acquire()
        widget = event.GetEventObject()
        self.horiz_pos = get_slider_value(widget)
        print 'horiz_position_slider: ', self.horiz_pos
        self.lock.release()

    def on_vert_space_slider(self, event):
        self.lock.acquire()
        widget = event.GetEventObject()
        self.vert_space = get_slider_value(widget)
        print 'vert_space_slider:', self.vert_space
        self.lock.release()

    def on_vert_position_slider(self, event):
        self.lock.acquire()
        widget = event.GetEventObject()
        self.vert_pos = get_slider_value(widget)
        print 'vert_position_slider:', self.vert_pos
        self.lock.release()

    # Callbacks for find sphere page ------------------------------------------
    def on_grab_image_button(self, event):
        self.lock.acquire()
        print 'grab image -- ',
        if self.lag_buf.is_ready():
            buf, time_stamp = self.lag_buf.val()
            self.plot_panel.plot_data(buf)
        else:
            print 'no image'
        self.lock.release()

    def on_delete_points_button(self, event):
        self.lock.acquire()
        print 'delete points -- '
        self.plot_panel.delete_points()
        self.lock.release()

    def on_find_sphere_button(self, event):

        print 'find_sphere'
        pts = self.plot_panel.get_points()

        if len(pts) < 3:
            dlg = wx.MessageDialog(self.plot_panel,
                                   'At least three boundary points require to locate sphere',
                                   'SphereTrax Error',
                                   wx.OK | wx.ICON_ERROR
                                   )
            dlg.ShowModal()
            dlg.Destroy()
            return

        self.lock.acquire()
        # Find positon of the sphere
        self.plot_panel.sphere_pos = find_sphere_pos(self.plot_panel.cam_cal,
                                                     self.plot_panel.radius,
                                                     self.plot_panel.z_guess,
                                                     pts)

        # Reproject sphere position
        u_list, v_list, u0, v0 = get_reproject_pts(self.plot_panel.cam_cal,
                                                   self.plot_panel.radius,
                                                   self.plot_panel.sphere_pos)

        self.plot_panel.update_reproj_points(u_list,v_list)
        self.plot_panel.update_center_points([u0],[v0])
        self.lock.release()

    # Callbacks for tracking panel page  ----------------------------------
    def on_tracking_enable(self, event):

        # Check that optic flow enabled before enabling tracking
        if not self.optic_flow_enable:
            dlg = wx.MessageDialog(self.tracking_panel,
                                   'Optic flow must be enabled before tracking',
                                   'SphereTrax Error',
                                   wx.OK | wx.ICON_ERROR
                                   )
            dlg.ShowModal()
            dlg.Destroy()
            self.lock.acquire()
            self.tracking_enable = False
            self.lock.release()
            self.tracking_enable_box.SetValue(False)
            return

        # Check that the sphere position has been found prior to enabling tracking
        if not self.plot_panel.sphere_pos:
            dlg = wx.MessageDialog(self.tracking_panel,
                                   'Sphere position is required before tracking can be enabled',
                                   'SphereTrax Error',
                                   wx.OK | wx.ICON_ERROR
                                   )
            dlg.ShowModal()
            dlg.Destroy()
            self.lock.acquire()
            self.tracking_enable = False
            self.lock.release()
            self.tracking_enable_box.SetValue(False)
            return

        # Check that there are at least three optic flow points prior to enabing tracking
        self.lock.acquire()
        num_pts = self.num_col*self.num_row
        self.lock.release()
        if num_pts < 3:
            dlg = wx.MessageDialog(self.tracking_panel,
                                   'At least 3 optic flow points required to enable tracking',
                                   'SphereTrax Error',
                                   wx.OK | wx.ICON_ERROR
                                   )
            dlg.ShowModal()
            dlg.Destroy()
            self.lock.acquire()
            self.tracking_enable = False
            self.lock.release()
            self.tracking_enable_box.SetValue(False)
            return


        self.lock.acquire()
        widget = event.GetEventObject()
        if widget.IsChecked():
            self.tracking_enable = True
        else:
            self.tracking_enable = False
        print "tracking_enable: ", self.tracking_enable
        self.lock.release()

    # Timer callback -----------------------------------------------------
    def on_tracking_plot_timer(self, event):
        #####################################################################
        # This function is currently pretty wasteful which may limit the
        # minimum allowable interval for the timer callback interval.
        #
        # Possible solutions:
        #
        # 1.) More efficient - less with the list comprehensions
        # 2.) Or don't create new PolyLine and PlotGraphics objects in each
        #     call
        # 3.) Put less data in the tracking plot queue
        #####################################################################
        if not self.tracking_enable:
            return

        try:
            while not self.tracking_plot_queue.empty():
                data = self.tracking_plot_queue.get(False)
                self.tracking_plot_data.append(data)
        except Queue.Empty:
            pass

        if not len(self.tracking_plot_data):
            # no data
            return

        # Cull tracking plot data which is too old
        t_last = self.tracking_plot_data[-1][0]
        self.tracking_plot_data = [(t,h,f,s) for t,h,f,s in self.tracking_plot_data
                                   if t_last-t <= TRACKING_DEFAULTS['tracking_plot_length']]

        # Do I need this ????
        if not self.frame.IsShown():
            return

        # Get plot defaults
        x_axis = (-TRACKING_DEFAULTS['tracking_plot_length'], 0)
        color = TRACKING_DEFAULTS['tracking_plot_line_color']
        width = TRACKING_DEFAULTS['tracking_plot_line_width']

        # Plot heading rate data
        data = [(t-t_last,h) for t,h,f,s in self.tracking_plot_data]
        line = wx.lib.plot.PolyLine(data, legend='', colour=color, width=width)
        gc = wx.lib.plot.PlotGraphics([line], 'heading rate', '', '(deg/sec)')
        self.head_rate_canvas.Draw(gc, xAxis=x_axis, yAxis =(-3,3))
        #hr = data[-1][1]

        # Plot forward velocity data
        data = [(t-t_last,f) for t,h,f,s in self.tracking_plot_data]
        line = wx.lib.plot.PolyLine(data, legend='', colour=color, width=width)
        gc = wx.lib.plot.PlotGraphics([line], 'forward velocity', '', '(mm/s)')
        self.forw_rate_canvas.Draw(gc, xAxis=x_axis, yAxis=(-10,10))
        #fr = data[-1][1]


        # Plot side velocity data
        data = [(t-t_last,s) for t,h,f,s in self.tracking_plot_data]
        line = wx.lib.plot.PolyLine(data, legend='', colour=color, width=width)
        gc = wx.lib.plot.PlotGraphics([line], 'sideways velocity', 't (sec)', '(mm/s)')
        self.side_rate_canvas.Draw(gc,xAxis=x_axis, yAxis=(-10,10))
        #sr = data[-1][1]
        #print 'hr: %1.2f, fr: %1.2f, sr: %1.2f'%(hr, fr, sr)


    #  UDP code copied from FlyTrax
    def OnEnableSendToIP(self,event):
        widget = event.GetEventObject()
        if widget.IsChecked():
            self.send_over_ip.set()
        else:
            self.send_over_ip.clear()

    def OnEditUDPReceivers(self,event):
        self.edit_udp_receivers_dlg.ShowModal()

    def remote_hosts_changed(self):
        listctrl = xrc.XRCCTRL(self.edit_udp_receivers_dlg,"UDP_RECEIVER_LIST")
        n = listctrl.GetCount()

        self.remote_host_lock.acquire()
        try:
            if n > 0:
                self.remote_host = []
                for idx in range(n):
                    self.remote_host.append( listctrl.GetClientData(idx) )
            else:
                self.remote_host = None
        finally:
            self.remote_host_lock.release()
            self.remote_host_changed.set()

        ctrl = xrc.XRCCTRL(self.frame,'SEND_TO_IP_ENABLED')
        ctrl.SetLabel('send data to %d receiver(s)'%n)

    def OnUDPAdd(self,event):
        listctrl = xrc.XRCCTRL(self.edit_udp_receivers_dlg,"UDP_RECEIVER_LIST")
        dlg = wx.TextEntryDialog(self.wx_parent,
                                 'Please add the hostname',
                                 )
        try:
            if dlg.ShowModal() == wx.ID_OK:
                hostname = dlg.GetValue()
                try:
                    ip = socket.gethostbyname(hostname)
                except socket.gaierror, x:
                    dlg2 = wx.MessageDialog(dlg,
                                            'error getting IP address: '+str(x),
                                            'FlyTrax: socket error',
                                            wx.OK | wx.ICON_ERROR)
                    dlg2.ShowModal()
                    dlg2.Destroy()
                else:
                    remote_host = (ip, 28932)
                    if hostname != '':
                        toshow = hostname
                    else:
                        toshow = str(ip)
                    idx = listctrl.Append( toshow )
                    listctrl.SetClientData(idx,remote_host)
                    self.remote_hosts_changed()
        finally:
            dlg.Destroy()

    def OnUDPEdit(self,event):
        widget = event.GetEventObject()

    def OnUDPRemove(self,event):
        listctrl = xrc.XRCCTRL(self.edit_udp_receivers_dlg,"UDP_RECEIVER_LIST")
        idx = listctrl.GetSelection()
        if idx==wx.NOT_FOUND:
            return
        remote_host = listctrl.GetClientData(idx)
        listctrl.Delete(idx)
        self.remote_hosts_changed()


    # Other frame processing, Andrew's stuff, etc -------------------------
    def get_frame(self):
        """return wxPython frame widget"""
        return self.frame

    def get_plugin_name(self):
        """
        """
        return 'SphereTrax'

    def process_frame(self,cam_id,buf,buf_offset,timestamp,framenumber):
        """
        do work on each frame
        This function gets called on every single frame capture. It is
        called within the realtime thread, NOT the wxPython
        application mainloops thread. Therefore, be extremely careful
        (use threading locks) when sharing data with the rest of the class.
        """
        # Empty pixel displacement list
        self.dpix_list = []

        # Get varibles which are shared with GUI thread
        self.lock.acquire()
        optic_flow_enable = self.optic_flow_enable
        tracking_enable = self.tracking_enable
        num_row = self.num_row
        num_col = self.num_col
        horiz_space = self.horiz_space
        horiz_pos = self.horiz_pos
        vert_space = self.vert_space
        vert_pos = self.vert_pos
        poll_int = self.poll_int
        wnd = self.wnd
        cam_cal = self.plot_panel.cam_cal
        sphere_radius = self.plot_panel.radius
        sphere_pos = self.plot_panel.sphere_pos
        self.lock.release()

        # Get image buffer information
        buf = numpy.asarray(buf)
        n,m = buf.shape

        if not self.lag_buf.is_ready() or not optic_flow_enable:
            # Buffer is not ready or optic flow is not enabled
            self.of_pix = []
            self.line_list = []
        else:
            # Get pixels for optic flow computation
            self.of_pix = get_optic_flow_pix(num_row,num_col,horiz_space,horiz_pos,
                                             vert_space,vert_pos,(0,m),(0,n),wnd)
            t_start = time.time()
            if timestamp - self.timestamp_last >= poll_int:

                # Perform optic flow computation for each pixel in pixel list
                self.line_list = []
                for pix in self.of_pix:
                    im_curr = buf
                    im_prev, timestamp_prev = self.lag_buf.val()
                    dt = timestamp - timestamp_prev
                    dpix = get_optic_flow(im_curr,im_prev,pix,wnd,dt)
                    self.dpix_list.append(dpix)
                    scal = 0.25
                    self.line_list.append([pix[0],pix[1],pix[0]-scal*dpix[0],pix[1]-scal*dpix[1]])
                # Perform Tracking if enabled

                if tracking_enable:
                    #####################################################################
                    # Will want to check that the appropriate conditions for tracking
                    # or disable all possible changes to these conditions when tracking
                    # is enabled. These include: optic flow enable, sphere position, and
                    # number of optic flow computation points.
                    #####################################################################

                    # Compute angular velocity

                    omega = get_ang_vel(self.of_pix, self.dpix_list, cam_cal, sphere_radius,
                                        sphere_pos)


                    # Apply Kalman filter
                    state_filt, error = self.kalman.step(omega,isinitial=self.kalman_isinitial)
                    self.kalman_isinitial = 0
                    omega_filt = state_filt[0:3]


                    # Compute heading rate, forward and side velocities
                    #head_rate, forw_rate, side_rate = get_hfs_rates(omega,SPHERE_ORIENTATION,
                    #                                                sphere_radius)
                    head_rate, forw_rate, side_rate = get_hfs_rates(omega_filt,SPHERE_ORIENTATION,
                                                                    sphere_radius)

                    # TESTING ##################################################
                    if self.tracking_plot_cntr == 1:
                        plot_data = (timestamp, head_rate, forw_rate, side_rate)
                        self.tracking_plot_queue.put(plot_data)
                        self.tracking_plot_cntr = 0
                    else:
                        self.tracking_plot_cntr+=1

                    #  copied from flytrax
                    # find any new IP addresses to send data to
                    if self.remote_host_changed.isSet():
                        self.remote_host_lock.acquire()
                        self.runthread_remote_host = self.remote_host
                        self.remote_host_lock.release()
                        self.remote_host_changed.clear()

                    # send data over UDP
                    if self.send_over_ip.isSet() and self.runthread_remote_host is not None:
                        databuf = struct.pack('cLdfff', 'a', framenumber, timestamp,
                                              head_rate, forw_rate, side_rate)
                        for remote_host in self.runthread_remote_host:
                            self.sockobj.sendto(databuf, remote_host)


                #Record timestamp of last optic flow calculation
                self.timestamp_last = timestamp

        draw_points = self.of_pix
        draw_linesegs = self.line_list
        self.lag_buf.add((numpy.array(buf,copy=True), timestamp))
        return draw_points, draw_linesegs

    def set_view_flip_LR( self, val ):
        pass

    def set_view_rotate_180( self, val):
        pass

    def quit(self):
        pass

    def camera_starting_notification(self,cam_id,
                                     pixel_format=None,
                                     max_width=None,
                                     max_height=None):
        pass


# -----------------------------------------------------------
class PlotPanel(wx.Panel):

    def __init__(self, parent,statbar=None):
        wx.Panel.__init__(self, parent, -1)

        # Create figure and canvas
        self.fig = Figure(figsize=(0.1,0.1))
        self.canvas = FigureCanvasWxAgg(self, -1, self.fig)

        # Allow resizing
        sizer = wx.BoxSizer(wx.VERTICAL)
        sizer.Add(self.canvas, 1, wx.GROW|wx.TOP|wx.LEFT)
        self.SetSizer(sizer)
        self.Fit()

        # Setup axes
        self.axes = self.fig.add_subplot(111)
        self.axes.set_visible(False)
        self.im = None

        # Connect button press event
        self.canvas.mpl_connect('button_press_event',self.on_button_press)

        # Setup lines
        self.input_line = self.axes.plot([],[],'.b')[0]
        self.reproj_line = self.axes.plot([],[],'.r')[0]
        self.center_line = self.axes.plot([],[],'.g')[0]

        # Sphere position
        self.sphere_pos = None
        self.extent = (0,0,0,0)

        # Load camera calibration and sphere data
        cam_cal_file = pkg_resources.resource_filename(__name__,
                                                       'data/camera_cal.txt')
        self.cam_cal = load_cam_cal(cam_cal_file)

        sphere_data_file = pkg_resources.resource_filename(__name__,
                                                           'data/sphere_defaults.txt')
        self.radius, self.z_guess = load_sphere_data(sphere_data_file)

    def plot_data(self, buf):
        print 'plotting buffer'
        self.axes.cla()
        n,m = buf.shape

        plot_buf = buf
        extent = (0,m,n,0)

        self.extent = extent
        self.axes.set_visible(True)
        self.axes.imshow(plot_buf,
                         cmap=matplotlib.cm.pink,
                         origin='upper',
                         extent=extent)

        self.axes.add_line(self.input_line)
        self.axes.add_line(self.reproj_line)
        self.axes.add_line(self.center_line)
        self.canvas.draw()

    def on_button_press(self,event):
        print 'pressed', event.xdata, event.ydata
        xdata = list(self.input_line.get_xdata())
        ydata = list(self.input_line.get_ydata())
        xdata.append(event.xdata)
        ydata.append(event.ydata)

        ##############################################
        #print self.extent, event.xdata, event.ydata
        ##############################################

        self.input_line.set_data(xdata,ydata)
        self.canvas.draw()


    def delete_points(self):
        self.input_line.set_data([],[])
        self.reproj_line.set_data([],[])
        self.center_line.set_data([],[])
        self.sphere_pos = None
        self.canvas.draw()

    def get_points(self):
        xdata = self.input_line.get_xdata()
        ydata = self.input_line.get_ydata()
        return zip(xdata,ydata)

    def update_reproj_points(self,xdata,ydata):
        self.reproj_line.set_data(xdata,ydata)
        self.canvas.draw()

    def update_center_points(self,xdata,ydata):
        self.center_line.set_data(xdata,ydata)
        self.canvas.draw()


# Utility functions -----------------------------------------
def load_cam_cal(calib_file):
    """
    Loads the camera calibration
    """
    fid = open(calib_file)
    line = fid.readline()
    fid.close()
    cal = [float(x) for x in line.split()]
    return tuple (cal)

def load_sphere_data(filename):
    fid = open(filename)
    for line in fid.readlines():
        line = line.split()
        # Should really add some error checing here
        if line[0] == 'radius':
            radius = float(line[1])
        if line[0] == 'z_guess':
            z_guess = float(line[1])
    fid.close()
    return radius, z_guess

def find_sphere_pos(cal, radius, z_guess, pts ):
    """
    Computes the position of the center of the sphere
    """
    # Convert points to world coords
    f0,f1,c0,c1 = cal
    pts_world = [ ((p[0]-c0)/f0, (p[1]-c1)/f1) for p in pts]

    # x,y position of initial guess
    x0 = 0.0, 0.0, z_guess
    x = fmin(find_sphere_cost,x0,(pts_world,radius))
    return tuple(x)

def find_sphere_cost(x,pts,radius):
    """
    Optimization cost function for find sphere position
    """
    val = 0.0
    for p in pts:
        A = p[0]**2 + p[1]**2 + 1
        B = -2.0*(p[0]*x[0] + p[1]*x[1] + x[2])
        C = x[0]**2 + x[1]**2 + x[2]**2 - radius**2
        val += (B**2 - 4.0*A*C)**2
    return val


def get_reproject_pts(cal, radius, pos, n=100):
    """
    Using camera calibration and extimated sphere position get
    points reprojected from the surface of the sphere and the sphere's
    center.
    """
    f0,f1,c0,c1 = cal
    # Get point from sphere's surfacE
    ang_list = numpy.linspace(0.0,2.0*numpy.pi,n)
    u_list, v_list = [], []
    for a in ang_list:
        for b in ang_list:
            x = pos[0] + radius*numpy.cos(a)*numpy.cos(b)
            y = pos[1] + radius*numpy.sin(a)*numpy.cos(b)
            z = pos[2] + radius*numpy.sin(b)
            u = f0*x/z + c0
            v = f1*y/z + c1
            u_list.append(u)
            v_list.append(v)
    # Get center of sphere
    u0 = f0*pos[0]/pos[2] + c0
    v0 = f1*pos[1]/pos[2] + c1
    return u_list, v_list, u0, v0


def get_slider_value(widget):
    """
    Return value of slide as float between 0 and 1
    """
    slider_max = float(widget.GetMax())
    slider_min = float(widget.GetMin())
    slider_val = float(widget.GetValue())
    value =  (slider_val - slider_min)/(slider_max - slider_min)
    return value

def set_slider_value(widget,value):
    """
    Set Slider value using a float between 0 and 1
    """
    slider_min = widget.GetMin()
    slider_max = widget.GetMax()
    slider_range = slider_max - slider_min
    widget.SetValue(int(slider_min + value*slider_range))


def get_optic_flow_pix(num_row, num_col, horiz_space, horiz_pos, vert_space,
                       vert_pos, x_limits, y_limits, wnd):
    """
    Get list of pixel corrdinates for the opticflow computation
    """

    # Determine allowable region accounting for optix flow window
    x_min = x_limits[0]+wnd+3
    x_max = x_limits[1]-wnd-3
    y_min = y_limits[0]+wnd+3
    y_max = y_limits[1]-wnd-3
    x_range = x_max-x_min
    y_range = y_max-y_min

    # Compute optic flow pixels
    of_pix = []
    for x_ind in range(0,num_col):
        for y_ind in range(0,num_row):

            # Compute x pixel location
            if num_col==1:
                x_pix = x_min*(1.0-horiz_pos) + x_max*horiz_pos
            else:
                x_pix = horiz_space*x_range*x_ind/float(num_col-1) + x_min
                x_pix = x_pix + x_range*(1.0-horiz_space)*horiz_pos
            x_pix = x_limits[1]-x_pix

            # Compute y pixel location
            if num_row==1:
                y_pix = y_min*(1.0-vert_pos) + y_max*vert_pos
            else:
                y_pix = vert_space*y_range*y_ind/float(num_row-1) + y_min
                y_pix = y_pix + y_range*(1.0-vert_space)*vert_pos
            y_pix = y_limits[1]-y_pix

            of_pix.append((x_pix, y_pix))

    return of_pix


class LagList:

    def __init__(self, len):
        self.len = len
        self.list = []

    def add(self, x):
        self.list.insert(0,x)
        if len(self.list) > self.len:
            self.list = self.list[:-1]

    def val(self):
        return self.list[-1]

    def is_ready(self):
        if len(self.list) == self.len:
            return True
        else:
            return False
