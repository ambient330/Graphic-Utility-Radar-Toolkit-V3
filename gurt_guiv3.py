#!/usr/bin/env python3
"""
GURT GUI — Graphic Utility Radar Toolkit (Graphical Edition)
@multidpppler  v4.0

Controls:
  Left/Right arrows  — previous/next file in folder
  Up/Down arrows     — higher/lower tilt (sweep)
  Click + drag       — rubber-band zoom into selection
  Right-click panel  — open Parameter & Colors editor for that panel
  Escape             — reset zoom to full range
"""

import os
import sys
import argparse
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from datetime import datetime

import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.patches import Rectangle
from scipy.interpolate import interp1d

try:
    import pyart
    PYART_AVAILABLE = True
except ImportError:
    PYART_AVAILABLE = False
    print("WARNING: pyart not found. Install with: pip install arm-pyart")

def _make_cmap(name, hex_colors):
    rgb = [mcolors.hex2color(h) for h in hex_colors]
    return mcolors.LinearSegmentedColormap.from_list(name, rgb[::-1])

ZDR_CMAP = _make_cmap("gurt_zdr", [
    '#922E97','#B26CB6','#C998CB','#E2C7E3','#FEF9FB','#F8BEDB','#EF77B2',
    '#CF3B58','#B00000','#C80603','#DE1A0B','#EE8836','#FEF861','#5ADE64',
    '#3FE2CF','#2474B4','#0B0D9C','#D1D1D9','#7F6CA2','#453B58','#292335','#060507'])
CC_CMAP = _make_cmap("gurt_cc", [
    '#8B1E4D','#E41000','#FC7F00','#FFB600','#FFFB00','#BCE906','#87D70B',
    '#61ED6E','#719CD2','#5151E8','#2929D1','#0A0ABD','#0C0CAC','#0D0D9C',
    '#0F0F8C','#1C1C9E','#2D2D84','#404068','#454561','#4F4F4F'])
KDP_CMAP = _make_cmap("gurt_kdp", [
    '#C361F9','#6F329A','#160234','#624264','#B18596','#FAC4C5','#FF7B00',
    '#FFBC00','#FEFF00','#84DA1A','#16BA31','#3ADB94','#60FEF6','#74C7D1',
    '#8987A2','#9B507A','#EA77B8','#CE5B93','#B03D6A','#921F42','#75021B',
    '#62000E','#4B0101','#4B2828','#4B4A4A','#5F5F5F','#757575'])
SW_CMAP = _make_cmap("gurt_sw", [
    '#02A0C8','#2CA7C6','#53AEC5','#78B4C3','#9FBBC1','#C1C1C1','#DCDCDC',
    '#E6E6E6','#F2F2F2','#FFFD01','#FDC60F','#FDB313','#FC991A','#F7742D',
    '#EF6341','#E54F5B','#DE406D','#B73192','#7D26BD','#31148A','#1A0855'])
VEL_CMAP = _make_cmap("gurt_vel", [
    '#F2465B','#E0365B','#CF2646','#B71630','#9F0100','#A05060','#7C544C',
    '#885840','#966440','#A5703E','#B37B36','#C1872E','#D09326','#DE9F1E',
    '#ECAB16','#EFB70E','#EFC306','#EFCF00','#ECE4B0','#D0E4D0','#BFDCBF',
    '#A1D0A1','#85C485','#67B867','#4AAD4A','#2DA12D','#0F950F','#007D00',
    '#017100','#30855F','#48969A','#718FFE','#6A60FE','#5A24E5','#4A0EC3',
    '#3A0EAB','#2A0E94','#673A8F','#7805A3'])
REF_CMAP = _make_cmap("gurt_ref", [
    '#F2465B','#E0365B','#CF2646','#B71630','#9F0100','#A05060','#7C544C',
    '#885840','#966440','#A5703E','#B37B36','#C1872E','#D09326','#DE9F1E',
    '#ECAB16','#EFB70E','#EFC306','#EFCF00','#ECE4B0','#D0E4D0','#BFDCBF',
    '#A1D0A1','#85C485','#67B867','#4AAD4A','#2DA12D','#0F950F','#007D00',
    '#017100','#30855F','#48969A','#718FFE','#6A60FE','#5A24E5','#4A0EC3',
    '#3A0EAB','#2A0E94','#673A8F','#7805A3'])

GURT_CMAPS = {
    "gurt_ref":  REF_CMAP,
    "gurt_vel":  VEL_CMAP,
    "gurt_zdr":  ZDR_CMAP,
    "gurt_kdp":  KDP_CMAP,
    "gurt_sw":   SW_CMAP,
    "gurt_cc":   CC_CMAP,
}

MPL_CMAPS = [
    "viridis","plasma","inferno","magma","cividis",
    "RdBu_r","RdYlBu_r","Spectral_r","coolwarm","bwr",
    "jet","rainbow","turbo","nipy_spectral",
    "Greys","hot","bone","copper","pink",
    "PuOr","PRGn","BrBG","PiYG",
    "YlOrRd","YlOrBr","OrRd","Reds","Blues","Greens","Purples",
]

PYART_CMAPS = []
if PYART_AVAILABLE:
    try:
        for _mod in (getattr(pyart.graph, 'cm', None),
                     getattr(pyart.graph, 'cm_tables', None)):
            if _mod is None:
                continue
            for _name in dir(_mod):
                if not _name.startswith('_') and f"pyart_{_name}" not in PYART_CMAPS:
                    PYART_CMAPS.append(f"pyart_{_name}")
    except Exception:
        pass

ALL_CMAP_NAMES = list(GURT_CMAPS.keys()) + MPL_CMAPS + PYART_CMAPS

def resolve_cmap(name_or_obj):
    if not isinstance(name_or_obj, str):
        return name_or_obj
    if name_or_obj in GURT_CMAPS:
        return GURT_CMAPS[name_or_obj]
    if name_or_obj.startswith("pyart_") and PYART_AVAILABLE:
        bare = name_or_obj[6:]
        for mod in (getattr(pyart.graph, 'cm', None),
                    getattr(pyart.graph, 'cm_tables', None)):
            if mod is None:
                continue
            obj = getattr(mod, bare, None)
            if obj is not None:
                return obj
    try:
        return plt.get_cmap(name_or_obj)
    except Exception:
        pass
    if PYART_AVAILABLE:
        for mod in (getattr(pyart.graph, 'cm', None),
                    getattr(pyart.graph, 'cm_tables', None)):
            if mod is None:
                continue
            obj = getattr(mod, 'Carbone42', None)
            if obj is not None:
                return obj
    return REF_CMAP

FIELD_DEFAULTS = {
    "DBZH":    ("Z_h (dBZ)",          -20,  70,  "pyart_Carbone42"),
    "DBZH1":   ("Z_h (dBZ)",          -28,  35,  "pyart_Carbone42"),
    "DBZ":     ("Z_h (dBZ)",          -20,  70,  "pyart_Carbone42"),
    "reflectivity": ("Z_h (dBZ)",     -20,  70,  "pyart_Carbone42"),
    "DBMHC":   ("Z_h (dBZ)",          -20,  70,  "pyart_Carbone42"),
    "VEL":     ("Vr (m/s)",           -30,  30,  "gurt_vel"),
    "velocity":("Vr (m/s)",           -30,  30,  "gurt_vel"),
    "VU":      ("Vr (m/s)",           -30,  30,  "gurt_vel"),
    "VELD":    ("Vr dealias (m/s)",   -60,  60,  "gurt_vel"),
    "ZDR":     ("Z_dr (dB)",           -2,   8,  "gurt_zdr"),
    "RHOHV":   ("rhoHV",               0,   1,  "gurt_cc"),
    "KDP":     ("K_dp (deg/km)",       -2,  12,  "gurt_kdp"),
    "DP":      ("K_dp (deg/km)",       -2,  12,  "gurt_kdp"),
    "PHIDP":   ("phi_dp (deg)",         0, 360,  "gurt_sw"),
    "PHI":     ("phi_dp (deg)",         0, 360,  "gurt_sw"),
    "differential_phase": ("phi_dp (deg)", 0, 360, "gurt_sw"),
    "width":   ("sigma (m/s)",          0,  15,  "gurt_sw"),
    "spectrum_width": ("sigma (m/s)",   0,  15,  "gurt_sw"),
    "SW":      ("sigma (m/s)",          0,  15,  "gurt_sw"),
    "SNR":     ("SNR (dB)",           -10,  50,  "pyart_Carbone42"),
}

FILE_EXTENSIONS = ['.nc', '.h5', '.buf', '.raw', '.gz', '_V06',
                   '.msg31', '.0', '.ar2v', '.RAW', '.HDF5', '.hdf5']

def calculate_kdp(radar, force=False):
    if 'DP' in radar.fields and not force:
        return radar
    PHIDP_NAMES = ['PHIDP', 'PHI', 'differential_phase', 'PH']
    REFL_NAMES  = ['DBZH', 'DBZH1', 'reflectivity', 'DBZ', 'DBMHC']
    phi_name  = next((n for n in PHIDP_NAMES if n in radar.fields), None)
    if phi_name is None:
        return radar
    refl_name = next((n for n in REFL_NAMES  if n in radar.fields), None)
    refl_mask = (radar.fields[refl_name]['data'] < 20) if refl_name else \
                np.zeros_like(radar.fields[phi_name]['data'].data, dtype=bool)
    phidp_raw = radar.fields[phi_name]['data'].copy()
    r  = radar.range['data']
    dr = (r[1] - r[0]) / 1000.0
    if hasattr(phidp_raw, 'mask'):
        mask  = phidp_raw.mask.copy()
        phidp = phidp_raw.filled(np.nan)
    else:
        fill  = radar.fields[phi_name].get('_FillValue', -9999.0)
        mask  = np.isclose(phidp_raw, fill) | np.isnan(phidp_raw)
        phidp = np.array(phidp_raw, dtype=float)
    combined = mask | refl_mask
    for i in range(phidp.shape[0]):
        vi = ~np.isnan(phidp[i]) & ~combined[i]
        if vi.sum() > 0:
            phidp[i, vi] = np.unwrap(np.deg2rad(phidp[i, vi])) * 180 / np.pi
    kdp = np.zeros_like(phidp)
    win = 9; kern = np.ones(win)/win; half = 4
    for i in range(phidp.shape[0]):
        v = ~np.isnan(phidp[i]) & ~combined[i]
        if v.sum() > win:
            p = np.pad(phidp[i, v], (win//2, win//2), mode='edge')
            phidp[i, v] = np.convolve(p, kern, mode='valid')
            kr = np.zeros_like(phidp[i])
            for j in range(half, len(phidp[i]) - half):
                if v[j-half] and v[j+half]:
                    kr[j] = (phidp[i,j+half] - phidp[i,j-half]) / (2*2*half*dr)
            vk = ~np.isnan(kr) & (kr != 0)
            if vk.sum() > win:
                pk = np.pad(kr[vk], (win//2, win//2), mode='edge')
                kr[vk] = np.convolve(pk, kern, mode='valid')
            kdp[i] = np.clip(kr, -2.0, 12.0)
    radar.add_field('DP', {
        'data': np.ma.masked_array(kdp, mask=combined),
        'units': 'degrees/km', 'long_name': 'Specific differential phase',
        'standard_name': 'KDP', '_FillValue': -9999.0,
    }, replace_existing=True)
    return radar

def dealias_velocity(radar, force=False):
    VEL_N  = ['VEL','velocity','VU','VC','V1','VE','VEL_F','VF']
    REFL_N = ['reflectivity','REF','DZ','DBZHC_F','DBMHC','SNR','SN','DBZH','DBZH1','DBZ']
    if 'VELD' in radar.fields and not force:
        return radar
    vn = next((n for n in VEL_N if n in radar.fields), None)
    if vn is None:
        return radar
    try:
        gf = pyart.filters.GateFilter(radar)
        rn = next((n for n in REFL_N if n in radar.fields), None)
        if rn:
            gf.exclude_below(rn, -56)
        nv = radar.instrument_parameters.get('nyquist_velocity', None)
        if nv is None:
            nyq = float(np.nanmax(np.abs(radar.fields[vn]['data'])))
        else:
            d = nv['data']
            nyq = float(d[0] if isinstance(d, np.ndarray) else d)
        dv = pyart.correct.dealias_region_based(
            radar, vel_field=vn, nyquist_vel=nyq,
            gatefilter=gf, centered=True,
            skip_between_rays=0, skip_along_ray=0)
        radar.add_field('VELD', dv)
    except Exception as e:
        print(f"Dealias error: {e}")
    return radar

class ParamColorDialog(tk.Toplevel):
    BG        = "#191970"
    FG        = "#000000"
    ENTRY_BG  = "#ffffff"
    LABEL_W   = 18

    def __init__(self, parent, slot, field_info, available_fields, on_apply):
        super().__init__(parent)
        fname, label, vmin, vmax, cmap_name = field_info
        self.slot     = slot
        self.on_apply = on_apply
        self.avail    = available_fields
        self.transient(parent)
        self.grab_set()
        self.resizable(False, False)
        self.title(f"Frame {slot+1} Parameter and Colors Widget")
        self.configure(bg=self.BG)

        top = tk.Frame(self, bg=self.BG, pady=4, padx=6)
        top.pack(fill="x")
        tk.Label(top, text="17 Colors", bg=self.BG, relief="groove",
                 padx=8).pack(side="left", padx=4)
        tk.Button(top, text="Replot", command=self._replot,
                  relief="raised", bg=self.BG, padx=12).pack(side="left", padx=4)
        tk.Button(top, text="  OK  ", command=self._ok,
                  relief="raised", bg=self.BG, padx=12).pack(side="left", padx=4)
        tk.Button(top, text="Cancel", command=self.destroy,
                  relief="raised", bg=self.BG, padx=12).pack(side="left", padx=4)

        body = tk.Frame(self, bg=self.BG)
        body.pack(fill="both", expand=True, padx=6, pady=4)

        left = tk.Frame(body, bg=self.BG)
        left.pack(side="left", fill="y", padx=(0, 10))
        tk.Label(left, text="Parameters", bg=self.BG, font=("TkDefaultFont",9,"bold")).pack()
        tk.Label(left, text="(Double-click to select)", bg=self.BG,
                 font=("TkDefaultFont",7), fg="#555").pack()
        self.listbox = tk.Listbox(left, width=14, height=18,
                                  bg="#ffffff", selectbackground="#000080",
                                  selectforeground="#ffffff", relief="sunken")
        sb = tk.Scrollbar(left, orient="vertical", command=self.listbox.yview)
        self.listbox.config(yscrollcommand=sb.set)
        self.listbox.pack(side="left", fill="y")
        sb.pack(side="left", fill="y")
        for f in available_fields:
            self.listbox.insert(tk.END, f)
        if fname in available_fields:
            idx = available_fields.index(fname)
            self.listbox.selection_set(idx)
            self.listbox.see(idx)
        self.listbox.bind("<Double-1>", self._list_select)

        right = tk.Frame(body, bg=self.BG)
        right.pack(side="left", fill="both", expand=True)

        self._fname_var = tk.StringVar(value=fname)
        self._vmin_var  = tk.StringVar(value=str(vmin))
        self._vmax_var  = tk.StringVar(value=str(vmax))
        ctr = (vmin + vmax) / 2
        inc = (vmax - vmin) / 10
        self._ctr_var   = tk.StringVar(value=f"{ctr:.3f}")
        self._inc_var   = tk.StringVar(value=f"{inc:.3f}")
        self._label_var = tk.StringVar(value=label)

        if not isinstance(cmap_name, str):
            cmap_name = getattr(cmap_name, 'name', 'gurt_ref')
        self._cmap_var  = tk.StringVar(value=cmap_name)

        self._bg_var    = tk.StringVar(value="midnightblue")
        self._missing_var = tk.StringVar(value="darkslateblue")
        self._exceed_var  = tk.StringVar(value="gray70")
        self._annot_var   = tk.StringVar(value="gray90")
        self._emph_var    = tk.StringVar(value="hotpink")
        self._emin_var    = tk.StringVar(value="0.000")
        self._emax_var    = tk.StringVar(value="0.000")

        rows = [
            ("Parameter Name",    self._fname_var,   False),
            ("Min",               self._vmin_var,    False),
            ("Max",               self._vmax_var,    False),
            ("Center",            self._ctr_var,     False),
            ("Increment",         self._inc_var,     False),
            ("Color Palette",     self._cmap_var,    True),
            ("Label",             self._label_var,   False),
            (None, None, None),
            ("Background Color",  self._bg_var,      False),
            ("Missing Data Color",self._missing_var, False),
            ("Exceeded Color",    self._exceed_var,  False),
            ("Annotation Color",  self._annot_var,   False),
            ("Emphasis Color",    self._emph_var,    False),
            ("Emphasis Min",      self._emin_var,    False),
            ("Emphasis Max",      self._emax_var,    False),
        ]

        for i, row in enumerate(rows):
            if row[0] is None:
                ttk.Separator(right, orient="horizontal").grid(
                    row=i, column=0, columnspan=2, sticky="ew", pady=3)
                continue
            lbl, var, is_combo = row
            tk.Label(right, text=lbl, bg=self.BG, anchor="e",
                     width=self.LABEL_W, font=("TkDefaultFont",9,"bold")
                     ).grid(row=i, column=0, sticky="e", padx=4, pady=2)
            if is_combo:
                cb = ttk.Combobox(right, textvariable=var,
                                  values=ALL_CMAP_NAMES, width=24,
                                  state="normal")
                cb.grid(row=i, column=1, sticky="w", padx=4, pady=2)
                cb.bind("<<ComboboxSelected>>", self._preview_cmap)
            else:
                tk.Entry(right, textvariable=var, width=26,
                         bg=self.ENTRY_BG, relief="sunken"
                         ).grid(row=i, column=1, sticky="w", padx=4, pady=2)

        tk.Label(right, text="Colormap Preview:", bg=self.BG,
                 font=("TkDefaultFont",8)).grid(
            row=len(rows), column=0, columnspan=2, sticky="w", padx=4, pady=(6,0))
        self._cmap_canvas = tk.Canvas(right, height=20, width=300,
                                      bg="#000", relief="sunken")
        self._cmap_canvas.grid(row=len(rows)+1, column=0, columnspan=2,
                               sticky="ew", padx=4, pady=2)
        self._draw_cmap_preview(cmap_name)
        self._cmap_var.trace_add("write", lambda *_: self._draw_cmap_preview(
            self._cmap_var.get()))

    def _draw_cmap_preview(self, name):
        c = self._cmap_canvas
        w = c.winfo_reqwidth() or 300
        c.delete("all")
        try:
            cm = resolve_cmap(name)
            for i in range(w):
                r,g,b,_ = cm(i/w)
                color = f"#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}"
                c.create_line(i, 0, i, 20, fill=color)
        except Exception:
            pass

    def _preview_cmap(self, event=None):
        self._draw_cmap_preview(self._cmap_var.get())

    def _list_select(self, event=None):
        sel = self.listbox.curselection()
        if not sel:
            return
        fname = self.avail[sel[0]]
        self._fname_var.set(fname)
        if fname in FIELD_DEFAULTS:
            lbl, mn, mx, cm = FIELD_DEFAULTS[fname]
            self._vmin_var.set(str(mn))
            self._vmax_var.set(str(mx))
            self._label_var.set(lbl)
            cname = cm if isinstance(cm, str) else getattr(cm, 'name', 'gurt_ref')
            self._cmap_var.set(cname)
            self._ctr_var.set(f"{(mn+mx)/2:.3f}")
            self._inc_var.set(f"{(mx-mn)/10:.3f}")

    def _collect(self):
        fname  = self._fname_var.get().strip()
        label  = self._label_var.get().strip() or fname
        cname  = self._cmap_var.get().strip()
        try:   vmin = float(self._vmin_var.get())
        except: vmin = -50
        try:   vmax = float(self._vmax_var.get())
        except: vmax  = 50
        return fname, label, vmin, vmax, cname

    def _replot(self):
        self.on_apply(self.slot, *self._collect())

    def _ok(self):
        self.on_apply(self.slot, *self._collect())
        self.destroy()

_CB_HEIGHT = 0.055
_CB_BOTTOM = 0.01

class GURTApp(tk.Tk):
    def __init__(self, folder=None, file=None):
        super().__init__()
        self.title("GURT — Graphic Utility Radar Toolkit")
        self.configure(bg="#1a1a2e")
        self.geometry("1280x820")
        self.minsize(800, 600)

        self.radar            = None
        self.file_list        = []
        self.file_index       = 0
        self.sweep_index      = tk.IntVar(value=0)
        self.n_panels         = tk.IntVar(value=1)
        self.panel_fields     = []
        self.available_fields = []
        self.panel_limits     = []

        self.show_rings    = tk.BooleanVar(value=True)
        self.show_azimuths = tk.BooleanVar(value=True)
        self.show_ticks    = tk.BooleanVar(value=True)
        self.ring_interval  = tk.DoubleVar(value=50.0)
        self.az_interval    = tk.DoubleVar(value=30.0)
        self.tick_interval  = tk.DoubleVar(value=50.0)
        self.max_range      = tk.DoubleVar(value=150.0)
        self.do_dealias    = tk.BooleanVar(value=True)
        self.do_kdp        = tk.BooleanVar(value=True)

        self._drag_start    = None
        self._drag_rect     = None
        self._drag_ax       = None
        self._right_click_panel = None
        self._single_file_mode  = False
        self._load_after_id     = None
        self._refresh_after_id  = None
        self._resize_after_id   = None

        self._build_menu()
        self._build_layout()
        self._bind_keys()

        if folder:
            self._single_file_mode = False
            self.load_folder(folder)
        elif file:
            self._single_file_mode = True
            self.file_list  = [os.path.abspath(file)]
            self.file_index = 0
            self.load_current_file()

    def _mk_menu(self, parent):
        return tk.Menu(parent, tearoff=0, bg="#d4d0c8", fg="#000",
                       activebackground="#000080", activeforeground="#fff")

    def _build_menu(self):
        mb = tk.Menu(self, tearoff=0, bg="#d4d0c8", fg="#000",
                     activebackground="#000080", activeforeground="#fff",
                     relief="flat")
        self.config(menu=mb)

        fm = self._mk_menu(mb)
        fm.add_command(label="Open File…",       command=self.open_file)
        fm.add_command(label="Open Folder…",     command=self.open_folder)
        fm.add_separator()
        fm.add_command(label="Save Image…",      command=self.save_image)
        fm.add_command(label="Save All Images…", command=self.save_all_images)
        fm.add_separator()
        fm.add_command(label="Exit",             command=self.quit)
        mb.add_cascade(label="File", menu=fm)

        zm = self._mk_menu(mb)
        zm.add_command(label="Reset Zoom (all panels)", command=self._reset_all_zoom)
        zm.add_separator()
        for lbl, factor in [("Data Extent", "data"),("Default 150 km", 150),
                             ("+50%",1.5),("+25%",1.25),("+10%",1.1),
                             ("-10%",0.9),("-25%",0.75),("-50%",0.5)]:
            zm.add_command(label=lbl, command=lambda f=factor: self._zoom_all(f))
        zm.add_separator()
        zm.add_command(label="Set Max Range…", command=self._set_max_range)
        mb.add_cascade(label="Zoom", menu=zm)

        cm = self._mk_menu(mb)
        cm.add_command(label="Radar Origin (reset pan)", command=self._reset_all_zoom)
        cm.add_command(label="Center on Last Click",     command=self._center_on_click)
        mb.add_cascade(label="Center", menu=cm)

        pm = self._mk_menu(mb)
        for n in [1,2,3,4,5]:
            pm.add_radiobutton(label=f"{n} Panel{'s' if n>1 else ''}",
                               variable=self.n_panels, value=n,
                               command=self._on_panels_changed)
        mb.add_cascade(label="Config", menu=pm)

        om = self._mk_menu(mb)
        om.add_checkbutton(label="Range Rings",   variable=self.show_rings,
                           command=self.refresh_plot)
        om.add_checkbutton(label="Azimuth Lines", variable=self.show_azimuths,
                           command=self.refresh_plot)
        om.add_checkbutton(label="Grid Ticks",    variable=self.show_ticks,
                           command=self.refresh_plot)
        om.add_separator()
        om.add_command(label="Ring Interval…",    command=self._set_ring_interval)
        om.add_command(label="Azimuth Interval…", command=self._set_az_interval)
        om.add_command(label="Tick Interval…",    command=self._set_tick_interval)
        mb.add_cascade(label="Overlays", menu=om)

        prm = self._mk_menu(mb)
        prm.add_checkbutton(label="Velocity Dealiasing",
                            variable=self.do_dealias,
                            command=self._reprocess_and_reload)
        prm.add_checkbutton(label="KDP Calculation",
                            variable=self.do_kdp,
                            command=self._reprocess_and_reload)
        mb.add_cascade(label="Processing", menu=prm)

        self._fields_menu = self._mk_menu(mb)
        mb.add_cascade(label="Fields", menu=self._fields_menu)

        hm = self._mk_menu(mb)
        hm.add_command(label="Controls", command=self._show_help)
        hm.add_command(label="About",    command=self._show_about)
        mb.add_cascade(label="Help", menu=hm)

    def _rebuild_fields_menu(self):
        fm = self._fields_menu
        fm.delete(0, tk.END)
        if not self.available_fields:
            fm.add_command(label="(no file loaded)", state="disabled")
            return
        fm.add_command(label="── Assign field to panel ──", state="disabled")
        fm.add_separator()
        for fname in self.available_fields:
            info = FIELD_DEFAULTS.get(fname, (fname, -50, 50, "gurt_ref"))
            lbl, mn, mx, cm = info
            sub = self._mk_menu(fm)
            for slot in range(min(self.n_panels.get(), 5)):
                sub.add_command(
                    label=f"Panel {slot+1}",
                    command=lambda f=fname,l=lbl,a=mn,b=mx,c=cm,s=slot:
                            self._assign_field(s,f,l,a,b,c))
            sub.add_separator()
            sub.add_command(label="Edit colors…",
                            command=lambda f=fname,s=0: self._open_param_dialog(0, fname=f))
            fm.add_cascade(label=fname, menu=sub)
        fm.add_separator()
        fm.add_command(label="Reset to defaults", command=self._reset_fields)

    def _build_layout(self):
        self.status_var = tk.StringVar(value="No file loaded")

        sb = tk.Frame(self, bg="#d4d0c8", height=20)
        sb.pack(fill="x", side="bottom")
        sb.pack_propagate(False)
        self.cursor_var = tk.StringVar(value="Cursor: —")
        tk.Label(sb, textvariable=self.cursor_var,
                 bg="#d4d0c8", fg="#000", font=("Courier", 9),
                 anchor="w", padx=6).pack(side="left")
        self.file_var = tk.StringVar(value="")
        tk.Label(sb, textvariable=self.file_var,
                 bg="#d4d0c8", fg="#000", font=("Courier", 9),
                 anchor="e", padx=6).pack(side="right")
        sweep_frame = tk.Frame(sb, bg="#d4d0c8")
        sweep_frame.pack(side="right", padx=8)
        tk.Label(sweep_frame, text="Sweep:", bg="#d4d0c8", fg="#000",
                 font=("Courier", 9)).pack(side="left")
        self.sweep_spin = tk.Spinbox(sweep_frame, from_=0, to=20,
                                     textvariable=self.sweep_index, width=3,
                                     command=self.refresh_plot,
                                     bg="#ffffff", fg="#000",
                                     buttonbackground="#bbb")
        self.sweep_spin.pack(side="left", padx=2)

        self.plot_frame = tk.Frame(self, bg="#191970")
        self.plot_frame.pack(fill="both", expand=True)
        self.fig = plt.Figure(facecolor="#191970")
        self.fig.subplots_adjust(left=0, right=1, top=1, bottom=0,
                                 wspace=0, hspace=0)
        matplotlib.rcParams['axes.xmargin'] = 0
        matplotlib.rcParams['axes.ymargin'] = 0
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.plot_frame)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

        self.canvas.mpl_connect("motion_notify_event",  self._on_mouse_move)
        self.canvas.mpl_connect("button_press_event",   self._on_press)
        self.canvas.mpl_connect("button_release_event", self._on_release)
        self.canvas.mpl_connect("motion_notify_event",  self._on_drag)
        self.canvas.mpl_connect("resize_event",         self._on_resize)

        self.canvas.get_tk_widget().bind("<ButtonPress>", self._canvas_focus_grab, add=True)

    def _bind_keys(self):
        self.bind("<Left>",  lambda e: self._change_file(-1))
        self.bind("<Right>", lambda e: self._change_file(+1))
        self.bind("<Up>",    lambda e: self._change_sweep(+1))
        self.bind("<Down>",  lambda e: self._change_sweep(-1))
        self.bind("<Escape>",lambda e: self._reset_all_zoom())
        self.after(100, self.focus_set)

    def _on_resize(self, event=None):
        """Debounced reposition on window resize."""
        if hasattr(self, '_resize_after_id') and self._resize_after_id:
            self.after_cancel(self._resize_after_id)
        self._resize_after_id = self.after(60, self._do_resize)

    def _do_resize(self):
        self._resize_after_id = None
        self._reposition_axes()
        self.canvas.draw_idle()

    def _reprocess_and_reload(self):
        """Re-read the current file and re-apply (or skip) processing steps.
        Called when the user toggles dealiasing or KDP calculation."""
        if self.file_list:
            self.load_current_file()

    def _change_file(self, delta):
        if not self.file_list or getattr(self, '_single_file_mode', False):
            return
        self.file_index = (self.file_index + delta) % len(self.file_list)
        self._schedule_load()

    def _schedule_load(self):
        """Debounce rapid arrow-key file navigation (100 ms)."""
        if hasattr(self, '_load_after_id') and self._load_after_id:
            self.after_cancel(self._load_after_id)
        self._load_after_id = self.after(100, self._do_scheduled_load)

    def _do_scheduled_load(self):
        self._load_after_id = None
        self.load_current_file()

    def _change_sweep(self, delta):
        if self.radar is None:
            return
        new = max(0, min(self.radar.nsweeps-1, self.sweep_index.get()+delta))
        self.sweep_index.set(new)
        self._schedule_refresh()

    def _schedule_refresh(self):
        """Debounce rapid sweep changes (80 ms)."""
        if hasattr(self, '_refresh_after_id') and self._refresh_after_id:
            self.after_cancel(self._refresh_after_id)
        self._refresh_after_id = self.after(80, self.refresh_plot)

    def open_file(self):
        path = filedialog.askopenfilename(
            title="Open Radar File",
            filetypes=[("Radar files","*.nc *.h5 *.buf *.raw *.gz *.HDF5 *.hdf5 *.RAW"),
                       ("All files","*.*")])
        if path:
            self._single_file_mode = True
            self.file_list  = [os.path.abspath(path)]
            self.file_index = 0
            self.file_var.set(f"[1/1]  {os.path.basename(path)}")
            self.load_current_file()

    def open_folder(self):
        d = filedialog.askdirectory(title="Open Radar Folder")
        if d:
            self._single_file_mode = False
            self.load_folder(d)

    def load_folder(self, folder):
        self._single_file_mode = False
        files = []
        for root, _, fnames in os.walk(folder):
            for f in sorted(fnames):
                if any(f.lower().endswith(e) or f.endswith(e) for e in FILE_EXTENSIONS):
                    files.append(os.path.join(root, f))
        self.file_list = sorted(files)
        self.file_index = 0
        if self.file_list:
            self.load_current_file()
        else:
            messagebox.showwarning("No files", f"No radar files found in:\n{folder}")

    def load_current_file(self):
        if not self.file_list:
            return
        path = self.file_list[self.file_index]
        self.file_var.set(f"[{self.file_index+1}/{len(self.file_list)}]  "
                          f"{os.path.basename(path)}")
        self.title(f"GURT — {os.path.basename(path)}")
        if not PYART_AVAILABLE:
            messagebox.showerror("pyart missing","Install arm-pyart: pip install arm-pyart")
            return
        try:
            self.status_var.set(f"Loading {os.path.basename(path)}…")
            self.update_idletasks()
            self.radar = pyart.io.read(path, linear_interp=False)
            if self.do_kdp.get():
                self.radar = calculate_kdp(self.radar)
            if self.do_dealias.get():
                self.radar = dealias_velocity(self.radar)
            sweep = min(self.sweep_index.get(), self.radar.nsweeps-1)
            self.sweep_index.set(sweep)
            self.sweep_spin.config(to=self.radar.nsweeps-1)
            self.available_fields = list(self.radar.fields.keys())

            current_fields_valid = bool(self.panel_fields) and any(
                fi[0] in self.available_fields for fi in self.panel_fields if fi
            )

            if not current_fields_valid:
                self._auto_select_fields()
                self.panel_limits = [None] * self.n_panels.get()
            else:
                n = self.n_panels.get()
                while len(self.panel_fields) < n:
                    self.panel_fields.append(self.panel_fields[0])
                self.panel_fields = self.panel_fields[:n]
                while len(self.panel_limits) < n:
                    self.panel_limits.append(self.panel_limits[0]
                                             if self.panel_limits else None)
                self.panel_limits = self.panel_limits[:n]

            self._rebuild_fields_menu()
            self.refresh_plot()
        except Exception as e:
            messagebox.showerror("Load Error", f"Could not read {path}\n\n{e}")
            self.status_var.set("Error loading file.")

    def _auto_select_fields(self):
        priority = ['DBZH','DBZH1','reflectivity','DBZ','DBMHC',
                    'VELD','VEL','velocity','DP','KDP','ZDR','RHOHV']
        chosen = []
        seen   = set()
        for fname in priority:
            if fname in self.available_fields and fname not in seen:
                info = FIELD_DEFAULTS.get(fname, (fname,-50,50,"gurt_ref"))
                chosen.append((fname, *info))
                seen.add(fname)
                if len(chosen) >= self.n_panels.get():
                    break
        for fname in self.available_fields:
            if fname not in seen:
                info = FIELD_DEFAULTS.get(fname, (fname,-50,50,"gurt_ref"))
                chosen.append((fname, *info))
                seen.add(fname)
                if len(chosen) >= self.n_panels.get():
                    break
        self.panel_fields = chosen[:self.n_panels.get()]

    def _assign_field(self, slot, fname, label, vmin, vmax, cmap):
        cname = cmap if isinstance(cmap, str) else getattr(cmap,'name','gurt_ref')
        while len(self.panel_fields) <= slot:
            self.panel_fields.append(self.panel_fields[0] if self.panel_fields else
                                     (fname, label, vmin, vmax, cname))
        self.panel_fields[slot] = (fname, label, vmin, vmax, cname)
        self.refresh_plot()

    def _reset_fields(self):
        self.panel_fields = []
        self._auto_select_fields()
        self.refresh_plot()

    def _on_panels_changed(self):
        self._auto_select_fields()
        self.panel_limits = [None] * self.n_panels.get()
        self._rebuild_fields_menu()
        self.refresh_plot()

    def refresh_plot(self):
        if self.radar is None:
            return
        self.fig.clear()
        n     = self.n_panels.get()
        sweep = max(0, min(self.radar.nsweeps-1, self.sweep_index.get()))

        while len(self.panel_fields) < n:
            self.panel_fields.append(self.panel_fields[0] if self.panel_fields
                                     else ('reflectivity','Z_h',-20,70,'gurt_ref'))
        while len(self.panel_limits) < n:
            self.panel_limits.append(None)

        rname = self.radar.metadata.get('instrument_name','Unknown')
        t_str = self.radar.time['units'].split('since')[-1].strip()
        try:
            scan_time = datetime.strptime(t_str,'%Y/%m/%d %H:%M:%S')
            t_fmt = scan_time.strftime('%Y-%m-%dT%H:%M:%SZ')
        except Exception:
            t_fmt = t_str
        tilt = self.radar.fixed_angle['data'][sweep]

        if n == 1:
            hdr_fontsize = 13
            HEADER_TOP   = 0.995
        elif n == 2:
            hdr_fontsize = 11
            HEADER_TOP   = 0.995
        else:
            hdr_fontsize = 9
            HEADER_TOP   = 0.995

        CB_TOP   = 0.0
        PLOT_PAD = 0.006

        if n <= 3:
            rows, cols = 1, n
        else:
            rows, cols = 2, 3 if n == 5 else 2

        fig_w_px, fig_h_px = self.fig.get_size_inches() * self.fig.dpi
        _plot_left, _plot_right = 0.0, 1.0
        _avail_w = (_plot_right - _plot_left) * fig_w_px
        _avail_h = (HEADER_TOP - CB_TOP)      * fig_h_px
        _pad_px  = PLOT_PAD * fig_w_px
        _cell_w  = (_avail_w - _pad_px * (cols - 1)) / cols
        _cell_h  = (_avail_h - _pad_px * (rows - 1)) / rows
        _cell_aspect = _cell_w / _cell_h if _cell_h > 0 else 1.0

        display = pyart.graph.RadarDisplay(self.radar)
        mr      = self.max_range.get()
        self._axes        = []
        self._panel_meta  = []

        for idx, field_info in enumerate(self.panel_fields[:n]):
            if field_info is None:
                continue
            fname, flabel, vmin, vmax, cmap_name = field_info
            cmap = resolve_cmap(cmap_name)

            ax = self.fig.add_axes([0, 0, 1, 1])
            self._axes.append(ax)
            ax.set_facecolor("#191970")

            try:
                if fname not in self.radar.fields:
                    ax.text(0.5, 0.5, f"'{fname}'\nnot in file",
                            transform=ax.transAxes, ha='center', va='center',
                            color='white', fontsize=9)
                    ax.set_title(flabel, fontsize=9, color='white', pad=3)
                    for spine in ax.spines.values():
                        spine.set_visible(False)
                    ax.set_xticks([])
                    ax.set_yticks([])
                    self._panel_meta.append((0, 0, mr))
                    continue

                display.plot_ppi(fname, sweep=sweep, ax=ax,
                                 vmin=vmin, vmax=vmax, cmap=cmap,
                                 colorbar_flag=False, title='',
                                 axislabels_flag=False, edges=False,
                                 filter_transitions=False)

                lims = self.panel_limits[idx] if idx < len(self.panel_limits) else None
                if lims:
                    xl0, xl1, yl0, yl1 = lims
                else:
                    xl0, xl1, yl0, yl1 = -mr, mr, -mr, mr
                cx   = (xl0 + xl1) / 2.0
                cy   = (yl0 + yl1) / 2.0
                half = max(xl1 - xl0, yl1 - yl0) / 2.0

                ax.set_aspect('auto')
                if _cell_aspect >= 1.0:
                    x_half, y_half = half * _cell_aspect, half
                else:
                    x_half, y_half = half, half / _cell_aspect
                ax.set_xlim(cx - x_half, cx + x_half)
                ax.set_ylim(cy - y_half, cy + y_half)
                ax.margins(0, 0)
                ax.autoscale(False)
                ax.use_sticky_edges = False

                self._panel_meta.append((cx, cy, half))

                for spine in ax.spines.values():
                    spine.set_visible(False)
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_xlabel('')
                ax.set_ylabel('')

                self._draw_overlays(ax, half, cx, cy)

            except Exception as e:
                ax.text(0.5, 0.5, f"Error:\n{e}",
                        transform=ax.transAxes, ha='center', va='center',
                        color='red', fontsize=7)
                self._panel_meta.append((0, 0, mr))

        self._header_texts = []
        for idx in range(n):
            fi = self.panel_fields[idx] if idx < len(self.panel_fields) else None
            flabel = fi[1] if fi else ''
            panel_header = f"{t_fmt}   {rname}   {tilt:.1f}° SUR   {flabel}"
            ht = self.fig.text(
                0.5, 0.985, panel_header,
                ha='center', va='bottom',
                fontsize=hdr_fontsize, fontweight='bold', color='white',
                bbox=dict(boxstyle='square,pad=0.3',
                          facecolor='#191970', alpha=0.92, edgecolor='#191970'),
                transform=self.fig.transFigure, clip_on=False,
            )
            self._header_texts.append(ht)

        self._cb_axes = []
        for idx in range(min(n, len(self.panel_fields))):
            fi = self.panel_fields[idx]
            if fi is None:
                self._cb_axes.append(None)
                continue
            fname_i, flabel_i, vmin_i, vmax_i, cmap_name_i = fi
            cmap_i = resolve_cmap(cmap_name_i)
            cb_ax = self.fig.add_axes([0.05, _CB_BOTTOM, 0.90, _CB_HEIGHT])
            sm = plt.cm.ScalarMappable(
                norm=mcolors.Normalize(vmin=vmin_i, vmax=vmax_i), cmap=cmap_i)
            sm.set_array([])
            cb = self.fig.colorbar(sm, cax=cb_ax, orientation='horizontal')
            cb_ax.xaxis.set_ticks_position('top')
            cb_ax.xaxis.set_label_position('top')
            cb.ax.tick_params(labelsize=8, colors='white', labelcolor='white',
                              direction='out', length=4)
            cb.outline.set_edgecolor('white')
            cb_ax.set_facecolor('#191970')
            for lbl in cb_ax.get_xticklabels():
                lbl.set_bbox(dict(facecolor='#191970', alpha=0.7,
                                  edgecolor='none', pad=1.5))
            self._cb_axes.append(cb_ax)

        self._layout = dict(rows=rows, cols=cols, n=n,
                            header_top=HEADER_TOP, cb_top=CB_TOP,
                            pad=PLOT_PAD)

        self._reposition_axes()
        self.canvas.draw_idle()

    def _reposition_axes(self):
        """Position each panel to fill its cell rectangle completely.

        The axes uses set_aspect('auto') so it fills the rectangle without
        dead space.  Undistorted rendering is achieved by adjusting the DATA
        LIMITS to match the cell's pixel aspect ratio: if a cell is twice as
        wide as it is tall, the x data range is twice the y data range.
        This crops the PPI at the edges rather than squashing or padding it —
        the same visual effect as zooming/cropping a photo to fill a frame.
        """
        if not hasattr(self, '_layout') or not self._axes:
            return

        ly    = self._layout
        n     = ly['n']
        rows  = ly['rows']
        cols  = ly['cols']
        pad_f = ly['pad']

        fig_w, fig_h = self.fig.get_size_inches() * self.fig.dpi

        plot_left   = 0.0
        plot_right  = 1.0
        plot_top    = ly['header_top']
        plot_bottom = ly['cb_top']

        px_left   = plot_left   * fig_w
        px_right  = plot_right  * fig_w
        px_top    = plot_top    * fig_h
        px_bottom = plot_bottom * fig_h

        avail_w = px_right  - px_left
        avail_h = px_top    - px_bottom

        pad_px_x = pad_f * fig_w
        pad_px_y = pad_f * fig_h

        cell_w = (avail_w - pad_px_x * (cols - 1)) / cols
        cell_h = (avail_h - pad_px_y * (rows - 1)) / rows

        for idx, ax in enumerate(self._axes):
            if idx >= n:
                break
            row = idx // cols
            col = idx  % cols

            cell_x0 = px_left   + col * (cell_w + pad_px_x)
            cell_y0 = px_bottom + (rows - 1 - row) * (cell_h + pad_px_y)

            ax.set_position([cell_x0 / fig_w,
                             cell_y0 / fig_h,
                             cell_w  / fig_w,
                             cell_h  / fig_h])

            if idx < len(self._panel_meta):
                cx, cy, half = self._panel_meta[idx]
                aspect = cell_w / cell_h
                if aspect >= 1.0:
                    x_half = half * aspect
                    y_half = half
                else:
                    x_half = half
                    y_half = half / aspect
                ax.set_xlim(cx - x_half, cx + x_half)
                ax.set_ylim(cy - y_half, cy + y_half)

            cx_px    = cell_x0 + cell_w / 2.0
            cy_px    = cell_y0 + cell_h / 2.0
            cb_axes  = getattr(self, '_cb_axes', [])
            if idx < len(cb_axes) and cb_axes[idx] is not None:
                cb_h_px      = _CB_HEIGHT * fig_h
                cb_margin_px = 0.003 * fig_h

                if ly['n'] == 1:
                    cb_left_f = 0.02
                    cb_w_f    = 0.96
                    cb_bot_f  = _CB_BOTTOM
                    cb_h_f    = _CB_HEIGHT
                else:
                    cb_top_px = cell_y0 - cb_margin_px
                    cb_bot_px = cb_top_px - cb_h_px
                    if cb_bot_px < 0:
                        cb_bot_px = 1
                    cb_left_f = cell_x0 / fig_w
                    cb_w_f    = cell_w  / fig_w
                    cb_bot_f  = cb_bot_px / fig_h
                    cb_h_f    = cb_h_px   / fig_h

                cb_axes[idx].set_position([cb_left_f, cb_bot_f, cb_w_f, cb_h_f])
                for lbl in cb_axes[idx].get_xticklabels():
                    lbl.set_bbox(dict(facecolor='#191970', alpha=0.75,
                                      edgecolor='none', pad=1.5))

        header_texts = getattr(self, '_header_texts', [])
        for idx, ht in enumerate(header_texts):
            if idx >= len(self._axes):
                break
            ax  = self._axes[idx]
            pos = ax.get_position()
            mid_x = (pos.x0 + pos.x1) / 2.0
            top_y = pos.y1 - 0.002
            ht.set_transform(self.fig.transFigure)
            ht.set_position((mid_x, top_y))
            ht.set_horizontalalignment('center')
            ht.set_verticalalignment('top')

    def _draw_overlays(self, ax, half_range, cx=0.0, cy=0.0):
        """Draw range rings, azimuth lines, and cross-hair ticks.

        All geometry is batched into LineCollections (one draw call each).
        Intervals are auto-scaled upward when they would produce more than a
        small number of elements, keeping rendering fast regardless of settings.
        """
        from matplotlib.collections import LineCollection

        MAX_RINGS = 20
        MAX_AZ    = 36
        MAX_TICKS = 20

        corners_r = max(
            np.sqrt((cx + sx * half_range)**2 + (cy + sy * half_range)**2)
            for sx in (-1, 1) for sy in (-1, 1)
        )
        draw_r = corners_r * 1.05
        tick_s = half_range * 0.016

        if self.show_rings.get():
            ri = max(self.ring_interval.get(), 0.1)
            while draw_r / ri > MAX_RINGS:
                ri *= 2
            RING_PTS = 180
            theta = np.linspace(0, 2 * np.pi, RING_PTS, endpoint=False)
            cos_t, sin_t = np.cos(theta), np.sin(theta)
            ring_segs = []
            r = ri
            while r <= draw_r:
                pts = np.column_stack([
                    np.append(r * cos_t, r * cos_t[0]),
                    np.append(r * sin_t, r * sin_t[0])
                ])
                ring_segs.append(pts)
                label_x, label_y = 0.0, r
                if (cx - half_range <= label_x <= cx + half_range and
                        cy - half_range <= label_y <= cy + half_range):
                    ax.text(label_x, label_y, f'{r:.0f}',
                            ha='center', va='bottom',
                            color='white', fontsize=6, fontweight='bold',
                            bbox=dict(facecolor='#191970', alpha=0.5,
                                      edgecolor='none', pad=0.5))
                r += ri
            if ring_segs:
                ax.add_collection(LineCollection(ring_segs, colors='white',
                    alpha=0.55, linewidths=0.7, linestyles='dashed'))

        if self.show_azimuths.get():
            ai = max(self.az_interval.get(), 0.1)
            while (360 / ai) > MAX_AZ:
                ai *= 2
            az_angles = np.arange(0, 360, ai)
            rads  = np.radians(90 - az_angles)
            cos_r, sin_r = np.cos(rads), np.sin(rads)
            az_segs = [np.array([[0, 0], [draw_r * c, draw_r * s]])
                       for c, s in zip(cos_r, sin_r)]
            if az_segs:
                ax.add_collection(LineCollection(az_segs, colors='white',
                    alpha=0.45, linewidths=0.4, linestyles='dashed'))
            lr = draw_r * 0.92
            for az, c, s in zip(az_angles, cos_r, sin_r):
                lx, ly = lr * c, lr * s
                if (cx - half_range <= lx <= cx + half_range and
                        cy - half_range <= ly <= cy + half_range):
                    ax.text(lx, ly, f'{az:.0f}°',
                            ha='center', va='center', color='white',
                            fontsize=6, fontweight='bold',
                            bbox=dict(facecolor='#191970', alpha=0.5,
                                      edgecolor='none', pad=0.5))

        if self.show_ticks.get():
            ti = max(self.tick_interval.get(), 0.1)
            xlim, ylim = ax.get_xlim(), ax.get_ylim()
            span = max(xlim[1] - xlim[0], ylim[1] - ylim[0])
            while span / ti > MAX_TICKS:
                ti *= 2
            xs = np.arange(np.ceil(xlim[0] / ti) * ti, xlim[1] + ti, ti)
            ys = np.arange(np.ceil(ylim[0] / ti) * ti, ylim[1] + ti, ti)
            tick_segs = []
            for x in xs:
                for y in ys:
                    tick_segs.append([[x - tick_s, y], [x + tick_s, y]])
                    tick_segs.append([[x, y - tick_s], [x, y + tick_s]])
            if tick_segs:
                ax.add_collection(LineCollection(tick_segs, colors='white',
                    alpha=0.8, linewidths=0.7))

    def _canvas_focus_grab(self, event=None):
        """Take keyboard focus when the canvas is clicked.
        This closes any open menubar menu on platforms that need an explicit
        focus change, without generating synthetic events that break matplotlib."""
        self.canvas.get_tk_widget().focus_set()

    def _panel_index_for_ax(self, ax):
        for i, a in enumerate(getattr(self, '_axes', [])):
            if a is ax:
                return i
        return None

    def _on_press(self, event):
        if event.button == 1 and event.inaxes:
            self._drag_start = (event.xdata, event.ydata)
            self._drag_ax    = event.inaxes
            self._drag_rect  = Rectangle(
                (event.xdata, event.ydata), 0, 0,
                linewidth=1.5, edgecolor='yellow', facecolor='yellow',
                alpha=0.15, linestyle='--')
            event.inaxes.add_patch(self._drag_rect)
            self.canvas.draw_idle()

        elif event.button == 3:
            if event.inaxes:
                pidx = self._panel_index_for_ax(event.inaxes)
                if pidx is not None:
                    self._right_click_panel = pidx
                    self._show_right_click_menu(event, pidx)

    def _on_drag(self, event):
        if self._drag_start is None or self._drag_rect is None:
            return
        if event.inaxes is not self._drag_ax:
            return
        x0, y0 = self._drag_start
        x1, y1 = event.xdata, event.ydata
        self._drag_rect.set_bounds(min(x0,x1), min(y0,y1),
                                   abs(x1-x0), abs(y1-y0))
        self.canvas.draw_idle()

    def _on_release(self, event):
        if event.button != 1 or self._drag_start is None:
            self._drag_start = None
            return
        x0, y0 = self._drag_start
        self._drag_start = None

        if self._drag_rect is not None:
            self._drag_rect.remove()
            self._drag_rect = None

        if event.inaxes is None or event.inaxes is not self._drag_ax:
            self.canvas.draw_idle()
            return

        x1, y1 = event.xdata, event.ydata
        dx, dy  = abs(x1-x0), abs(y1-y0)

        xlim = self._drag_ax.get_xlim()
        ylim = self._drag_ax.get_ylim()
        view_w = xlim[1]-xlim[0]
        view_h = ylim[1]-ylim[0]
        if dx < view_w * 0.02 or dy < view_h * 0.02:
            self.canvas.draw_idle()
            return

        pidx = self._panel_index_for_ax(self._drag_ax)
        if pidx is None:
            return
        new_lims = [min(x0,x1), max(x0,x1), min(y0,y1), max(y0,y1)]
        n = self.n_panels.get()
        self.panel_limits = [new_lims[:] for _ in range(n)]
        self.refresh_plot()

    def _show_right_click_menu(self, mpl_event, panel_idx):
        menu = self._mk_menu(self)
        fname = self.panel_fields[panel_idx][0] if panel_idx < len(self.panel_fields) else ""
        menu.add_command(label=f"Panel {panel_idx+1}: {fname}",
                         state="disabled",
                         font=("TkDefaultFont",9,"bold"))
        menu.add_separator()
        menu.add_command(label="Edit Parameters & Colors…",
                         command=lambda: self._open_param_dialog(panel_idx))
        menu.add_separator()
        menu.add_command(label="Reset Zoom (this panel)",
                         command=lambda: self._reset_panel_zoom(panel_idx))
        menu.add_command(label="Reset Zoom (all panels)",
                         command=self._reset_all_zoom)
        menu.add_separator()
        cmap_sub = self._mk_menu(menu)
        for group, names in [
            ("GURT", list(GURT_CMAPS.keys())),
            ("Diverging", ["RdBu_r","RdYlBu_r","coolwarm","bwr","Spectral_r","PuOr","PRGn"]),
            ("Sequential", ["viridis","plasma","inferno","magma","turbo","Greys","hot","jet"]),
        ]:
            cmap_sub.add_command(label=f"── {group} ──", state="disabled")
            for cn in names:
                cmap_sub.add_command(
                    label=cn,
                    command=lambda c=cn,s=panel_idx: self._quick_set_cmap(s, c))
        menu.add_cascade(label="Quick Colormap", menu=cmap_sub)

        widget = self.canvas.get_tk_widget()
        rx = widget.winfo_rootx() + int(mpl_event.x)
        ry = widget.winfo_rooty() + int(self.fig.get_size_inches()[1]*self.fig.dpi
                                        - mpl_event.y)
        try:
            menu.tk_popup(rx, ry)
        finally:
            menu.grab_release()
            self.after(100, self.focus_set)

    def _quick_set_cmap(self, slot, cname):
        if slot < len(self.panel_fields) and self.panel_fields[slot]:
            fname, lbl, mn, mx, _ = self.panel_fields[slot]
            self.panel_fields[slot] = (fname, lbl, mn, mx, cname)
            self.refresh_plot()

    def _open_param_dialog(self, slot, fname=None):
        if not self.available_fields:
            messagebox.showinfo("No file","Load a radar file first.")
            return
        if slot >= len(self.panel_fields) or self.panel_fields[slot] is None:
            info = FIELD_DEFAULTS.get(
                self.available_fields[0] if self.available_fields else "",
                (self.available_fields[0] if self.available_fields else "field",
                 -50, 50, "gurt_ref"))
            field_info = (self.available_fields[0] if self.available_fields else "",
                          *info)
        else:
            field_info = self.panel_fields[slot]

        if fname and fname in self.available_fields:
            info = FIELD_DEFAULTS.get(fname, (fname,-50,50,"gurt_ref"))
            field_info = (fname, *info)

        def on_apply(s, fn, lbl, mn, mx, cn):
            while len(self.panel_fields) <= s:
                self.panel_fields.append(field_info)
            self.panel_fields[s] = (fn, lbl, mn, mx, cn)
            self.refresh_plot()

        ParamColorDialog(self, slot, field_info, self.available_fields, on_apply)

    def _reset_panel_zoom(self, idx):
        self.panel_limits = [None] * self.n_panels.get()
        self.refresh_plot()

    def _reset_all_zoom(self):
        self.panel_limits = [None] * self.n_panels.get()
        self.refresh_plot()

    def _zoom_all(self, factor):
        if factor == "data":
            self.max_range.set(150)
        elif isinstance(factor, float):
            self.max_range.set(self.max_range.get() * factor)
        else:
            self.max_range.set(float(factor))
        self._reset_all_zoom()

    def _set_max_range(self):
        self._ask_float("Max range (km):", self.max_range,
                        lambda: (self._reset_all_zoom(), self.refresh_plot()))

    def _center_on_click(self):
        lc = getattr(self, '_last_click', None)
        if lc is None:
            return
        cx, cy = lc
        mr = self.max_range.get()
        new = [cx-mr, cx+mr, cy-mr, cy+mr]
        self.panel_limits = [new] * self.n_panels.get()
        self.refresh_plot()

    def _set_ring_interval(self):
        self._ask_float("Ring interval (km):", self.ring_interval, self.refresh_plot)

    def _set_az_interval(self):
        self._ask_float("Azimuth interval (°):", self.az_interval, self.refresh_plot)

    def _set_tick_interval(self):
        self._ask_float("Tick interval (km):", self.tick_interval, self.refresh_plot)

    def _ask_float(self, prompt, var, callback=None):
        win = tk.Toplevel(self)
        win.title("Set value")
        win.resizable(False, False)
        win.grab_set()
        tk.Label(win, text=prompt, padx=10, pady=8).pack()
        entry = tk.Entry(win, width=12)
        entry.insert(0, str(var.get()))
        entry.pack(padx=10, pady=4)
        entry.focus_set()
        def ok(event=None):
            try:
                var.set(float(entry.get()))
                if callback:
                    callback()
            except ValueError:
                pass
            win.destroy()
        entry.bind("<Return>", ok)
        tk.Button(win, text="OK", command=ok, width=8).pack(pady=6)

    def save_image(self):
        if self.radar is None:
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG","*.png"),("SVG","*.svg"),("PDF","*.pdf")])
        if path:
            self.fig.savefig(path, dpi=200,
                             facecolor=self.fig.get_facecolor(),
                             bbox_inches='tight')
            messagebox.showinfo("Saved", f"Saved to {path}")

    def save_all_images(self):
        if not self.file_list:
            return
        out = filedialog.askdirectory(title="Output folder")
        if not out:
            return
        orig = self.file_index
        for i, fpath in enumerate(self.file_list):
            self.file_index = i
            self.load_current_file()
            name = os.path.splitext(os.path.basename(fpath))[0]
            op   = os.path.join(out, f"{name}_sweep{self.sweep_index.get()}.png")
            self.fig.savefig(op, dpi=150,
                             facecolor=self.fig.get_facecolor(),
                             bbox_inches='tight')
        self.file_index = orig
        self.load_current_file()
        messagebox.showinfo("Done", f"Saved {len(self.file_list)} images to {out}")

    def _on_mouse_move(self, event):
        if event.inaxes:
            x, y = event.xdata, event.ydata
            az   = (np.degrees(np.arctan2(x, y)) + 360) % 360
            rng  = np.sqrt(x**2 + y**2)
            self.cursor_var.set(
                f"X: {x:.1f} km   Y: {y:.1f} km   "
                f"Az: {az:.1f}°   Range: {rng:.1f} km")
            self._last_click = (x, y)
        else:
            self.cursor_var.set("Cursor: —")

    def _show_help(self):
        messagebox.showinfo("GURT Controls",
            "Keyboard\n"
            "────────────────────────────────\n"
            "← / →     Previous / Next file\n"
            "↑ / ↓     Higher / Lower sweep\n"
            "Escape    Reset all panel zoom\n\n"
            "Mouse\n"
            "────────────────────────────────\n"
            "Left-drag  Rubber-band zoom into selection\n"
            "Right-click  Parameter & Colors editor,\n"
            "             Quick colormap, Reset zoom\n\n"
            "Menus\n"
            "────────────────────────────────\n"
            "Fields   Assign field to panel; edit via Fields menu\n"
            "Overlays Toggle rings / azimuths / ticks + set intervals\n"
            "Config   1-5 panel layouts\n")

    def _show_about(self):
        messagebox.showinfo("About GURT",
            "GURT — Graphic Utility Radar Toolkit\n"
            "GUI Edition  v4.0\n"
            "@multidpppler\n\n"
            "pyart + matplotlib + tkinter")

def main():
    parser = argparse.ArgumentParser(description="GURT GUI — Radar Viewer")
    grp = parser.add_mutually_exclusive_group()
    grp.add_argument("--file",   help="Single radar file to open")
    grp.add_argument("--folder", help="Folder of radar files to load")
    parser.add_argument("--panels", type=int, choices=[1,2,3,4,5], default=1)
    parser.add_argument("--sweep",  type=int, default=0)
    parser.add_argument("--range",  type=float, default=150)
    args = parser.parse_args()

    app = GURTApp(folder=args.folder, file=args.file)
    app.n_panels.set(args.panels)
    app.sweep_index.set(args.sweep)
    app.max_range.set(args.range)
    app.mainloop()

if __name__ == "__main__":
    main()
