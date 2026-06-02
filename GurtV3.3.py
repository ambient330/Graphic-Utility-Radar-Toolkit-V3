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
from datetime import datetime, timezone
from fractions import Fraction
from tkinter import colorchooser

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

DORADE_AVAILABLE = True
import struct
import re
import numpy as np

class DoradeFile:
    def __init__(self, filename, endian="<"):
        if hasattr(filename, "read"):
            self._fh = filename
        else:
            self._fh = open(filename, "rb")

        self.buf = self._fh.read()
        #self.endian = endian
        self.endian = self._detect_endian()

        self.blocks = []
        self.params = {}
        self.rdat = []
        self.rays = []
        

        self._parse()

    def _detect_endian(self):
        buf = self.buf
        block_id = buf[0:4]
        nbytes_le = struct.unpack("<i", buf[4:8])[0]
        nbytes_be = struct.unpack(">i", buf[4:8])[0]
        file_len = len(buf)
        le_ok = 0 < nbytes_le < file_len
        be_ok = 0 < nbytes_be < file_len
        if le_ok and not be_ok:
            return "<"
        elif be_ok and not le_ok:
            return ">"
        elif be_ok and le_ok:
            if block_id in BLOCK_MAP:
                return ">"
            else:
                return "<"
        else:
            raise RuntimeError("Cannot determine endian (both invalid)")


    def _parse(self):
        pos = 0
        buf = self.buf
        while pos < len(buf):
            block_id = buf[pos:pos+4]
            if block_id == b"PARM":
                block = read_parm_block(buf, pos, self.endian)
                name = block["parameter_name"].decode(errors="ignore").strip("\x00").strip()
                scale = block["parameter_scale"]
                offset = block["parameter_bias"]
                bad_data = block["bad_data"]
                
                #print(bad_data)
                self.params[name] = block

            if block_id == b"RYIB":
                block = _unpack_from_buf(buf, pos, RYIB_BLOCK, self.endian)
                current_ray = {
                    "azimuth": block["azimuth"],
                    "elevation": block["elevation"],
                    "status": block["ray_status"],
                    "rdat": {}
                }
                self.rays.append(current_ray)

            elif block_id == b"RADD":
                block = _unpack_from_buf(buf, pos, RADD_BLOCK, self.endian)
                self.instrument_name = block["radar_name"].decode(errors="ignore").strip("\x00").strip()
                self.compression = block["data_compress"]
                self.lat = block["radar_latitude"]
                self.lon = block["radar_longitude"]
                self.alt = block["radar_altitude"] * 1000 # this is in km
                self.frequency = block["freq1"] # freq1 is always the usual frequency block with the true value, when working with DORADE data
                self.sweep_mode = _sweep_mode(block["scan_mode"])
                self.nyquist_velocity = block["eff_unamb_vel"]
                self.unambiguous_range = block["eff_unamb_range"]
                # Store PRTs + IPP count for stagger detection
                # Units: prt1/prt2 are in milliseconds (DoradeRadxFile.cc: prt / 1000.0)
                # num_ipps_trans >= 2 is required to declare stagger (Radx convention)
                self._radd_prt1     = block["prt1"]
                self._radd_prt2     = block["prt2"]
                self._radd_num_ipps = block["num_ipps_trans"]

            elif block_id == b"RDAT":
                block = _unpack_from_buf(buf, pos, RDAT_BLOCK, self.endian)
                header_size = struct.calcsize(self.endian + "4si8s")
                product_name = block["pdata_name"].decode(errors="ignore").strip("\x00").strip()
                data_length = block["nbytes"]
                raw = buf[pos + header_size : pos + data_length]
                data = np.frombuffer(raw, dtype=self.endian + "i2")

                if self.compression == 1:  # COMPRESSION_HRD
                    if hasattr(self, "dist_cells") and self.dist_cells is not None:
                        ngates = len(self.dist_cells)
                    elif product_name in self.params and "number_cells" in self.params[product_name]:
                        ngates = self.params[product_name]["number_cells"]
                    else:
                        raise RuntimeError(f"Cannot determine ngates for compressed param '{product_name}'")
                    data = self._rle_decode_row(data, ngates)

                data = data.astype("float32")
                if current_ray is not None:
                    current_ray["rdat"][product_name] = data

                
            elif block_id == b"SSWB":
                block = _unpack_from_buf(buf, pos, SSWB_BLOCK, self.endian)
                volume_time = datetime.fromtimestamp(block["volume_time_stamp"], tz=timezone.utc).replace(tzinfo=None)
                #radar_name = block["radar_name"]
                self.sweep_start_time = datetime.fromtimestamp(block["start_time"], tz=timezone.utc).replace(tzinfo=None)
                self.sweep_stop_time = datetime.fromtimestamp(block["stop_time"], tz=timezone.utc).replace(tzinfo=None)

            elif block_id == b"VOLD":
                block = _unpack_from_buf(buf, pos, VOLD_BLOCK, self.endian)
                self.volume_number = block["volume_num"]

                def get_volume_datetime():
                    year = block["year"]
                    month = block["month"]
                    day = block["day"]
                    hour = block["data_set_hour"]
                    minute = block["data_set_minute"]
                    second = block["data_set_second"]

                    return datetime(year, month, day, hour, minute, second)
                
            elif block_id == b"SWIB":
                block = _unpack_from_buf(buf, pos, SWIB_BLOCK, self.endian)
                self.sweep_number = block["sweep_num"]
                start_angle = block["start_angle"]
                end_angle = block["stop_angle"]
                self.fixed_angle = block["fixed_angle"]
                self.num_rays = block["num_rays"]

            elif block_id == b"RYIB":
                block = _unpack_from_buf(buf, pos, RYIB_BLOCK, self.endian)
                azimuth = block["azimuth"]
                elevation = block["elevation"]
                ray_status = _ray_status(block["ray_status"])
                #print(ray_status)

                self.rays.append({"azimuth": block["azimuth"], "elevation": block["elevation"], "status": block["ray_status"]})

            elif block_id == b"CELV":
                block = _unpack_from_buf(buf, pos, CELV_BLOCK, self.endian)
                ncells = block["number_cells"]
                raw_dist = block["dist_cells"][:ncells * 4]
                self.dist_cells = np.frombuffer(raw_dist, dtype=self.endian + "f4").copy()
                
            

            elif block_id in BLOCK_MAP:
                block_def = BLOCK_MAP[block_id]
                block = _unpack_from_buf(buf, pos, block_def, self.endian)
                self.blocks.append((block_id, block))

            nbytes = block["nbytes"]
            pos += nbytes

    def get_param(self, name):
        return self.params.get(name)
    
    def _rle_decode_row(self, data_comp, ngates):
        decoded = np.full(ngates, -32768, dtype=np.int16)
        comp_be = data_comp.byteswap().view(np.uint16)
        comp_le = data_comp.view(np.uint16)
        def decode(comp):
            out = np.full(ngates, -32768, dtype=np.int16)
            i = j = 0
            while i < len(comp) and j < ngates:
                val = int(comp[i])
                if val <= 0:
                    i += 1
                    continue
                n = val & 0x7FFF
                is_literal = val & 0x8000
                if n <= 0 or n > ngates:
                    return None
                if is_literal:
                    available = len(data_comp) - (i + 1)
                    if available <= 0:
                        return None
                    n = min(n, available, ngates - j)
                    try:
                        out[j:j+n] = data_comp[i+1:i+1+n]
                    except ValueError:
                        return None
                    i += 1 + n
                else:
                    n = min(n, ngates - j)
                    out[j:j+n] = -32768
                    i += 1
                j += n
            return out

        d1 = decode(comp_be)
        d2 = decode(comp_le)

        if d1 is None and d2 is None:
            raise RuntimeError("Both RLE decodes failed")
        if d1 is None:
            return d2
        if d2 is None:
            return d1
        score1 = np.sum(d1 != -32768)
        score2 = np.sum(d2 != -32768)

        return d1 if score1 >= score2 else d2
    
    def get_sweep(self, param_name):
        parm = self.params.get(param_name)
        if parm is None:
            raise KeyError(f"Product '{param_name}' not found")

        scale = parm["parameter_scale"]
        bias  = parm["parameter_bias"]
        bad   = parm["bad_data"]

        azimuth   = []
        elevation = []
        data      = []

        for ray in self.rays:
            if param_name not in ray["rdat"]:
                continue
            raw = ray["rdat"][param_name]
            masked = np.where(raw == bad, np.nan, (raw - bias) / scale)
            azimuth.append(ray["azimuth"])
            elevation.append(ray["elevation"])
            data.append(masked)

        # DORADE products will have different shapes at times, so as a fallback we find the product with the largest shape, and pad them with bad data (it gets filtered out)
        target_len = max(len(d) for d in data)
        padded = []
        for d in data:
            if len(d) < target_len:
                pad = np.full(target_len - len(d), bad, dtype="float32")
                padded.append(np.concatenate([d, pad]))
            else:
                padded.append(d)

        data_arr = np.vstack(padded)
        ngates = data_arr.shape[1]

        if "meters_to_first_cell" in parm and parm["meters_to_first_cell"] != 0:
            r0      = parm["meters_to_first_cell"]
            spacing = parm["meters_between_cells"]
        elif hasattr(self, "dist_cells") and self.dist_cells is not None:
            r0      = self.dist_cells[0]
            spacing = self.dist_cells[1] - self.dist_cells[0]
        else:
            raise RuntimeError("No range information available")

        ranges = r0 + np.arange(ngates) * spacing

        return {
            "azimuth":    np.array(azimuth),
            "elevation":  np.array(elevation),
            "data":       data_arr,
            "range":      ranges,
            "radar_name": self.instrument_name,
            "start_time": self.sweep_start_time,
            "stop_time":  self.sweep_stop_time,
        }

def read_parm_block(buf, pos, endian="<"):
    header = struct.unpack(endian + "4s i", buf[pos:pos+8])
    block_id, nbytes = header

    if block_id != b"PARM":
        raise ValueError("Not a PARM block")

    if nbytes == 104:
        block_def = PARM_BLOCK_104
    elif nbytes == 216:
        block_def = PARM_BLOCK_216
    else:
        raise ValueError(f"Unknown PARM size: {nbytes}")

    return _unpack_from_buf(buf, pos, block_def, endian)

def _sweep_mode(value):
    """Convert DORADE scan_mode integer to a CfRadial-standard sweep mode string.

    Mapping follows Radx::sweepModeToStr() exactly so that the string written
    into sweep_mode['data'] of the pyart Radar object (and ultimately into any
    CfRadial file) matches what RadxConvert / Radx-based tools expect.

    DORADE scan_mode values (from dorade_system_codes.h):
      0 = Calibration
      1 = PPI / sector
      2 = Coplane
      3 = RHI
      4 = Vertical pointing
      5 = Target / pointing
      6 = Manual
      7 = Idle
      8 = Surveillance (full 360° PPI)
    """
    return {
        0: "calibration",
        1: "sector",
        2: "coplane",
        3: "rhi",
        4: "vertical_pointing",
        5: "pointing",
        6: "manual_ppi",
        7: "idle",
        8: "azimuth_surveillance",
    }.get(value, "azimuth_surveillance")

def _ray_status(value):
    return {
        0: "normal",
        1: "transition",
        2: "bad",
    }[value]

def _structure_size(structure, endian=">"):
    return struct.calcsize(endian + "".join([i[1] for i in structure]))


def _unpack_from_buf(buf, pos, structure, endian=">"):
    size = _structure_size(structure, endian)
    return _unpack_structure(buf[pos : pos + size], structure, endian)


def _unpack_structure(string, structure, endian=">"):
    fmt = endian + "".join([i[1] for i in structure])
    lst = struct.unpack(fmt, string)
    return dict(zip([i[0] for i in structure], lst))

BYTE = "B"
INT1 = "B"
INT2 = "H"
INT4 = "I"
REAL4 = "f"
REAL8 = "d"
SINT1 = "b"
SINT2 = "h"
SINT4 = "i"

# below claude did the block organizing, saved me a lot of time

# ── SSWB  (196 bytes) ────────────────────────────────────────────────────────
SSWB_BLOCK = (
    ("id",                   "4s"),
    ("nbytes",               SINT4),
    ("last_used",            SINT4),   # Unix time; 0 = never age off
    ("start_time",           SINT4),   # Unix time
    ("stop_time",            SINT4),   # Unix time
    ("sizeof_file",          SINT4),
    ("compression_flag",     SINT4),
    ("volume_time_stamp",    SINT4),
    ("num_params",           SINT4),
    ("radar_name",           "8s"),
    ("d_start_time",         REAL8),   # high-precision volume start
    ("d_stop_time",          REAL8),   # high-precision volume stop
    ("version_num",          SINT4),
    ("num_key_tables",       SINT4),
    ("status",               SINT4),
    ("place_holder",         "28s"),   # 7 × si32 unused
    # key_table[0]
    ("key_table_0_offset",   SINT4),
    ("key_table_0_size",     SINT4),
    ("key_table_0_type",     SINT4),
    # key_table[1]
    ("key_table_1_offset",   SINT4),
    ("key_table_1_size",     SINT4),
    ("key_table_1_type",     SINT4),
    # key_table[2]
    ("key_table_2_offset",   SINT4),
    ("key_table_2_size",     SINT4),
    ("key_table_2_type",     SINT4),
    # key_table[3]
    ("key_table_3_offset",   SINT4),
    ("key_table_3_size",     SINT4),
    ("key_table_3_type",     SINT4),
    # key_table[4]
    ("key_table_4_offset",   SINT4),
    ("key_table_4_size",     SINT4),
    ("key_table_4_type",     SINT4),
    # key_table[5]
    ("key_table_5_offset",   SINT4),
    ("key_table_5_size",     SINT4),
    ("key_table_5_type",     SINT4),
    # key_table[6]
    ("key_table_6_offset",   SINT4),
    ("key_table_6_size",     SINT4),
    ("key_table_6_type",     SINT4),
    # key_table[7]
    ("key_table_7_offset",   SINT4),
    ("key_table_7_size",     SINT4),
    ("key_table_7_type",     SINT4),
)

# ── COMM  (508 bytes) ────────────────────────────────────────────────────────
COMM_BLOCK = (
    ("id",      "4s"),
    ("nbytes",  SINT4),
    ("comment", "500s"),
)

# ── VOLD  (72 bytes) ─────────────────────────────────────────────────────────
VOLD_BLOCK = (
    ("id",                "4s"),
    ("nbytes",            SINT4),
    ("format_version",    SINT2),
    ("volume_num",        SINT2),
    ("maximum_bytes",     SINT4),
    ("proj_name",         "20s"),
    ("year",              SINT2),
    ("month",             SINT2),
    ("day",               SINT2),
    ("data_set_hour",     SINT2),
    ("data_set_minute",   SINT2),
    ("data_set_second",   SINT2),
    ("flight_num",        "8s"),
    ("gen_facility",      "8s"),
    ("gen_year",          SINT2),
    ("gen_month",         SINT2),
    ("gen_day",           SINT2),
    ("number_sensor_des", SINT2),
)

# ── RADD  (300 bytes) ────────────────────────────────────────────────────────
RADD_BLOCK = (
    ("id",                    "4s"),
    ("nbytes",                SINT4),
    ("radar_name",            "8s"),
    ("radar_const",           REAL4),
    ("peak_power",            REAL4),
    ("noise_power",           REAL4),
    ("receiver_gain",         REAL4),
    ("antenna_gain",          REAL4),
    ("system_gain",           REAL4),
    ("horz_beam_width",       REAL4),
    ("vert_beam_width",       REAL4),
    ("radar_type",            SINT2),
    ("scan_mode",             SINT2),
    ("req_rotat_vel",         REAL4),
    ("scan_mode_pram0",       REAL4),
    ("scan_mode_pram1",       REAL4),
    ("num_parameter_des",     SINT2),
    ("total_num_des",         SINT2),
    ("data_compress",         SINT2),
    ("data_reduction",        SINT2),
    ("data_red_parm0",        REAL4),
    ("data_red_parm1",        REAL4),
    ("radar_longitude",       REAL4),
    ("radar_latitude",        REAL4),
    ("radar_altitude",        REAL4),
    ("eff_unamb_vel",         REAL4),
    ("eff_unamb_range",       REAL4),
    ("num_freq_trans",        SINT2),
    ("num_ipps_trans",        SINT2),
    ("freq1",                 REAL4),
    ("freq2",                 REAL4),
    ("freq3",                 REAL4),
    ("freq4",                 REAL4),
    ("freq5",                 REAL4),
    ("prt1",                  REAL4),
    ("prt2",                  REAL4),
    ("prt3",                  REAL4),
    ("prt4",                  REAL4),
    ("prt5",                  REAL4),
)

# ── CFAC  (72 bytes) ─────────────────────────────────────────────────────────
CFAC_BLOCK = (
    ("id",                "4s"),
    ("nbytes",            SINT4),
    ("azimuth_corr",      REAL4),
    ("elevation_corr",    REAL4),
    ("range_delay_corr",  REAL4),
    ("longitude_corr",    REAL4),
    ("latitude_corr",     REAL4),
    ("pressure_alt_corr", REAL4),
    ("radar_alt_corr",    REAL4),
    ("ew_gndspd_corr",    REAL4),
    ("ns_gndspd_corr",    REAL4),
    ("vert_vel_corr",     REAL4),
    ("heading_corr",      REAL4),
    ("roll_corr",         REAL4),
    ("pitch_corr",        REAL4),
    ("drift_corr",        REAL4),
    ("rot_angle_corr",    REAL4),
    ("tilt_corr",         REAL4),
)

PARM_BLOCK_104 = (
    ("id",                    "4s"),
    ("nbytes",                SINT4),
    ("parameter_name",        "8s"),
    ("param_description",     "40s"),
    ("param_units",           "8s"),
    ("interpulse_time",       SINT2),
    ("xmitted_freq",          SINT2),
    ("recvr_bandwidth",       REAL4),
    ("pulse_width",           SINT2),
    ("polarization",          SINT2),
    ("num_samples",           SINT2),
    ("binary_format",         SINT2),
    ("threshold_field",       "8s"),
    ("threshold_value",       REAL4),
    ("parameter_scale",       REAL4),
    ("parameter_bias",        REAL4),
    ("bad_data",              SINT4),
)

PARM_BLOCK_216 = (
    ("id",                    "4s"),
    ("nbytes",                SINT4),
    ("parameter_name",        "8s"),
    ("param_description",     "40s"),
    ("param_units",           "8s"),
    ("interpulse_time",       SINT2),
    ("xmitted_freq",          SINT2),
    ("recvr_bandwidth",       REAL4),
    ("pulse_width",           SINT2),
    ("polarization",          SINT2),
    ("num_samples",           SINT2),
    ("binary_format",         SINT2),
    ("threshold_field",       "8s"),
    ("threshold_value",       REAL4),
    ("parameter_scale",       REAL4),
    ("parameter_bias",        REAL4),
    ("bad_data",              SINT4),
    ("extension_num",         SINT4),
    ("config_name",           "8s"),
    ("config_num",            SINT4),
    ("offset_to_data",        SINT4),
    ("mks_conversion",        REAL4),
    ("num_qnames",            SINT4),
    ("qdata_names",           "32s"),
    ("num_criteria",          SINT4),
    ("criteria_names",        "32s"),
    ("number_cells",          SINT4),
    ("meters_to_first_cell",  REAL4),
    ("meters_between_cells",  REAL4),
    ("eff_unamb_vel",         REAL4),
)

# ── CELV  (6012 bytes) ───────────────────────────────────────────────────────
CELV_BLOCK = (
    ("id",           "4s"),
    ("nbytes",       SINT4),
    ("number_cells", SINT4),
    ("dist_cells",   "6000s"),  # fl32[1500] — parse separately
)

# ── CSFD  (64 bytes) ─────────────────────────────────────────────────────────
CSFD_BLOCK = (
    ("id",            "4s"),
    ("nbytes",        SINT4),
    ("num_segments",  SINT4),
    ("dist_to_first", REAL4),
    ("spacing",       "32s"),   # fl32[8]
    ("num_cells",     "16s"),   # si16[8]
)

# ── SWIB  (40 bytes) ─────────────────────────────────────────────────────────
SWIB_BLOCK = (
    ("id",          "4s"),
    ("nbytes",      SINT4),
    ("radar_name",  "8s"),
    ("sweep_num",   SINT4),
    ("num_rays",    SINT4),
    ("start_angle", REAL4),
    ("stop_angle",  REAL4),
    ("fixed_angle", REAL4),
    ("filter_flag", SINT4),
)

# ── ASIB  (80 bytes) ─────────────────────────────────────────────────────────
ASIB_BLOCK = (
    ("id",              "4s"),
    ("nbytes",          SINT4),
    ("longitude",       REAL4),
    ("latitude",        REAL4),
    ("altitude_msl",    REAL4),
    ("altitude_agl",    REAL4),
    ("ew_velocity",     REAL4),
    ("ns_velocity",     REAL4),
    ("vert_velocity",   REAL4),
    ("heading",         REAL4),
    ("roll",            REAL4),
    ("pitch",           REAL4),
    ("drift_angle",     REAL4),
    ("rotation_angle",  REAL4),
    ("tilt",            REAL4),
    ("ew_horiz_wind",   REAL4),
    ("ns_horiz_wind",   REAL4),
    ("vert_wind",       REAL4),
    ("heading_change",  REAL4),
    ("pitch_change",    REAL4),
)

# ── RYIB  (44 bytes) ─────────────────────────────────────────────────────────
RYIB_BLOCK = (
    ("id",             "4s"),
    ("nbytes",         SINT4),
    ("sweep_num",      SINT4),
    ("julian_day",     SINT4),
    ("hour",           SINT2),
    ("minute",         SINT2),
    ("second",         SINT2),
    ("millisecond",    SINT2),
    ("azimuth",        REAL4),
    ("elevation",      REAL4),
    ("peak_power",     REAL4),
    ("true_scan_rate", REAL4),
    ("ray_status",     SINT4),
)

# ── RDAT  (16 bytes header; field data follows) ──────────────────────────────
RDAT_BLOCK = (
    ("id",         "4s"),
    ("nbytes",     SINT4),
    ("pdata_name", "8s"),
)

# ── QDAT  (56 bytes header; field data follows) ──────────────────────────────
QDAT_BLOCK = (
    ("id",              "4s"),
    ("nbytes",          SINT4),
    ("pdata_name",      "8s"),
    ("extension_num",   SINT4),
    ("config_num",      SINT4),
    ("first_cell",      "8s"),   # si16[4]
    ("num_cells",       "8s"),   # si16[4]
    ("criteria_value",  "16s"),  # fl32[4]
)

# ── XSTF  (24 bytes) ─────────────────────────────────────────────────────────
XSTF_BLOCK = (
    ("id",                  "4s"),
    ("nbytes",              SINT4),
    ("one",                 SINT4),   # always 1 (endian flag)
    ("source_format",       SINT4),
    ("offset_to_first_item", SINT4),
    ("transition_flag",     SINT4),
)

# ── NULL  (8 bytes) ──────────────────────────────────────────────────────────
NULL_BLOCK = (
    ("id",     "4s"),
    ("nbytes", SINT4),
)

# ── RKTB  (28 bytes header) ──────────────────────────────────────────────────
RKTB_BLOCK = (
    ("id",                  "4s"),
    ("nbytes",              SINT4),
    ("angle2ndx",           REAL4),   # 360.0 / ndx_que_size
    ("ndx_que_size",        SINT4),
    ("first_key_offset",    SINT4),
    ("angle_table_offset",  SINT4),
    ("num_rays",            SINT4),
)

# ── rot_table_entry  (12 bytes, repeated num_rays times after RKTB) ──────────
ROT_TABLE_ENTRY = (
    ("rotation_angle", REAL4),
    ("offset",         SINT4),
    ("size",           SINT4),
)

# ── FRAD  (52 bytes) ─────────────────────────────────────────────────────────
FRAD_BLOCK = (
    ("id",                "4s"),
    ("nbytes",            SINT4),
    ("data_sys_status",   SINT4),
    ("radar_name",        "8s"),
    ("test_pulse_level",  REAL4),
    ("test_pulse_dist",   REAL4),
    ("test_pulse_width",  REAL4),
    ("test_pulse_freq",   REAL4),
    ("test_pulse_atten",  SINT2),
    ("test_pulse_fnum",   SINT2),
    ("noise_power",       REAL4),
    ("ray_count",         SINT4),
    ("first_rec_gate",    SINT2),
    ("last_rec_gate",     SINT2),
)

# ── FRIB  (264 bytes) ────────────────────────────────────────────────────────
FRIB_BLOCK = (
    ("id",                      "4s"),
    ("nbytes",                  SINT4),
    ("data_sys_id",             SINT4),
    ("loss_out",                REAL4),
    ("loss_in",                 REAL4),
    ("loss_rjoint",             REAL4),
    ("ant_v_dim",               REAL4),
    ("ant_h_dim",               REAL4),
    ("ant_noise_temp",          REAL4),
    ("r_noise_figure",          REAL4),
    ("xmit_power",              "20s"),  # fl32[5]
    ("x_band_gain",             REAL4),
    ("receiver_gain",           "20s"),  # fl32[5]
    ("if_gain",                 "20s"),  # fl32[5]
    ("conversion_gain",         REAL4),
    ("scale_factor",            "20s"),  # fl32[5]
    ("processor_const",         REAL4),
    ("dly_tube_antenna",        SINT4),
    ("dly_rndtrip_chip_atod",   SINT4),
    ("dly_timmod_testpulse",    SINT4),
    ("dly_modulator_on",        SINT4),
    ("dly_modulator_off",       SINT4),
    ("peak_power_offset",       REAL4),
    ("test_pulse_offset",       REAL4),
    ("E_plane_angle",           REAL4),
    ("H_plane_angle",           REAL4),
    ("encoder_antenna_up",      REAL4),
    ("pitch_antenna_up",        REAL4),
    ("indepf_times_flg",        SINT2),
    ("indep_freq_gate",         SINT2),
    ("time_series_gate",        SINT2),
    ("num_base_params",         SINT2),
    ("file_name",               "80s"),
)

# ── LIDR  (148 bytes) ────────────────────────────────────────────────────────
LIDR_BLOCK = (
    ("id",                "4s"),
    ("nbytes",            SINT4),
    ("lidar_name",        "8s"),
    ("lidar_const",       REAL4),
    ("pulse_energy",      REAL4),
    ("peak_power",        REAL4),
    ("pulse_width",       REAL4),
    ("aperture_size",     REAL4),
    ("field_of_view",     REAL4),
    ("aperture_eff",      REAL4),
    ("beam_divergence",   REAL4),
    ("lidar_type",        SINT2),
    ("scan_mode",         SINT2),
    ("req_rotat_vel",     REAL4),
    ("scan_mode_pram0",   REAL4),
    ("scan_mode_pram1",   REAL4),
    ("num_parameter_des", SINT2),
    ("total_num_des",     SINT2),
    ("data_compress",     SINT2),
    ("data_reduction",    SINT2),
    ("data_red_parm0",    REAL4),
    ("data_red_parm1",    REAL4),
    ("lidar_longitude",   REAL4),
    ("lidar_latitude",    REAL4),
    ("lidar_altitude",    REAL4),
    ("eff_unamb_vel",     REAL4),
    ("eff_unamb_range",   REAL4),
    ("num_wvlen_trans",   SINT4),
    ("prf",               REAL4),
    ("wavelength",        "40s"),  # fl32[10]
)

# ── FLIB  (748 bytes) ────────────────────────────────────────────────────────
FLIB_BLOCK = (
    ("id",                  "4s"),
    ("nbytes",              SINT4),
    ("data_sys_id",         SINT4),
    ("transmit_beam_div",   "40s"),  # fl32[10]
    ("xmit_power",          "40s"),  # fl32[10]
    ("receiver_fov",        "40s"),  # fl32[10]
    ("receiver_type",       "40s"),  # si32[10]
    ("r_noise_floor",       "40s"),  # fl32[10]
    ("receiver_spec_bw",    "40s"),  # fl32[10]
    ("receiver_elec_bw",    "40s"),  # fl32[10]
    ("calibration",         "40s"),  # fl32[10]
    ("range_delay",         SINT4),
    ("peak_power_multi",    "40s"),  # fl32[10]
    ("encoder_mirror_up",   REAL4),
    ("pitch_mirror_up",     REAL4),
    ("max_digitizer_count", SINT4),
    ("max_digitizer_volt",  REAL4),
    ("digitizer_rate",      REAL4),
    ("total_num_samples",   SINT4),
    ("samples_per_cell",    SINT4),
    ("cells_per_ray",       SINT4),
    ("pmt_temp",            REAL4),
    ("pmt_gain",            REAL4),
    ("apd_temp",            REAL4),
    ("apd_gain",            REAL4),
    ("transect",            SINT4),
    ("derived_names",       "120s"),  # char[10][12]
    ("derived_units",       "80s"),   # char[10][8]
    ("temp_names",          "120s"),  # char[10][12]
)

# ── SITU  (4108 bytes) ───────────────────────────────────────────────────────
SITU_BLOCK = (
    ("id",            "4s"),
    ("nbytes",        SINT4),
    ("number_params", SINT4),
    ("params",        "4096s"),  # insitu_parameter_t[256]: name[8]+units[8] each
)

# ── ISIT  (16 bytes) ─────────────────────────────────────────────────────────
ISIT_BLOCK = (
    ("id",         "4s"),
    ("nbytes",     SINT4),
    ("julian_day", SINT2),
    ("hours",      SINT2),
    ("minutes",    SINT2),
    ("seconds",    SINT2),
)

# ── INDF  (8 bytes) ──────────────────────────────────────────────────────────
INDF_BLOCK = (
    ("id",     "4s"),
    ("nbytes", SINT4),
)

# ── MINI  (4112 bytes) ───────────────────────────────────────────────────────
MINI_BLOCK = (
    ("id",           "4s"),
    ("nbytes",       SINT4),
    ("command",      SINT2),
    ("status",       SINT2),
    ("temperature",  REAL4),
    ("x_axis_gyro",  "512s"),   # fl32[128]
    ("y_axis_gyro",  "512s"),   # fl32[128]
    ("z_axis_gyro",  "512s"),   # fl32[128]
    ("xr_axis_gyro", "512s"),   # fl32[128]
    ("x_axis_vel",   "512s"),   # fl32[128]
    ("y_axis_vel",   "512s"),   # fl32[128]
    ("z_axis_vel",   "512s"),   # fl32[128]
    ("x_axis_pos",   "512s"),   # fl32[128]
)

# ── NDDS  (16 bytes) ─────────────────────────────────────────────────────────
NDDS_BLOCK = (
    ("id",             "4s"),
    ("nbytes",         SINT4),
    ("ins_flag",       SINT2),
    ("gps_flag",       SINT2),
    ("minirims_flag",  SINT2),
    ("kalman_flag",    SINT2),
)

# ── TIME  (8 bytes) ──────────────────────────────────────────────────────────
TIME_BLOCK = (
    ("id",     "4s"),
    ("nbytes", SINT4),
)

# ── WAVE  (364 bytes) ────────────────────────────────────────────────────────
WAVE_BLOCK = (
    ("id",              "4s"),
    ("nbytes",          SINT4),
    ("ps_file_name",    "16s"),
    ("num_chips",       "12s"),   # si16[6]
    ("blank_chip",      "256s"),
    ("repeat_seq",      REAL4),
    ("repeat_seq_dwel", SINT2),
    ("total_pcp",       SINT2),
    ("chip_offset",     "12s"),   # si16[6]
    ("chip_width",      "12s"),   # si16[6]
    ("ur_pcp",          REAL4),
    ("uv_pcp",          REAL4),
    ("num_gates",       "12s"),   # si16[6]
    ("gate_dist1",      "4s"),    # si16[2]
    ("gate_dist2",      "4s"),    # si16[2]
    ("gate_dist3",      "4s"),    # si16[2]
    ("gate_dist4",      "4s"),    # si16[2]
    ("gate_dist5",      "4s"),    # si16[2]
)


# ── Helper: build a struct format string from a block definition ──────────────
def _block_fmt(block, big_endian=True):
    endian = ">" if big_endian else "<"
    return endian + "".join(fmt for _, fmt in block)


def _block_size(block):
    return struct.calcsize(_block_fmt(block))


def unpack_block(block_def, data, big_endian=True):
    """Unpack raw bytes into an OrderedDict using a block definition tuple."""
    fmt = _block_fmt(block_def, big_endian)
    size = struct.calcsize(fmt)
    values = struct.unpack(fmt, data[:size])
    return dict(zip((name for name, _ in block_def), values))


# ── Block-ID → definition map ────────────────────────────────────────────────
BLOCK_MAP = {
    b"COMM": COMM_BLOCK,
    b"SSWB": SSWB_BLOCK,
    b"VOLD": VOLD_BLOCK,
    b"RADD": RADD_BLOCK,
    b"CFAC": CFAC_BLOCK,
    b"CELV": CELV_BLOCK,
    b"CSFD": CSFD_BLOCK,
    b"SWIB": SWIB_BLOCK,
    b"ASIB": ASIB_BLOCK,
    b"RYIB": RYIB_BLOCK,
    b"RDAT": RDAT_BLOCK,
    b"QDAT": QDAT_BLOCK,
    b"XSTF": XSTF_BLOCK,
    b"NULL": NULL_BLOCK,
    b"RKTB": RKTB_BLOCK,
    b"FRAD": FRAD_BLOCK,
    b"FRIB": FRIB_BLOCK,
    b"LIDR": LIDR_BLOCK,
    b"FLIB": FLIB_BLOCK,
    b"SITU": SITU_BLOCK,
    b"ISIT": ISIT_BLOCK,
    b"INDF": INDF_BLOCK,
    b"MINI": MINI_BLOCK,
    b"NDDS": NDDS_BLOCK,
    b"TIME": TIME_BLOCK,
    b"WAVE": WAVE_BLOCK,
}


def _is_dorade(path):
    """Return True if the file looks like a DORADE sweepfile.

    Detection is two-stage:
      1. Filename pattern — DORADE files are typically named 'swp.*' or end
         with '_SUR', '_PPI', '_RHI', '_COP' suffixes (no conventional
         extension).  We also accept any explicit .swp extension.
      2. Magic bytes — the first 4 bytes of a valid DORADE file are the ASCII
         block identifier of the first block, always one of the known DORADE
         block IDs (COMM, SSWB, VOLD, …).  We check for 'SSWB' or 'VOLD'
         which are almost always the first blocks.
    """
    basename = os.path.basename(path)
    # Filename heuristic: starts with 'swp.' or ends with '.swp'
    if basename.lower().startswith("swp."):
        return True
    try:
        with open(path, "rb") as fh:
            magic = fh.read(4)
        return magic in (b"SSWB", b"VOLD", b"COMM")
    except OSError:
        return False


def dorade_to_pyart_radar(dorade_file):
    """Convert a DoradeFile object into a pyart.core.Radar object.

    Only the fields present in the DORADE sweep are copied across.  All
    coordinate metadata (range, azimuth, elevation, fixed_angle, sweep
    indices) is derived from the DoradeFile's parsed rays and PARM blocks.

    Returns a pyart.core.Radar instance that is compatible with the rest of
    the GURT pipeline (RadarDisplay.plot_ppi, calculate_kdp, dealias, …).
    """
    import pyart.core

    # ── Collect rays and determine sweep geometry ─────────────────────────────
    rays = dorade_file.rays
    nrays = len(rays)
    if nrays == 0:
        raise ValueError("DORADE file contains no rays.")

    # ── Range axis ───────────────────────────────────────────────────────────
    # Use first available parameter to derive the range vector.
    first_param_name = next(iter(dorade_file.params))
    first_param = dorade_file.params[first_param_name]

    if "meters_to_first_cell" in first_param and first_param["meters_to_first_cell"] != 0:
        r0      = float(first_param["meters_to_first_cell"])
        spacing = float(first_param["meters_between_cells"])
        # Derive ngates from any ray that has this param
        ngates = max(
            len(r["rdat"][first_param_name])
            for r in rays if first_param_name in r["rdat"]
        )
    elif hasattr(dorade_file, "dist_cells") and dorade_file.dist_cells is not None:
        r0      = float(dorade_file.dist_cells[0])
        spacing = float(dorade_file.dist_cells[1] - dorade_file.dist_cells[0])
        ngates  = len(dorade_file.dist_cells)
    else:
        raise RuntimeError("Cannot determine range geometry from DORADE file.")

    range_data = r0 + np.arange(ngates, dtype="float32") * spacing

    # ── Azimuth / elevation arrays ────────────────────────────────────────────
    azimuths   = np.array([r["azimuth"]   for r in rays], dtype="float32")
    elevations = np.array([r["elevation"] for r in rays], dtype="float32")
    fixed_angle = float(getattr(dorade_file, "fixed_angle", np.nanmean(elevations)))

    # ── Time axis ─────────────────────────────────────────────────────────────
    start_dt = getattr(dorade_file, "sweep_start_time", None)
    stop_dt  = getattr(dorade_file, "sweep_stop_time",  None)

    if start_dt is None or stop_dt is None:
        start_dt = datetime.utcnow()
        stop_dt  = start_dt

    time_start_str = start_dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    total_seconds  = (stop_dt - start_dt).total_seconds()
    if total_seconds <= 0 or np.isnan(total_seconds):
        total_seconds = float(nrays)
    ray_times = np.linspace(0.0, total_seconds, nrays, dtype="float64")

    # ── Build pyart metadata dicts ────────────────────────────────────────────
    time_dict = {
        "data":     ray_times,
        "units":    f"seconds since {time_start_str}",
        "standard_name": "time",
        "long_name":     "time_in_seconds_since_volume_start",
        "calendar":      "gregorian",
    }
    range_dict = {
        "data":       range_data,
        "units":      "meters",
        "standard_name": "projection_range_coordinate",
        "long_name":     "range_to_measurement_volume",
        "axis":          "radial_range_coordinate",
        "spacing_is_constant": "true",
        "meters_to_center_of_first_gate": r0,
        "meters_between_gates":            spacing,
    }
    latitude_dict  = {"data": np.array([getattr(dorade_file, "lat", 0.0)],  dtype="float64"), "units": "degrees_north"}
    longitude_dict = {"data": np.array([getattr(dorade_file, "lon", 0.0)],  dtype="float64"), "units": "degrees_east"}
    altitude_dict  = {"data": np.array([getattr(dorade_file, "alt", 0.0)],  dtype="float64"), "units": "meters"}

    azimuth_dict   = {"data": azimuths,   "units": "degrees", "standard_name": "ray_azimuth_angle",   "long_name": "azimuth_angle_from_true_north"}
    elevation_dict = {"data": elevations, "units": "degrees", "standard_name": "ray_elevation_angle",  "long_name": "elevation_angle_from_horizontal_plane"}

    # Single sweep covering all rays
    sweep_start_ray  = np.array([0],           dtype="int32")
    sweep_end_ray    = np.array([nrays - 1],   dtype="int32")
    fixed_angle_dict = {"data": np.array([fixed_angle], dtype="float32"), "units": "degrees"}
    sweep_number_dict = {"data": np.array([getattr(dorade_file, "sweep_number", 0)], dtype="int32")}
    sweep_mode_str    = getattr(dorade_file, "sweep_mode", "azimuth_surveillance")
    sweep_mode_dict   = {"data": np.array([sweep_mode_str.ljust(32)[:32]], dtype="S32")}

    # ── Instrument parameters ─────────────────────────────────────────────────
    nyq = getattr(dorade_file, "nyquist_velocity", None)
    instrument_parameters = {}
    if nyq is not None and nyq != 0.0:
        instrument_parameters["nyquist_velocity"] = {
            "data":  np.full(nrays, float(nyq), dtype="float32"),
            "units": "meters_per_second",
        }
    freq_ghz = getattr(dorade_file, "frequency", None)
    # DORADE freq1 is stored in GHz (DoradeRadxFile.cc: freq1 * 1.0e9 for Hz)
    if freq_ghz is not None and freq_ghz != 0.0 and np.isfinite(float(freq_ghz)):
        instrument_parameters["frequency"] = {
            "data":  np.array([float(freq_ghz) * 1e9], dtype="float64"),
            "units": "Hz",
        }

    # ── Field data ────────────────────────────────────────────────────────────
    fields = {}
    for param_name, parm in dorade_file.params.items():
        scale  = parm["parameter_scale"]
        bias   = parm["parameter_bias"]
        bad    = parm["bad_data"]
        target_len = ngates  # normalise all rays to the same gate count

        rays_data = []
        for ray in rays:
            raw = ray["rdat"].get(param_name)
            if raw is None:
                rays_data.append(np.full(target_len, np.nan, dtype="float32"))
                continue
            # Convert back from raw int16 storage to physical units
            physical = np.where(raw == bad, np.nan, (raw - bias) / scale).astype("float32")
            if len(physical) < target_len:
                pad = np.full(target_len - len(physical), np.nan, dtype="float32")
                physical = np.concatenate([physical, pad])
            elif len(physical) > target_len:
                physical = physical[:target_len]
            rays_data.append(physical)

        data_arr = np.ma.masked_invalid(np.vstack(rays_data))
        units    = parm.get("param_units", b"")
        if isinstance(units, bytes):
            units = units.decode(errors="ignore").strip("\x00").strip()

        fields[param_name] = {
            "data":          data_arr,
            "units":         units,
            "long_name":     param_name,
            "standard_name": param_name,
            "_FillValue":    np.nan,
        }

    # ── Assemble radar metadata ───────────────────────────────────────────────
    metadata = {
        "instrument_name": getattr(dorade_file, "instrument_name", "Unknown"),
        "source":          "DORADE sweepfile",
        "original_container": "DORADE",
    }

    # ── Build the Radar object ────────────────────────────────────────────────
    radar = pyart.core.Radar(
        time            = time_dict,
        _range          = range_dict,
        fields          = fields,
        metadata        = metadata,
        scan_type       = sweep_mode_str,
        latitude        = latitude_dict,
        longitude       = longitude_dict,
        altitude        = altitude_dict,
        sweep_number    = sweep_number_dict,
        sweep_mode      = sweep_mode_dict,
        fixed_angle     = fixed_angle_dict,
        sweep_start_ray_index = {"data": sweep_start_ray},
        sweep_end_ray_index   = {"data": sweep_end_ray},
        azimuth         = azimuth_dict,
        elevation       = elevation_dict,
        instrument_parameters = instrument_parameters if instrument_parameters else None,
    )
    return radar

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
                   '.msg31', '.0', '.ar2v', '.RAW', '.HDF5', '.hdf5', 'swp.']
# DORADE sweepfiles typically have no extension and start with 'swp.'
# They are detected by _is_dorade() at load time; the extension list above
# covers the rare case where someone gives them a .swp extension.
_DORADE_NAME_PREFIXES = ("swp.",)

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

def detect_stagger_nyquist(radar, dorade_file=None):
    """Try to extract short/long Nyquist from stagger info.

    Accepts an optional ``dorade_file`` (DoradeFile instance) so that the
    RADD block fields (prt1/prt2, freq1, eff_unamb_vel) can be read directly,
    mirroring the Radx/DoradeRadxFile.cc reader exactly.

    Key unit facts from the DORADE spec (confirmed in DoradeRadxFile.cc):
      - freq1..freq5 are stored in GHz  → multiply by 1e9 for Hz
      - prt1..prt5  are stored in ms    → divide by 1000 for seconds
      - num_ipps_trans >= 2 is required for stagger mode (Radx check)
      - Negative prt values are used as "not set" sentinels by some writers

    Returns a dict with keys:
      'short_nyquist'    – float or None  (m/s)
      'long_nyquist'     – float or None  (m/s)
      'stagger_ratio'    – str like "2/3" or None
      'fixed_nyquist'    – float or None  (m/s, when no stagger)
      'extended_nyquist' – float or None  (m/s, m_stag * short_nyquist)
      'source'           – str describing where the values came from
    """
    result = {
        'short_nyquist':    None,
        'long_nyquist':     None,
        'stagger_ratio':    None,
        'fixed_nyquist':    None,
        'extended_nyquist': None,
        'source':           'none',
    }

    ip = getattr(radar, 'instrument_parameters', None) or {}

    # ── Step 1: Wavelength ────────────────────────────────────────────────────
    # DORADE freq1 is in GHz — DoradeRadxFile.cc line 2330:
    #   _readVol->addFrequencyHz(_ddRadar.freq1 * 1.0e9)
    # pyart instrument_parameters['frequency'] is already in Hz.
    wl_m = None
    if dorade_file is not None:
        try:
            freq_ghz = float(getattr(dorade_file, 'frequency', 0) or 0)
            # freq_ghz < 0.1 → probably 0 / unset; > 300 GHz → implausible
            if np.isfinite(freq_ghz) and 0.1 <= freq_ghz <= 300.0:
                wl_m = 3e8 / (freq_ghz * 1e9)   # convert GHz → Hz first
        except Exception:
            pass

    if wl_m is None:
        if 'frequency' in ip:
            try:
                freq_hz = float(np.ma.filled(ip['frequency']['data'], np.nan).flat[0])
                if np.isfinite(freq_hz) and freq_hz > 1e6:
                    wl_m = 3e8 / freq_hz
            except Exception:
                pass

    if wl_m is None:
        wl_m = 0.0533  # C-band fallback (~5.6 GHz)

    # ── Step 2: PRT values from DORADE RADD block ─────────────────────────────
    # DORADE prt1/prt2 are in milliseconds — DoradeRadxFile.cc line 1838:
    #   prtShort = _ddRadar.prt1 / 1000.0
    # Radx also checks for negative sentinels (some writers use -9999 for
    # "not set") and only uses prt2 for stagger when num_ipps_trans >= 2.
    prt_short_s  = None   # seconds
    prt_long_s   = None   # seconds
    num_ipps     = 1      # number of IPPs transmitted

    if dorade_file is not None:
        try:
            p1_ms = float(getattr(dorade_file, '_radd_prt1', 0) or 0)
            p2_ms = float(getattr(dorade_file, '_radd_prt2', 0) or 0)
            n_ipps = int(getattr(dorade_file, '_radd_num_ipps', 1) or 1)
            num_ipps = n_ipps

            # Mirror DoradeRadxFile.cc logic exactly:
            if p1_ms > 0 and p2_ms < 0:
                prt_short_s = p1_ms / 1000.0
            elif p2_ms > 0 and p1_ms < 0:
                prt_short_s = p2_ms / 1000.0
            elif p1_ms > 0 and p2_ms > 0:
                ps = p1_ms / 1000.0
                pl = p2_ms / 1000.0
                if pl < ps:           # swap so short < long
                    ps, pl = pl, ps
                prt_short_s = ps
                prt_long_s  = pl
            elif p1_ms > 0:
                prt_short_s = p1_ms / 1000.0

            # Radx only treats as stagger when num_ipps_trans >= 2
            if num_ipps < 2:
                prt_long_s = None

        except Exception:
            pass

    # ── Step 3: Fall back to pyart instrument_parameters ─────────────────────
    # pyart stores prt already in seconds (no unit conversion needed)
    if prt_short_s is None:
        prt_data = ip.get('prt', None)
        if prt_data is not None:
            try:
                arr = np.ma.filled(prt_data['data'], np.nan).ravel()
                prt_short_s = float(np.nanmedian(arr))
            except Exception:
                pass

        ratio_data = ip.get('prt_ratio', None)
        if prt_short_s is not None and ratio_data is not None:
            try:
                ratio_arr = np.ma.filled(ratio_data['data'], np.nan).ravel()
                r = float(np.nanmedian(ratio_arr))
                # prt_ratio in pyart = prt_short / prt_long  (< 1 for stagger)
                if np.isfinite(r) and 0.0 < r < 0.99:
                    prt_long_s = prt_short_s / r
            except Exception:
                pass

    # ── Step 4: No PRT → fall back to eff_unamb_vel directly ─────────────────
    if prt_short_s is None or not np.isfinite(prt_short_s) or prt_short_s <= 0:
        nyq_radd = None
        if dorade_file is not None:
            try:
                v = float(getattr(dorade_file, 'nyquist_velocity', 0) or 0)
                if np.isfinite(v) and v > 0:
                    nyq_radd = v
            except Exception:
                pass
        if nyq_radd is None:
            nv = ip.get('nyquist_velocity')
            if nv is not None:
                try:
                    nyq_radd = float(np.ma.filled(nv['data'], np.nan).flat[0])
                    if not (np.isfinite(nyq_radd) and nyq_radd > 0):
                        nyq_radd = None
                except Exception:
                    pass
        if nyq_radd is not None:
            result['fixed_nyquist'] = nyq_radd
            result['source'] = 'eff_unamb_vel'
        return result

    # ── Step 5: Fixed PRF ─────────────────────────────────────────────────────
    if prt_long_s is None or not np.isfinite(prt_long_s) or prt_long_s <= 0:
        nyq = wl_m / (4.0 * prt_short_s)
        result['fixed_nyquist'] = nyq if np.isfinite(nyq) else None
        result['source'] = f'prt_fixed ({prt_short_s*1000:.3f} ms, λ={wl_m*100:.2f} cm)'
        return result

    # ── Step 6: Staggered PRF ─────────────────────────────────────────────────
    # ratio = prt_short / prt_long  (< 1)
    ratio  = prt_short_s / prt_long_s
    frac   = Fraction(ratio).limit_denominator(10)
    m_stag = frac.numerator
    n_stag = frac.denominator

    nyq_short    = wl_m / (4.0 * prt_short_s)
    nyq_long     = wl_m / (4.0 * prt_long_s)
    nyq_extended = m_stag * nyq_short

    result['short_nyquist']    = nyq_short    if np.isfinite(nyq_short)    else None
    result['long_nyquist']     = nyq_long     if np.isfinite(nyq_long)     else None
    result['extended_nyquist'] = nyq_extended if np.isfinite(nyq_extended) else None
    result['stagger_ratio']    = f"{m_stag}/{n_stag}"
    result['source'] = (
        f'prt_stagger {m_stag}/{n_stag} '
        f'({prt_short_s*1000:.3f}/{prt_long_s*1000:.3f} ms, λ={wl_m*100:.2f} cm)'
    )
    return result


class ParamColorDialog(tk.Toplevel):
    BG        = "#b8b8db"
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
                                  bg="#ffffff", selectbackground="#C0C0C0",
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

        self._bg_var      = tk.StringVar(value="midnightblue")
        self._missing_var = tk.StringVar(value="darkslateblue")
        self._exceed_var  = tk.StringVar(value="gray70")
        self._annot_var   = tk.StringVar(value="gray90")
        self._emph_var    = tk.StringVar(value="hotpink")
        self._emin_var    = tk.StringVar(value="0.000")
        self._emax_var    = tk.StringVar(value="0.000")

        # ── colour-swatch buttons: keep a reference so we can update them ──
        self._color_btns = {}   # var_name -> swatch Label widget

        rows = [
            ("Parameter Name",    self._fname_var,   "entry"),
            ("Min",               self._vmin_var,    "entry"),
            ("Max",               self._vmax_var,    "entry"),
            ("Center",            self._ctr_var,     "entry"),
            ("Increment",         self._inc_var,     "entry"),
            ("Color Palette",     self._cmap_var,    "combo"),
            ("Label",             self._label_var,   "entry"),
            (None, None, None),
            ("Background Color",  self._bg_var,      "color"),
            ("Missing Data Color",self._missing_var, "color"),
            ("Exceeded Color",    self._exceed_var,  "color")
        ]

        for i, row in enumerate(rows):
            if row[0] is None:
                ttk.Separator(right, orient="horizontal").grid(
                    row=i, column=0, columnspan=2, sticky="ew", pady=3)
                continue
            lbl, var, kind = row
            tk.Label(right, text=lbl, bg=self.BG, anchor="e",
                     width=self.LABEL_W, font=("TkDefaultFont",9,"bold")
                     ).grid(row=i, column=0, sticky="e", padx=4, pady=2)

            if kind == "combo":
                cb = ttk.Combobox(right, textvariable=var,
                                  values=ALL_CMAP_NAMES, width=24,
                                  state="normal")
                cb.grid(row=i, column=1, sticky="w", padx=4, pady=2)
                cb.bind("<<ComboboxSelected>>", self._preview_cmap)

            elif kind == "color":
                self._make_color_row(right, i, var)

            else:  # "entry"
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

    # ------------------------------------------------------------------ #
    #  NEW: build one colour-picker row                                   #
    # ------------------------------------------------------------------ #
    def _make_color_row(self, parent, row_idx, var):
        """
        Place a coloured swatch + text button in column 1.
        Clicking either widget opens the system colour-chooser.
        The StringVar holds the chosen hex string (e.g. '#1a2b3c').
        """
        frame = tk.Frame(parent, bg=self.BG)
        frame.grid(row=row_idx, column=1, sticky="w", padx=4, pady=2)

        # resolve initial colour to a valid hex for the swatch
        def _to_hex(name):
            try:
                r, g, b = parent.winfo_rgb(name)
                return "#{:02x}{:02x}{:02x}".format(r >> 8, g >> 8, b >> 8)
            except Exception:
                return "#888888"

        swatch = tk.Label(frame, width=3, relief="sunken",
                          bg=_to_hex(var.get()))
        swatch.pack(side="left", padx=(0, 4))

        name_lbl = tk.Label(frame, textvariable=var, bg=self.BG,
                            anchor="w", width=18, cursor="hand2")
        name_lbl.pack(side="left")

        def _pick(v=var, sw=swatch, nl=name_lbl):
            initial = _to_hex(v.get())
            result = colorchooser.askcolor(color=initial, parent=self,
                                           title="Choose a color")
            if result and result[1]:          # result == ((r,g,b), '#rrggbb')
                hex_val = result[1]
                v.set(hex_val)
                sw.configure(bg=hex_val)

        swatch.bind("<Button-1>", lambda e, f=_pick: f())
        name_lbl.bind("<Button-1>", lambda e, f=_pick: f())

    # ------------------------------------------------------------------ #

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


_CB_HEIGHT = 0.025
_CB_BOTTOM = 0.01


# ══════════════════════════════════════════════════════════════════════════════
# DuplicateFieldDialog — lets the user copy one field to a new name
# ══════════════════════════════════════════════════════════════════════════════

class DuplicateFieldDialog(tk.Toplevel):
    """Modal dialog: choose a source field and type a destination name."""

    BG = "#191970"

    def __init__(self, parent, available_fields, on_apply):
        super().__init__(parent)
        self.on_apply = on_apply
        self.title("Duplicate Field")
        self.configure(bg=self.BG)
        self.resizable(False, False)
        self.transient(parent)
        self.grab_set()

        pad = dict(padx=10, pady=5)

        tk.Label(self, text="Source field:", bg=self.BG, fg="white",
                 anchor="w").grid(row=0, column=0, sticky="w", **pad)
        self._src_var = tk.StringVar()
        src_cb = ttk.Combobox(self, textvariable=self._src_var,
                              values=available_fields, width=18, state="readonly")
        if available_fields:
            src_cb.current(0)
        src_cb.grid(row=0, column=1, sticky="w", **pad)

        tk.Label(self, text="Output field name:", bg=self.BG, fg="white",
                 anchor="w").grid(row=1, column=0, sticky="w", **pad)
        self._dst_var = tk.StringVar()
        tk.Entry(self, textvariable=self._dst_var, width=20,
                 bg="#ffffff", relief="sunken").grid(row=1, column=1, sticky="w", **pad)

        btn_frame = tk.Frame(self, bg=self.BG)
        btn_frame.grid(row=2, column=0, columnspan=2, pady=8)
        tk.Button(btn_frame, text="Duplicate", command=self._ok,
                  bg=self.BG, relief="raised", padx=10).pack(side="left", padx=4)
        tk.Button(btn_frame, text="Cancel", command=self.destroy,
                  bg=self.BG, relief="raised", padx=10).pack(side="left", padx=4)

    def _ok(self):
        self.on_apply(self._src_var.get(), self._dst_var.get())
        self.destroy()


# ══════════════════════════════════════════════════════════════════════════════
# EditorPanel — floating toolbar for editing tools
# ══════════════════════════════════════════════════════════════════════════════

class EditorPanel(tk.Toplevel):
    """Floating panel that hosts editing tools (Unfold Brush, Deglitch Brush).

    It holds references back to the parent GURTApp so it can read/write
    the shared state variables (_editor_mode, _brush_radius_km, etc.).
    """

    BG  = "#e6e6e6"
    FG  = "white"
    ABG = "#2a2a5e"   # active / selected button background

    def __init__(self, app):
        super().__init__(app)
        self.app = app
        self.title("GURT Editor")
        self.configure(bg=self.BG)
        self.resizable(False, False)
        self.protocol("WM_DELETE_WINDOW", self._on_close)

        self._build()

        # Place near the top-right of the main window
        self.update_idletasks()
        mx = app.winfo_x() + app.winfo_width() - self.winfo_reqwidth() - 10
        my = app.winfo_y() + 40
        self.geometry(f"+{mx}+{my}")

    def _lbl(self, parent, text):
        return tk.Label(parent, text=text, bg=self.BG, fg=self.FG,
                        font=("TkDefaultFont", 9))

    def _build(self):
        app = self.app
        pad = dict(padx=8, pady=3)

        # ── Section header ────────────────────────────────────────────────────
        hdr = tk.Frame(self, bg="#0d0d3a")
        hdr.pack(fill="x")
        tk.Label(hdr, text="GURT Editor Panel", bg="#b8b8db", fg="black",
                 font=("TkDefaultFont", 10, "bold"), pady=5).pack()

        # ── Mode selector ─────────────────────────────────────────────────────
        mode_frame = tk.LabelFrame(self, text=" Mode ", bg=self.BG, fg=self.FG,
                                   font=("TkDefaultFont", 9, "bold"),
                                   relief="groove", bd=1)
        mode_frame.pack(fill="x", padx=8, pady=(8, 4))

        self._zoom_btn  = tk.Radiobutton(
            mode_frame, text="Zoom / Pan  (default)",
            variable=app._editor_mode, value="zoom",
            bg=self.BG, fg=self.FG, selectcolor=self.ABG,
            activebackground=self.ABG, activeforeground=self.FG,
            indicatoron=True, font=("TkDefaultFont", 9),
            command=self._on_mode_change)
        self._zoom_btn.pack(anchor="w", **pad)

        self._brush_btn = tk.Radiobutton(
            mode_frame, text="Unfold Brush",
            variable=app._editor_mode, value="unfold_brush",
            bg=self.BG, fg=self.FG, selectcolor=self.ABG,
            activebackground=self.ABG, activeforeground=self.FG,
            indicatoron=True, font=("TkDefaultFont", 9),
            command=self._on_mode_change)
        self._brush_btn.pack(anchor="w", **pad)

        self._deglitch_btn = tk.Radiobutton(
            mode_frame, text="Deglitch Brush",
            variable=app._editor_mode, value="deglitch_brush",
            bg=self.BG, fg=self.FG, selectcolor=self.ABG,
            activebackground=self.ABG, activeforeground=self.FG,
            indicatoron=True, font=("TkDefaultFont", 9),
            command=self._on_mode_change)
        self._deglitch_btn.pack(anchor="w", **pad)

        self._eraser_btn = tk.Radiobutton(
            mode_frame, text="Eraser Brush",
            variable=app._editor_mode, value="eraser_brush",
            bg=self.BG, fg=self.FG, selectcolor=self.ABG,
            activebackground=self.ABG, activeforeground=self.FG,
            indicatoron=True, font=("TkDefaultFont", 9),
            command=self._on_mode_change)
        self._eraser_btn.pack(anchor="w", **pad)

        self._boundary_btn = tk.Radiobutton(
            mode_frame, text="Boundary (polygon)",
            variable=app._editor_mode, value="boundary",
            bg=self.BG, fg=self.FG, selectcolor=self.ABG,
            activebackground=self.ABG, activeforeground=self.FG,
            indicatoron=True, font=("TkDefaultFont", 9),
            command=self._on_mode_change)
        self._boundary_btn.pack(anchor="w", **pad)

        # ── Brush radius (shared by both brushes) ─────────────────────────────
        self._brush_frame = tk.LabelFrame(
            self, text=" Brush Settings ", bg=self.BG, fg=self.FG,
            font=("TkDefaultFont", 9, "bold"), relief="groove", bd=1)
        self._brush_frame.pack(fill="x", padx=8, pady=4)

        r0 = tk.Frame(self._brush_frame, bg=self.BG)
        r0.pack(fill="x", **pad)
        self._lbl(r0, "Brush radius (km):").pack(side="left")
        tk.Entry(r0, textvariable=app._brush_radius_km, width=7,
                 bg="#ffffff", fg="#000", relief="sunken").pack(side="left", padx=4)

        # ── Unfold brush settings ─────────────────────────────────────────────
        self._unfold_frame = tk.LabelFrame(
            self, text=" Unfold Brush Settings ", bg=self.BG, fg=self.FG,
            font=("TkDefaultFont", 9, "bold"), relief="groove", bd=1)
        self._unfold_frame.pack(fill="x", padx=8, pady=4)

        r1 = tk.Frame(self._unfold_frame, bg=self.BG)
        r1.pack(fill="x", **pad)
        self._lbl(r1, "Nyquist velocity (m/s):").pack(side="left")
        tk.Entry(r1, textvariable=app._brush_nyquist, width=7,
                 bg="#ffffff", fg="#000", relief="sunken").pack(side="left", padx=4)
        self._lbl(r1, "(0 = auto)").pack(side="left")

        r2 = tk.Frame(self._unfold_frame, bg=self.BG)
        r2.pack(fill="x", **pad)
        self._lbl(r2, "Fold Center (m/s):").pack(side="left")
        tk.Entry(r2, textvariable=app._brush_center, width=7,
                 bg="#ffffff", fg="#000", relief="sunken").pack(side="left", padx=4)
        self._lbl(r2, "(target velocity)").pack(side="left")

        # ── Deglitch brush settings ───────────────────────────────────────────
        self._deglitch_frame = tk.LabelFrame(
            self, text=" Deglitch Brush Settings ", bg=self.BG, fg=self.FG,
            font=("TkDefaultFont", 9, "bold"), relief="groove", bd=1)
        self._deglitch_frame.pack(fill="x", padx=8, pady=4)

        # Stagger info readout
        stag_row = tk.Frame(self._deglitch_frame, bg=self.BG)
        stag_row.pack(fill="x", padx=8, pady=(4, 2))
        self._stagger_info_var = tk.StringVar(value="No file loaded")
        tk.Label(stag_row, textvariable=self._stagger_info_var,
                 bg=self.BG, fg="#aaddff",
                 font=("TkDefaultFont", 8), wraplength=250, justify="left"
                 ).pack(anchor="w")

        # Unfold-by selector
        mode_row = tk.Frame(self._deglitch_frame, bg=self.BG)
        mode_row.pack(fill="x", padx=8, pady=2)
        self._lbl(mode_row, "Unfold by:").pack(side="left")
        for lbl, val in [("Short Nyq (default)", "short"), ("Long Nyq", "long"), ("Manual", "manual")]:
            tk.Radiobutton(mode_row, text=lbl, variable=app._deglitch_nyquist_mode,
                           value=val, bg=self.BG, fg=self.FG,
                           selectcolor=self.ABG, activebackground=self.ABG,
                           activeforeground=self.FG, indicatoron=True,
                           font=("TkDefaultFont", 8),
                           command=self._refresh_stagger_display
                           ).pack(side="left", padx=4)

        # Manual Nyquist entry
        man_row = tk.Frame(self._deglitch_frame, bg=self.BG)
        man_row.pack(fill="x", padx=8, pady=2)
        self._lbl(man_row, "Manual Nyquist (m/s):").pack(side="left")
        self._manual_nyq_entry = tk.Entry(
            man_row, textvariable=app._deglitch_manual_nyq,
            width=7, bg="#ffffff", fg="#000", relief="sunken")
        self._manual_nyq_entry.pack(side="left", padx=4)

        # Tolerance
        tol_row = tk.Frame(self._deglitch_frame, bg=self.BG)
        tol_row.pack(fill="x", padx=8, pady=2)
        self._lbl(tol_row, "Tolerance (m/s):").pack(side="left")
        tk.Entry(tol_row, textvariable=app._deglitch_tolerance,
                 width=7, bg="#ffffff", fg="#000", relief="sunken"
                 ).pack(side="left", padx=4)
        self._lbl(tol_row, "(gates > this get deglitched)").pack(side="left")

        # ── Boundary action chooser (shown only in boundary mode) ─────────────
        self._boundary_action_frame = tk.LabelFrame(
            self, text=" Boundary Action ", bg=self.BG, fg=self.FG,
            font=("TkDefaultFont", 9, "bold"), relief="groove", bd=1)
        # (packed/unpacked dynamically by _on_mode_change)

        ba_row = tk.Frame(self._boundary_action_frame, bg=self.BG)
        ba_row.pack(fill="x", padx=6, pady=4)
        for lbl, val in [("Erase gates", "erase"),
                         ("Unfold",      "unfold"),
                         ("Deglitch",    "deglitch")]:
            tk.Radiobutton(ba_row, text=lbl,
                           variable=app._boundary_action, value=val,
                           bg=self.BG, fg=self.FG, selectcolor=self.ABG,
                           activebackground=self.ABG, activeforeground=self.FG,
                           indicatoron=True, font=("TkDefaultFont", 9)
                           ).pack(side="left", padx=6)

        tk.Button(self._boundary_action_frame,
                  text="✔  Apply Edit  (Enter)",
                  command=app._boundary_apply,
                  bg="#224422", fg="white", relief="raised",
                  activebackground="#336633", activeforeground="white",
                  padx=8).pack(fill="x", padx=6, pady=(2, 4))
        tk.Button(self._boundary_action_frame,
                  text="✖  Cancel / Clear  (Esc)",
                  command=app._boundary_clear,
                  bg="#442222", fg="white", relief="raised",
                  activebackground="#663333", activeforeground="white",
                  padx=8).pack(fill="x", padx=6, pady=(0, 6))

        # ── Tip label ─────────────────────────────────────────────────────────
        self._tip_var = tk.StringVar(value="Mode: Zoom")
        tk.Label(self, textvariable=self._tip_var, bg=self.BG, fg="#aaaadd",
                 font=("TkDefaultFont", 8), wraplength=260, justify="left"
                 ).pack(fill="x", padx=8, pady=2)

        # ── Undo ─────────────────────────────────────────────────────────────────
        tk.Button(self, text="Undo Last Edit  (Ctrl+Z)",
                  command=app._undo_brush,
                  bg=self.BG, fg=self.FG, relief="raised",
                  activebackground=self.ABG, activeforeground=self.FG,
                  padx=8).pack(fill="x", padx=8, pady=2)

        # ── Duplicate Field button ────────────────────────────────────────────
        sep = tk.Frame(self, bg="#333366", height=1)
        sep.pack(fill="x", padx=8, pady=6)

        tk.Button(self, text="Duplicate Field…",
                  command=app._duplicate_field_dialog,
                  bg=self.BG, fg=self.FG, relief="raised",
                  activebackground=self.ABG, activeforeground=self.FG,
                  padx=8).pack(fill="x", padx=8, pady=2)

        # ── Close ─────────────────────────────────────────────────────────────
        tk.Button(self, text="Close",
                  command=self._on_close,
                  bg=self.BG, fg=self.FG, relief="raised",
                  activebackground="#550000", activeforeground=self.FG,
                  padx=8).pack(fill="x", padx=8, pady=(2, 8))

        self._refresh_stagger_display()
        self._on_mode_change()   # sync tip label on open

    def _refresh_stagger_display(self):
        """Update the stagger info label in the deglitch frame."""
        app = self.app
        ratio = app._stagger_ratio_str.get()
        short_nyq = app._stagger_short_nyq.get()
        long_nyq  = app._stagger_long_nyq.get()
        fixed_nyq = app._stagger_fixed_nyq.get()
        mode = app._deglitch_nyquist_mode.get()

        if ratio:
            info = (f"Stagger: {ratio}  |  "
                    f"Short: {short_nyq:.2f} m/s  |  Long: {long_nyq:.2f} m/s")
        elif fixed_nyq > 0:
            info = f"Fixed PRF — Nyquist: {fixed_nyq:.2f} m/s"
        else:
            info = "No Nyquist info detected — use Manual."

        # Show which value will actually be used
        if mode == "manual":
            man = app._deglitch_manual_nyq.get()
            info += f"\n→ Using manual: {man:.2f} m/s"
        elif mode == "short" and short_nyq > 0:
            info += f"\n→ Using short: {short_nyq:.2f} m/s"
        elif mode == "long" and long_nyq > 0:
            info += f"\n→ Using long: {long_nyq:.2f} m/s"
        elif fixed_nyq > 0:
            info += f"\n→ Using fixed: {fixed_nyq:.2f} m/s"
        else:
            info += "\n→ No Nyquist — set manually!"

        self._stagger_info_var.set(info)

    def _on_mode_change(self):
        mode = self.app._editor_mode.get()
        if mode == "unfold_brush":
            self._tip_var.set(
                "Unfold Brush active.\n"
                "Left-click / drag to fold gates toward the centre velocity.\n"
                "Set Nyquist & fold centre above.\n"
                "Press Escape or switch to Zoom to exit.")
            self.app._drag_start = None
            if self.app._drag_rect is not None:
                try:
                    self.app._drag_rect.remove()
                except Exception:
                    pass
                self.app._drag_rect = None
                self.app.canvas.draw_idle()
            self.app._boundary_clear()
            self._boundary_action_frame.pack_forget()
        elif mode == "deglitch_brush":
            self._tip_var.set(
                "Deglitch Brush active.\n"
                "Left-click / drag over glitchy gates.\n"
                "Gates beyond tolerance are folded toward the local median\n"
                "in steps of 2 × short Nyquist.\n"
                "Press Escape or switch to Zoom to exit.")
            self.app._drag_start = None
            if self.app._drag_rect is not None:
                try:
                    self.app._drag_rect.remove()
                except Exception:
                    pass
                self.app._drag_rect = None
                self.app.canvas.draw_idle()
            self._refresh_stagger_display()
            self.app._boundary_clear()
            self._boundary_action_frame.pack_forget()
        elif mode == "eraser_brush":
            self._tip_var.set(
                "Eraser Brush active.\n"
                "Left-click / drag to mask (remove) gates.\n"
                "Gates are set to missing/NaN.\n"
                "Press Escape or switch to Zoom to exit.")
            self.app._drag_start = None
            if self.app._drag_rect is not None:
                try:
                    self.app._drag_rect.remove()
                except Exception:
                    pass
                self.app._drag_rect = None
                self.app.canvas.draw_idle()
            self.app._boundary_clear()
            self._boundary_action_frame.pack_forget()
        elif mode == "boundary":
            self._tip_var.set(
                "Boundary mode active.\n"
                "Left-click to place polygon vertices.\n"
                "Double-click or press Enter to close & apply.\n"
                "Escape cancels. Choose action in the panel below.")
            self.app._drag_start = None
            if self.app._drag_rect is not None:
                try:
                    self.app._drag_rect.remove()
                except Exception:
                    pass
                self.app._drag_rect = None
                self.app.canvas.draw_idle()
            self.app._boundary_clear()
            # Show the boundary action chooser sub-frame
            self._boundary_action_frame.pack(fill="x", padx=8, pady=4)
        else:
            self._tip_var.set("Mode: Zoom / rubber-band pan.")
            self.app._remove_brush_overlay()
            self.app._boundary_clear()
            self._boundary_action_frame.pack_forget()

    def _on_close(self):
        # Return to zoom mode when the panel is closed
        self.app._editor_mode.set("zoom")
        self.app._remove_brush_overlay()
        self.app._boundary_clear()
        self.app._editor_panel_win = None
        self.destroy()


# ══════════════════════════════════════════════════════════════════════════════
# CfRadialExportDialog — directory picker + scope selector
# ══════════════════════════════════════════════════════════════════════════════

class CfRadialExportDialog(tk.Toplevel):
    """Modal dialog for CfRadial export.

    Lets the user:
      • Set (or remember) the output directory
      • Choose scope: current sweep / current volume / all loaded files
    """
    BG      = "#191970"
    FG      = "white"
    ENTRY_BG = "#ffffff"

    def __init__(self, app, initial_scope="sweep"):
        super().__init__(app)
        self.app   = app
        self.title("Export to CfRadial")
        self.configure(bg=self.BG)
        self.resizable(False, False)
        self.grab_set()
        self.transient(app)

        pad = dict(padx=10, pady=4)

        # ── Header ───────────────────────────────────────────────────────────
        hdr = tk.Frame(self, bg="#0d0d3a")
        hdr.pack(fill="x")
        tk.Label(hdr, text="Export to CfRadial (.nc)",
                 bg="#0d0d3a", fg="white",
                 font=("TkDefaultFont", 11, "bold"), pady=7).pack()

        body = tk.Frame(self, bg=self.BG)
        body.pack(fill="both", expand=True, padx=12, pady=8)

        # ── Output directory ─────────────────────────────────────────────────
        tk.Label(body, text="Output directory:", bg=self.BG, fg=self.FG,
                 anchor="w", font=("TkDefaultFont", 9, "bold")
                 ).grid(row=0, column=0, sticky="w", **pad)

        dir_frame = tk.Frame(body, bg=self.BG)
        dir_frame.grid(row=1, column=0, sticky="ew", padx=10, pady=2)
        body.columnconfigure(0, weight=1)

        self._dir_var = tk.StringVar(value=getattr(app, "_cfrad_export_dir", ""))
        self._dir_entry = tk.Entry(dir_frame, textvariable=self._dir_var,
                                   width=46, bg=self.ENTRY_BG, relief="sunken")
        self._dir_entry.pack(side="left", fill="x", expand=True)
        tk.Button(dir_frame, text="Browse…", command=self._browse,
                  bg=self.BG, fg=self.FG, relief="raised",
                  activebackground="#2a2a5e", activeforeground=self.FG,
                  padx=6).pack(side="left", padx=(6, 0))

        # ── Scope selector ───────────────────────────────────────────────────
        scope_frame = tk.LabelFrame(body, text=" Export scope ",
                                    bg=self.BG, fg=self.FG,
                                    font=("TkDefaultFont", 9, "bold"),
                                    relief="groove", bd=1)
        scope_frame.grid(row=2, column=0, sticky="ew", padx=10, pady=(10, 4))

        self._scope_var = tk.StringVar(value=initial_scope)

        sweep_idx = app.sweep_index.get() if app.radar else 0
        fixed_ang = ""
        if app.radar:
            try:
                fa = app.radar.fixed_angle["data"][sweep_idx]
                fixed_ang = f"  ({fa:.1f}°)"
            except Exception:
                pass

        n_files = len(app.file_list)
        options = [
            ("sweep",  f"Current sweep only{fixed_ang}"),
            ("volume", "Current volume (all sweeps in loaded file)"),
            ("all",    f"All loaded files  ({n_files} file{'s' if n_files != 1 else ''})"),
        ]
        for val, lbl in options:
            tk.Radiobutton(scope_frame, text=lbl,
                           variable=self._scope_var, value=val,
                           bg=self.BG, fg=self.FG,
                           selectcolor="#2a2a5e",
                           activebackground="#2a2a5e", activeforeground=self.FG,
                           indicatoron=True,
                           font=("TkDefaultFont", 9)
                           ).pack(anchor="w", padx=8, pady=2)

        # ── Info label ───────────────────────────────────────────────────────
        self._info_var = tk.StringVar(value="")
        tk.Label(body, textvariable=self._info_var,
                 bg=self.BG, fg="#aaddff",
                 font=("TkDefaultFont", 8), wraplength=400, justify="left"
                 ).grid(row=3, column=0, sticky="w", padx=10, pady=(2, 0))
        self._scope_var.trace_add("write", self._update_info)
        self._update_info()

        # ── Buttons ──────────────────────────────────────────────────────────
        btn_row = tk.Frame(body, bg=self.BG)
        btn_row.grid(row=4, column=0, pady=(12, 4))
        tk.Button(btn_row, text="Export", command=self._ok,
                  bg=self.BG, fg=self.FG, relief="raised",
                  activebackground="#2a2a5e", activeforeground=self.FG,
                  font=("TkDefaultFont", 9, "bold"), padx=18, pady=4
                  ).pack(side="left", padx=6)
        tk.Button(btn_row, text="Cancel", command=self.destroy,
                  bg=self.BG, fg=self.FG, relief="raised",
                  activebackground="#2a2a5e", activeforeground=self.FG,
                  padx=14, pady=4
                  ).pack(side="left", padx=6)

        self.update_idletasks()
        # Center on parent
        pw = app.winfo_width();  ph = app.winfo_height()
        px = app.winfo_rootx(); py = app.winfo_rooty()
        dw = self.winfo_reqwidth(); dh = self.winfo_reqheight()
        self.geometry(f"+{px + (pw - dw)//2}+{py + (ph - dh)//2}")

    def _browse(self):
        d = filedialog.askdirectory(title="CfRadial output directory")
        if d:
            self._dir_var.set(d)

    def _update_info(self, *_):
        scope = self._scope_var.get()
        out_dir = self._dir_var.get().strip() or "(not set)"
        if scope == "sweep":
            self._info_var.set(f"One .nc file will be written to:\n{out_dir}")
        elif scope == "volume":
            self._info_var.set(f"One .nc file (all sweeps) will be written to:\n{out_dir}")
        else:
            n = len(self.app.file_list)
            self._info_var.set(
                f"{n} .nc file(s) will be written to:\n{out_dir}\n"
                "(existing files with the same name will be overwritten)")

    def _ok(self):
        out_dir = self._dir_var.get().strip()
        if not out_dir:
            messagebox.showwarning("Export CfRadial", "Please set an output directory.",
                                   parent=self)
            return
        # Remember for next time
        self.app._cfrad_export_dir = out_dir
        scope = self._scope_var.get()
        self.destroy()
        self.app._do_cfradial_export(out_dir, scope)



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
        self.ring_interval  = tk.DoubleVar(value=10.0)
        self.az_interval    = tk.DoubleVar(value=10.0)
        self.tick_interval  = tk.DoubleVar(value=10.0)
        self.max_range      = tk.DoubleVar(value=50.0)
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
        self._cfrad_export_dir  = ""   # remembered CfRadial output directory

        # ── Editor / brush state ──────────────────────────────────────────────
        self._editor_mode       = tk.StringVar(value="zoom")  # "zoom" | "unfold_brush" | "deglitch_brush" | "eraser_brush" | "boundary"
        self._brush_radius_km   = tk.DoubleVar(value=2.0)
        self._brush_nyquist     = tk.DoubleVar(value=0.0)   # 0 = auto from radar
        self._brush_center      = tk.DoubleVar(value=0.0)   # folding centre velocity
        self._editor_panel_win  = None   # reference to the floating EditorPanel window
        self._brush_overlay     = None   # Circle artist drawn while brushing
        self._brush_active      = False  # True while LMB is held in brush mode

        # ── Boundary (polygon lasso) tool state ──────────────────────────────
        self._boundary_poly_pts  = []    # list of (x_km, y_km) clicked so far
        self._boundary_artists   = []    # Line2D artists drawn on the axes
        self._boundary_ax        = None  # axes the boundary is being drawn on
        self._boundary_action    = tk.StringVar(value="erase")  # "erase" | "unfold" | "deglitch"
        self._boundary_win       = None  # floating boundary action chooser window

        # ── Per-field colormap offset/scale adjustments ───────────────────────
        # {field_name: [offset, scale]}  — offset shifts midpoint, scale expands range
        self._cmap_adjustments   = {}    # field_name -> [offset, scale]
        self._cmap_slider_win    = None  # floating slider window

        # ── Stagger / Nyquist state (populated on file load) ──────────────────
        self._stagger_short_nyq  = tk.DoubleVar(value=0.0)  # 0 = unknown
        self._stagger_long_nyq   = tk.DoubleVar(value=0.0)
        self._stagger_fixed_nyq  = tk.DoubleVar(value=0.0)
        self._stagger_ratio_str  = tk.StringVar(value="")
        # Deglitch settings
        self._deglitch_nyquist_mode = tk.StringVar(value="short")  # "short" | "long" | "manual"
        self._deglitch_manual_nyq   = tk.DoubleVar(value=0.0)
        self._deglitch_tolerance    = tk.DoubleVar(value=3.0)       # m/s

        # ── Undo stack ───────────────────────────────────────────────────────────
        # Each entry is a dict mapping field_name -> deep copy of field data array
        # captured just before a brush stroke is applied.  Max 50 entries (FIFO).
        self._undo_stack        = []   # list of {field_name: np.ndarray}
        self._undo_stack_max    = 50

        # ── Rotate-azimuth offset (degrees, persists across reloads) ─────────────
        self._az_rotation_deg   = 0.0
        # ── Elevation offset (degrees, persists across reloads) ──────────────────
        self._el_offset_deg     = 0.0

        # ── Render cache ──────────────────────────────────────────────────────────
        # Maps (panel_idx, field_name) -> QuadMesh artist from plot_ppi.
        # Used by _fast_refresh_field to update color data without a full redraw.
        self._ppi_meshes        = {}

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
        cfr_sub = self._mk_menu(fm)
        cfr_sub.add_command(label="Current sweep…",
                            command=lambda: self.export_cfradial("sweep"))
        cfr_sub.add_command(label="Current volume…",
                            command=lambda: self.export_cfradial("volume"))
        cfr_sub.add_command(label="All loaded files…",
                            command=lambda: self.export_cfradial("all"))
        cfr_sub.add_separator()
        cfr_sub.add_command(label="Set export directory…",
                            command=self._set_cfrad_export_dir)
        fm.add_cascade(label="Export to CfRadial", menu=cfr_sub)
        fm.add_separator()
        fm.add_command(label="Exit",             command=self.quit)
        mb.add_cascade(label="File", menu=fm)

        zm = self._mk_menu(mb)
        zm.add_command(label="Reset Zoom (all panels)", command=self._reset_all_zoom)
        zm.add_separator()
        for lbl, factor in [("Data Extent", "data"),("Default 150 km", 150),
                             #("+50%",1.5),("+25%",1.25),("+10%",1.1),
                             #("-10%",0.9),("-25%",0.75),("-50%",0.5)]:
                             ("+50%",0.5),("+25%",0.75),("+10%",0.9),
                             ("-10%",1.1),("-25%",1.25),("-50%",1.5)]:
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

        em = self._mk_menu(mb)
        em.add_command(label="Undo  (Ctrl+Z)",     command=self._undo_brush)
        em.add_separator()
        em.add_command(label="Open Editor Panel…", command=self._open_editor_panel)
        em.add_separator()
        em.add_command(label="Duplicate Field…",   command=self._duplicate_field_dialog)
        em.add_separator()
        em.add_command(label="Rotate Azimuths…",        command=self._rotate_azimuths_dialog)
        em.add_command(label="Offset Elevations…",      command=self._offset_elevations_dialog)
        em.add_command(label="Set Geolocation…",        command=self._set_geolocation_dialog)
        em.add_command(label="Edit Range / Gate Spacing…", command=self._edit_range_dialog)
        mb.add_cascade(label="Edit", menu=em)

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
        self.bind("<Control-z>", lambda e: self._undo_brush())
        self.bind("<Control-Z>", lambda e: self._undo_brush())
        self.bind("<Left>",  lambda e: self._change_file(-1))
        self.bind("<Right>", lambda e: self._change_file(+1))
        self.bind("<Up>",    lambda e: self._change_sweep(+1))
        self.bind("<Down>",  lambda e: self._change_sweep(-1))
        self.bind("<Return>", lambda e: self._boundary_apply()
                  if self._editor_mode.get() == "boundary" else None)
        self.bind("<Escape>", self._on_escape_key)
        self.after(100, self.focus_set)

    def _on_escape_key(self, event=None):
        if self._editor_mode.get() == "boundary" and self._boundary_poly_pts:
            self._boundary_clear()   # first Escape clears polygon
        else:
            self._reset_all_zoom()   # second Escape / zoom mode resets zoom

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
            filetypes=[("Radar files","*.nc *.h5 *.buf *.raw *.gz *.HDF5 *.hdf5 *.RAW swp.*"),
                       ("DORADE sweepfiles", "swp.*"),
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
                is_known_ext = any(f.lower().endswith(e) or f.endswith(e) for e in FILE_EXTENSIONS)
                is_dorade_name = any(f.lower().startswith(p) for p in _DORADE_NAME_PREFIXES)
                if is_known_ext or is_dorade_name:
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
            _current_dorade_file = None
            # Clear undo history when a new file is loaded
            self._undo_stack.clear()
            self._az_rotation_deg = 0.0
            self._boundary_clear()
            if _is_dorade(path):
                if not DORADE_AVAILABLE:
                    messagebox.showerror(
                        "Dorade reader missing",
                        "dorade_reader_expanded.py not found.\n"
                        "Place it in the same folder as gurt_guiv3.py.")
                    self.status_var.set("Error: Dorade reader missing.")
                    return
                _current_dorade_file = DoradeFile(path)
                self.radar = dorade_to_pyart_radar(_current_dorade_file)
            else:
                self.radar = pyart.io.read(path, linear_interp=False)
            if self.do_kdp.get():
                self.radar = calculate_kdp(self.radar)
            if self.do_dealias.get():
                self.radar = dealias_velocity(self.radar)

            # ── Detect stagger / Nyquist velocities ──────────────────────────
            stagger = detect_stagger_nyquist(self.radar, dorade_file=_current_dorade_file)
            self._stagger_short_nyq.set(stagger['short_nyquist'] or 0.0)
            self._stagger_long_nyq.set(stagger['long_nyquist']   or 0.0)
            self._stagger_fixed_nyq.set(stagger['fixed_nyquist'] or 0.0)
            self._stagger_ratio_str.set(stagger['stagger_ratio'] or "")
            # If EditorPanel is open, refresh its Nyquist readout
            if self._editor_panel_win is not None and self._editor_panel_win.winfo_exists():
                try:
                    self._editor_panel_win._refresh_stagger_display()
                except Exception:
                    pass
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
        self._ppi_meshes = {}   # invalidate mesh cache on full redraw
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

            # Apply per-field colormap offset/scale adjustments
            adj = self._cmap_adjustments.get(fname)
            if adj:
                offset, scale = adj
                mid   = (vmin + vmax) / 2.0 + offset
                half  = (vmax - vmin) / 2.0 * scale
                vmin  = mid - half
                vmax  = mid + half

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

                # Cache the QuadMesh artist for fast in-place updates (brush tool)
                if ax.collections:
                    self._ppi_meshes[(idx, fname)] = ax.collections[-1]

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
            # Apply per-field colormap offset/scale for colorbar too
            adj_i = self._cmap_adjustments.get(fname_i)
            if adj_i:
                offset_i, scale_i = adj_i
                mid_i   = (vmin_i + vmax_i) / 2.0 + offset_i
                half_i  = (vmax_i - vmin_i) / 2.0 * scale_i
                vmin_i  = mid_i - half_i
                vmax_i  = mid_i + half_i
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
        """Position each panel to fill its cell rectangle completely."""
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
                # ── Derive the view centre/half from the best available source ─
                # Priority: panel_limits (set by drag-zoom) > current ax limits
                # (set by a previous _reposition_axes call) > panel_meta (set at
                # refresh_plot time, i.e. the original unzoomed view).
                # We deliberately do NOT fall back to panel_meta alone so that a
                # zoom followed by a window resize keeps the zoomed view.
                lims = self.panel_limits[idx] if idx < len(self.panel_limits) else None
                if lims:
                    xl0, xl1, yl0, yl1 = lims
                    cx   = (xl0 + xl1) / 2.0
                    cy   = (yl0 + yl1) / 2.0
                    half = max(xl1 - xl0, yl1 - yl0) / 2.0
                else:
                    # No zoom drag yet — use panel_meta (original full view)
                    cx, cy, half = self._panel_meta[idx]

                # Always keep panel_meta in sync so other callers stay correct
                self._panel_meta[idx] = (cx, cy, half)

                aspect = cell_w / cell_h
                if aspect >= 1.0:
                    x_half = half * aspect
                    y_half = half
                else:
                    x_half = half
                    y_half = half / aspect
                ax.set_xlim(cx - x_half, cx + x_half)
                ax.set_ylim(cy - y_half, cy + y_half)

                # ── Redraw ticks at the current zoom level ────────────────────
                from matplotlib.collections import LineCollection as _LC
                for col_artist in ax.collections[:]:
                    if col_artist.get_label() == '_gurt_ticks':
                        col_artist.remove()
                # Remove old edge-distance labels
                for txt in ax.texts[:]:
                    if getattr(txt, '_gurt_tick_label', False):
                        txt.remove()
                if self.show_ticks.get():
                    ti = max(self.tick_interval.get(), 0.1)
                    xlim2 = ax.get_xlim()
                    ylim2 = ax.get_ylim()
                    x_span = xlim2[1] - xlim2[0]
                    y_span = ylim2[1] - ylim2[0]
                    span2  = max(x_span, y_span)
                    MAX_TICKS = 20
                    while span2 / ti > MAX_TICKS:
                        ti *= 2

                    # Tick cross arm length: fixed at ~5 pixels regardless of
                    # window size or zoom level.  We convert pixels → data units
                    # using the current cell dimensions and data span.
                    arm_px   = 5.0
                    tick_s2  = arm_px * (x_span / cell_w) if cell_w > 0 else x_span * 0.01

                    xs2 = np.arange(np.ceil(xlim2[0] / ti) * ti, xlim2[1] + ti, ti)
                    ys2 = np.arange(np.ceil(ylim2[0] / ti) * ti, ylim2[1] + ti, ti)
                    segs2 = []
                    for x in xs2:
                        for y in ys2:
                            segs2.append([[x - tick_s2, y], [x + tick_s2, y]])
                            segs2.append([[x, y - tick_s2], [x, y + tick_s2]])
                    if segs2:
                        lc2 = _LC(segs2, colors='white', alpha=0.8, linewidths=0.7)
                        lc2.set_label('_gurt_ticks')
                        ax.add_collection(lc2)

                    # ── Edge distance labels ──────────────────────────────────
                    x0v, x1v = xlim2
                    y0v, y1v = ylim2
                    x_inset = (x1v - x0v) * 0.015
                    lbl_kw = dict(color='white', fontsize=6, fontweight='bold',
                                  clip_on=True,
                                  bbox=dict(facecolor='#191970', alpha=0.55,
                                            edgecolor='none', pad=0.8))

                    def _fmt(v):
                        """Format km value: drop decimals when integer."""
                        return f'{v:.0f}' if v == int(v) else f'{v:.1f}'

                    # Minimum pixel gap between x-edge labels to avoid overlap.
                    min_px_gap = 28
                    px_per_km  = cell_w / x_span if x_span > 0 else 1.0
                    tick_px    = ti * px_per_km

                    # Left edge: y-grid labels
                    for y in ys2:
                        if y0v < y < y1v:
                            t = ax.text(x0v + x_inset, y, _fmt(y),
                                        ha='left', va='center', **lbl_kw)
                            t._gurt_tick_label = True

                    # Top edge: x-grid labels just below the header
                    y_top_inset = (y1v - y0v) * 0.015
                    if tick_px >= min_px_gap:
                        for x in xs2:
                            if x0v < x < x1v:
                                t = ax.text(x, y1v - y_top_inset, _fmt(x),
                                            ha='center', va='top', **lbl_kw)
                                t._gurt_tick_label = True

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
        Ticks are drawn here on initial plot but are redrawn by _reposition_axes
        on every zoom/resize so they always reflect the current view.
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
            pass   # Ticks and edge labels are drawn by _reposition_axes, which
                   # runs immediately after refresh_plot calls _draw_overlays and
                   # uses pixel-accurate arm sizing.  Drawing them here too would
                   # just produce a stale duplicate that gets removed anyway.

    def _canvas_focus_grab(self, event=None):
        """Take keyboard focus when the canvas is clicked."""
        self.canvas.get_tk_widget().focus_set()

    def _panel_index_for_ax(self, ax):
        for i, a in enumerate(getattr(self, '_axes', [])):
            if a is ax:
                return i
        return None

    def _on_press(self, event):
        if event.button == 1 and event.inaxes:
            mode = self._editor_mode.get()
            if mode in ("unfold_brush", "deglitch_brush", "eraser_brush"):
                self._brush_active = True
                self._drag_ax = event.inaxes
                pidx = self._panel_index_for_ax(event.inaxes)
                if pidx is not None and pidx < len(self.panel_fields):
                    fname = self.panel_fields[pidx][0]
                    self._push_undo([fname])
                if mode == "unfold_brush":
                    self._apply_unfold_brush(event.inaxes, event.xdata, event.ydata)
                elif mode == "deglitch_brush":
                    self._apply_deglitch_brush(event.inaxes, event.xdata, event.ydata)
                else:
                    self._apply_eraser_brush(event.inaxes, event.xdata, event.ydata)
            elif mode == "boundary":
                if event.dblclick:
                    self._boundary_apply()
                else:
                    self._boundary_add_point(event.inaxes,
                                             event.xdata, event.ydata)
            else:
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
        mode = self._editor_mode.get()
        if mode in ("unfold_brush", "deglitch_brush", "eraser_brush"):
            if self._brush_active and event.inaxes and event.xdata is not None:
                if mode == "unfold_brush":
                    self._apply_unfold_brush(event.inaxes, event.xdata, event.ydata)
                elif mode == "deglitch_brush":
                    self._apply_deglitch_brush(event.inaxes, event.xdata, event.ydata)
                else:
                    self._apply_eraser_brush(event.inaxes, event.xdata, event.ydata)
                self._update_brush_overlay(event.inaxes, event.xdata, event.ydata)
            return
        if mode == "boundary":
            return
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
        if event.button == 1 and self._editor_mode.get() in ("unfold_brush", "deglitch_brush", "eraser_brush"):
            self._brush_active = False
            self.refresh_plot()
            return
        if self._editor_mode.get() == "boundary":
            return
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
        self._apply_zoom_only()

    def _show_right_click_menu(self, mpl_event, panel_idx):
        menu = self._mk_menu(self)
        fname = self.panel_fields[panel_idx][0] if panel_idx < len(self.panel_fields) else ""
        menu.add_command(label=f"Panel {panel_idx+1}: {fname}",
                         state="disabled",
                         font=("TkDefaultFont",9,"bold"))
        menu.add_separator()
        menu.add_command(label="Edit Parameters & Colors…",
                         command=lambda: self._open_param_dialog(panel_idx))
        menu.add_command(label="Editor Panel…",
                         command=self._open_editor_panel)
        menu.add_command(label="Duplicate Field…",
                         command=self._duplicate_field_dialog)
        menu.add_separator()
        widget_w = self.canvas.get_tk_widget()
        rx_anchor = widget_w.winfo_rootx() + int(mpl_event.x) + 10
        ry_anchor = widget_w.winfo_rooty() + int(
            self.fig.get_size_inches()[1] * self.fig.dpi - mpl_event.y) - 20
        menu.add_command(label="Colormap Adjust… (offset & scale)",
                         command=lambda: self._open_cmap_slider_window(
                             panel_idx, rx_anchor, ry_anchor))
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
        menu.add_separator()
        export_sub = self._mk_menu(menu)
        export_sub.add_command(
            label="Current sweep…",
            command=lambda: self.export_cfradial("sweep"))
        export_sub.add_command(
            label="Current volume…",
            command=lambda: self.export_cfradial("volume"))
        export_sub.add_command(
            label=f"All loaded files  ({len(self.file_list)})…",
            command=lambda: self.export_cfradial("all"))
        menu.add_cascade(label="Export to CfRadial", menu=export_sub)

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

    # ══════════════════════════════════════════════════════════════════════
    # Editor Panel, Brush Tool, and Duplicate Field
    # ══════════════════════════════════════════════════════════════════════

    def _open_editor_panel(self):
        """Open (or bring to front) the floating Editor Panel window."""
        if self._editor_panel_win is not None and self._editor_panel_win.winfo_exists():
            self._editor_panel_win.lift()
            self._editor_panel_win.focus_set()
            return
        self._editor_panel_win = EditorPanel(self)

    def _duplicate_field_dialog(self):
        """Open the Duplicate Field dialog."""
        if self.radar is None:
            messagebox.showinfo("No file", "Load a radar file first.")
            return
        DuplicateFieldDialog(self, self.available_fields, self._do_duplicate_field)

    def _do_duplicate_field(self, src_name, dst_name):
        """Copy field src_name to a new field dst_name in the loaded radar."""
        if self.radar is None:
            return
        if src_name not in self.radar.fields:
            messagebox.showerror("Duplicate Field",
                                 f"Source field '{src_name}' not found.")
            return
        if not dst_name or not dst_name.strip():
            messagebox.showerror("Duplicate Field", "Output field name cannot be empty.")
            return
        dst_name = dst_name.strip()
        src = self.radar.fields[src_name]
        new_data = src['data'].copy()
        new_field = {k: v for k, v in src.items() if k != 'data'}
        new_field['data'] = new_data
        new_field['long_name'] = dst_name
        self.radar.add_field(dst_name, new_field, replace_existing=True)
        self.available_fields = list(self.radar.fields.keys())
        self._rebuild_fields_menu()
        messagebox.showinfo("Duplicate Field",
                            f"Field '{src_name}' duplicated as '{dst_name}'.")

    # ══════════════════════════════════════════════════════════════════════
    # Undo — brush-stroke undo stack (max 50)
    # ══════════════════════════════════════════════════════════════════════

    def _push_undo(self, field_names):
        """Snapshot the current data for *field_names* onto the undo stack.

        Call this **before** applying any brush stroke.  Each stack entry is a
        dict mapping field_name → deep copy of that field's masked array.
        The stack is capped at _undo_stack_max entries (oldest entry dropped).
        """
        if self.radar is None:
            return
        snapshot = {}
        for fn in field_names:
            if fn in self.radar.fields:
                d = self.radar.fields[fn]['data']
                snapshot[fn] = d.copy() if hasattr(d, 'copy') else np.array(d)
        if snapshot:
            self._undo_stack.append(snapshot)
            if len(self._undo_stack) > self._undo_stack_max:
                self._undo_stack.pop(0)

    def _undo_brush(self):
        """Restore the last snapshot from the undo stack (Ctrl+Z)."""
        if self.radar is None:
            return
        if not self._undo_stack:
            self.status_var.set("Nothing to undo.")
            self.after(1500, lambda: self.status_var.set(""))
            return
        snapshot = self._undo_stack.pop()
        for fn, data in snapshot.items():
            if fn in self.radar.fields:
                self.radar.fields[fn]['data'] = data
        remaining = len(self._undo_stack)
        self.status_var.set(
            f"Undo applied — {remaining} step{'s' if remaining != 1 else ''} remaining.")
        self.after(2000, lambda: self.status_var.set(""))
        self.refresh_plot()

    # ══════════════════════════════════════════════════════════════════════
    # Rotate Azimuths
    # ══════════════════════════════════════════════════════════════════════

    def _rotate_azimuths_dialog(self):
        """Open a dialog to add a fixed degree offset to all azimuth values."""
        if self.radar is None:
            messagebox.showinfo("Rotate Azimuths", "No radar file loaded.")
            return

        win = tk.Toplevel(self)
        win.title("Rotate Azimuths")
        win.resizable(False, False)
        win.grab_set()
        win.configure(bg="#191970")
        win.transient(self)

        hdr = tk.Frame(win, bg="#0d0d3a")
        hdr.pack(fill="x")
        tk.Label(hdr, text="Rotate Azimuths",
                 bg="#0d0d3a", fg="white",
                 font=("TkDefaultFont", 11, "bold"), pady=6).pack()

        body = tk.Frame(win, bg="#191970")
        body.pack(fill="both", expand=True, padx=16, pady=10)

        tk.Label(body, text="Offset (degrees, + = clockwise):",
                 bg="#191970", fg="white",
                 font=("TkDefaultFont", 9)).grid(row=0, column=0, sticky="w", pady=4)
        offset_var = tk.StringVar(value=f"{self._az_rotation_deg:.3f}")
        entry = tk.Entry(body, textvariable=offset_var, width=12,
                         bg="#ffffff", fg="#000", relief="sunken")
        entry.grid(row=0, column=1, sticky="w", padx=8, pady=4)
        entry.focus_set()

        info = tk.Label(body,
            text="Adds the offset to every ray's azimuth (mod 360).\n"
                 "The change is applied to the loaded radar immediately.\n"
                 "Use Edit → Undo to revert.",
            bg="#191970", fg="#aaddff",
            font=("TkDefaultFont", 8), justify="left", wraplength=280)
        info.grid(row=1, column=0, columnspan=2, sticky="w", pady=(4, 8))

        btn_frame = tk.Frame(win, bg="#191970")
        btn_frame.pack(pady=(0, 10))

        def _apply(event=None):
            try:
                offset = float(offset_var.get())
            except ValueError:
                messagebox.showerror("Rotate Azimuths",
                                     "Invalid offset — enter a number.", parent=win)
                return
            self._apply_az_rotation(offset)
            win.destroy()

        tk.Button(btn_frame, text="Apply", command=_apply,
                  bg="#191970", fg="white", relief="raised", padx=14).pack(side="left", padx=6)
        tk.Button(btn_frame, text="Cancel", command=win.destroy,
                  bg="#191970", fg="white", relief="raised", padx=10).pack(side="left", padx=6)
        entry.bind("<Return>", _apply)

    def _apply_az_rotation(self, offset_deg):
        """Rotate all azimuth values in self.radar by *offset_deg* degrees."""
        if self.radar is None:
            return
        az = self.radar.azimuth['data']
        az = (az + offset_deg) % 360.0
        self.radar.azimuth['data'] = az.astype(az.dtype)
        self._az_rotation_deg = (self._az_rotation_deg + offset_deg) % 360.0
        self.status_var.set(
            f"Azimuths rotated by {offset_deg:+.3f}°  "
            f"(cumulative: {self._az_rotation_deg:.3f}°)")
        self.after(3000, lambda: self.status_var.set(""))
        self.refresh_plot()

    # ── Brush overlay (circle drawn on the axes while hovering) ──────────────

    def _update_brush_overlay(self, ax, x_km, y_km):
        """Draw/move a circle showing the brush footprint."""
        import matplotlib.patches as mpatches
        self._remove_brush_overlay()
        r = self._brush_radius_km.get()
        circle = mpatches.Circle((x_km, y_km), r,
                                  linewidth=1.2, edgecolor='cyan',
                                  facecolor='cyan', alpha=0.18,
                                  linestyle='-', zorder=10)
        ax.add_patch(circle)
        self._brush_overlay = (ax, circle)
        self.canvas.draw_idle()

    def _remove_brush_overlay(self):
        if self._brush_overlay is not None:
            ax, circle = self._brush_overlay
            try:
                circle.remove()
            except Exception:
                pass
            self._brush_overlay = None
            self.canvas.draw_idle()

    # ── Gate lookup ───────────────────────────────────────────────────────────

    def _xy_to_gates_in_radius(self, x_km, y_km, radius_km):
        """Return list of (ray_idx, gate_idx) pairs within radius_km of (x_km, y_km).

        Works in Cartesian km space, consistent with how pyart RadarDisplay
        renders PPIs (x = range * sin(az), y = range * cos(az), both in km).
        """
        if self.radar is None:
            return []
        sweep = max(0, min(self.radar.nsweeps - 1, self.sweep_index.get()))
        s_start = int(self.radar.sweep_start_ray_index['data'][sweep])
        s_end   = int(self.radar.sweep_end_ray_index['data'][sweep])

        az_data  = self.radar.azimuth['data'][s_start:s_end + 1]   # degrees
        rng_data = self.radar.range['data'] / 1000.0                 # m → km

        # Pre-compute Cartesian centre of each gate (vectorised)
        az_rad = np.deg2rad(az_data)                      # (nrays,)
        sin_az = np.sin(az_rad)[:, np.newaxis]            # (nrays, 1)
        cos_az = np.cos(az_rad)[:, np.newaxis]
        gx = rng_data[np.newaxis, :] * sin_az             # (nrays, ngates)
        gy = rng_data[np.newaxis, :] * cos_az

        dist2 = (gx - x_km) ** 2 + (gy - y_km) ** 2
        hits  = np.argwhere(dist2 <= radius_km ** 2)      # (N, 2)  [ray, gate]
        return [(s_start + int(r), int(g)) for r, g in hits]

    # ── Forced-unfolding brush (ported from ForcedUnfoldingCmd) ──────────────

    def _apply_unfold_brush(self, ax, x_km, y_km):
        """Apply forced-unfolding to all gates within the brush circle.

        Logic ported directly from ForcedUnfoldingCmd::doIt() in solo3:
          diff = ctr - v
          if |diff| > nyqv:
              n = round(diff / nyqi)        # number of Nyquist intervals to shift
              v += n * nyqi
        """
        if self.radar is None:
            return

        # Which panel / field are we editing?
        pidx = self._panel_index_for_ax(ax)
        if pidx is None or pidx >= len(self.panel_fields):
            return
        fname = self.panel_fields[pidx][0]
        if fname not in self.radar.fields:
            return

        # Nyquist velocity: user override or radar metadata
        nyqv = self._brush_nyquist.get()
        if nyqv <= 0:
            # Try to pull from instrument_parameters
            ip = getattr(self.radar, 'instrument_parameters', None) or {}
            nv = ip.get('nyquist_velocity')
            if nv is not None:
                sweep  = max(0, min(self.radar.nsweeps - 1, self.sweep_index.get()))
                s_start = int(self.radar.sweep_start_ray_index['data'][sweep])
                nyqv = float(nv['data'][s_start])
            else:
                nyqv = 0.0
        if nyqv <= 0:
            messagebox.showwarning("Unfold Brush",
                                   "Could not determine Nyquist velocity.\n"
                                   "Set it manually in the Editor Panel.")
            return

        ctr  = self._brush_center.get()
        nyqi = 2.0 * nyqv
        rcp_nyqi = 1.0 / nyqi

        field = self.radar.fields[fname]
        data  = field['data']                 # masked array, shape (total_rays, ngates)
        bad   = field.get('_FillValue', None)

        gates = self._xy_to_gates_in_radius(x_km, y_km, self._brush_radius_km.get())
        if not gates:
            return

        changed = False
        for ray_idx, gate_idx in gates:
            if ray_idx >= data.shape[0] or gate_idx >= data.shape[1]:
                continue
            v = float(data[ray_idx, gate_idx])
            if np.ma.is_masked(data[ray_idx, gate_idx]):
                continue
            if bad is not None and not np.isnan(bad) and v == bad:
                continue
            if np.isnan(v):
                continue
            diff = ctr - v
            if abs(diff) > nyqv:
                nn = int(diff * rcp_nyqi + (-0.5 if diff < 0 else 0.5))
                data[ray_idx, gate_idx] = v + nn * nyqi
                changed = True

        if changed:
            # Push the modified array back into the radar object
            self.radar.fields[fname]['data'] = data
            # Fast path: update only the mesh colours, skip full redraw
            self._fast_refresh_field(fname)

    def _get_deglitch_nyquist(self):
        """Return the short Nyquist to use for deglitch stepping.

        Priority:
          1. manual override (deglitch_nyquist_mode == "manual")
          2. short or long Nyquist from stagger detection
          3. fixed Nyquist (no stagger)
          4. fall back to the unfold brush Nyquist setting
        Returns float > 0 or None on failure.
        """
        mode = self._deglitch_nyquist_mode.get()
        if mode == "manual":
            v = self._deglitch_manual_nyq.get()
            return v if v > 0 else None
        elif mode == "short":
            v = self._stagger_short_nyq.get()
            if v > 0:
                return v
            # Fall back to fixed if no stagger
            v = self._stagger_fixed_nyq.get()
            if v > 0:
                return v
        elif mode == "long":
            v = self._stagger_long_nyq.get()
            if v > 0:
                return v
            v = self._stagger_fixed_nyq.get()
            if v > 0:
                return v
        # Final fallback: brush nyquist or radar metadata
        v = self._brush_nyquist.get()
        if v > 0:
            return v
        ip = getattr(self.radar, 'instrument_parameters', None) or {}
        nv = ip.get('nyquist_velocity')
        if nv is not None:
            try:
                sweep   = max(0, min(self.radar.nsweeps - 1, self.sweep_index.get()))
                s_start = int(self.radar.sweep_start_ray_index['data'][sweep])
                nyq = float(np.ma.filled(nv['data'], np.nan)[s_start])
                if np.isfinite(nyq) and nyq > 0:
                    return nyq
            except Exception:
                pass
        return None

    def _apply_deglitch_brush(self, ax, x_km, y_km):
        """Deglitch brush: compute median of gates in brush, then fold any gate
        whose value differs from the median by more than *tolerance* toward the
        median using steps of 2 * short_nyquist.

        This is useful for staggered-PRF data where isolated glitches appear at
        multiples of the short Nyquist away from the true velocity.
        """
        if self.radar is None:
            return

        pidx = self._panel_index_for_ax(ax)
        if pidx is None or pidx >= len(self.panel_fields):
            return
        fname = self.panel_fields[pidx][0]
        if fname not in self.radar.fields:
            return

        short_nyq = self._get_deglitch_nyquist()
        if short_nyq is None or short_nyq <= 0:
            messagebox.showwarning(
                "Deglitch Brush",
                "Could not determine short Nyquist velocity.\n"
                "Set it manually in the Editor Panel → Deglitch Settings.")
            return

        tolerance = self._deglitch_tolerance.get()
        step = 2.0 * short_nyq

        field = self.radar.fields[fname]
        data  = field['data']
        bad   = field.get('_FillValue', None)

        gates = self._xy_to_gates_in_radius(x_km, y_km, self._brush_radius_km.get())
        if not gates:
            return

        # Gather valid values to compute median
        valid_vals = []
        for ray_idx, gate_idx in gates:
            if ray_idx >= data.shape[0] or gate_idx >= data.shape[1]:
                continue
            if np.ma.is_masked(data[ray_idx, gate_idx]):
                continue
            v = float(data[ray_idx, gate_idx])
            if np.isnan(v):
                continue
            if bad is not None and not np.isnan(bad) and v == bad:
                continue
            valid_vals.append(v)

        if len(valid_vals) < 3:
            return  # not enough data to compute a reliable median

        median_val = float(np.median(valid_vals))

        changed = False
        for ray_idx, gate_idx in gates:
            if ray_idx >= data.shape[0] or gate_idx >= data.shape[1]:
                continue
            if np.ma.is_masked(data[ray_idx, gate_idx]):
                continue
            v = float(data[ray_idx, gate_idx])
            if np.isnan(v):
                continue
            if bad is not None and not np.isnan(bad) and v == bad:
                continue

            diff = v - median_val
            if abs(diff) > tolerance:
                # Number of steps to shift toward median
                n_steps = int(round(diff / step))
                if n_steps != 0:
                    new_v = v - n_steps * step
                    # Only accept if it brings value closer to median
                    if abs(new_v - median_val) < abs(diff):
                        data[ray_idx, gate_idx] = new_v
                        changed = True

        if changed:
            self.radar.fields[fname]['data'] = data
            self._fast_refresh_field(fname)

    # ══════════════════════════════════════════════════════════════════════
    # Boundary (polygon lasso) tool
    # ══════════════════════════════════════════════════════════════════════

    def _boundary_add_point(self, ax, x_km, y_km):
        """Add a vertex to the in-progress polygon boundary."""
        if self._boundary_ax is None:
            self._boundary_ax = ax
        elif ax is not self._boundary_ax:
            return   # don't mix panels

        self._boundary_poly_pts.append((x_km, y_km))
        pts = self._boundary_poly_pts

        # Remove old artists and redraw the whole polygon so far
        for art in self._boundary_artists:
            try:
                art.remove()
            except Exception:
                pass
        self._boundary_artists = []

        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]

        # Vertices dots
        dot, = ax.plot(xs, ys, 'o', color='yellow', markersize=5,
                       zorder=15, markeredgecolor='black', markeredgewidth=0.5)
        self._boundary_artists.append(dot)

        # Lines connecting vertices (close preview line to first pt if ≥3)
        if len(pts) >= 2:
            lx = xs + [xs[0]] if len(pts) >= 3 else xs
            ly = ys + [ys[0]] if len(pts) >= 3 else ys
            line, = ax.plot(lx, ly, '-', color='yellow', linewidth=1.4,
                            zorder=14, alpha=0.85)
            self._boundary_artists.append(line)

        self.canvas.draw_idle()

    def _boundary_clear(self):
        """Remove all polygon artists and reset boundary state."""
        for art in self._boundary_artists:
            try:
                art.remove()
            except Exception:
                pass
        self._boundary_artists  = []
        self._boundary_poly_pts = []
        self._boundary_ax       = None
        self.canvas.draw_idle()

    def _boundary_apply(self):
        """Apply the chosen edit to all gates inside the closed polygon."""
        pts = self._boundary_poly_pts
        if len(pts) < 3:
            self.status_var.set("Boundary needs at least 3 points.")
            self.after(2000, lambda: self.status_var.set(""))
            return
        if self.radar is None:
            return
        ax = self._boundary_ax
        if ax is None:
            return
        pidx = self._panel_index_for_ax(ax)
        if pidx is None or pidx >= len(self.panel_fields):
            return
        fname = self.panel_fields[pidx][0]
        if fname not in self.radar.fields:
            return

        # Build polygon path for point-in-polygon test
        from matplotlib.path import Path as MplPath
        poly_path = MplPath(pts + [pts[0]])   # closed

        # Find all gates inside the polygon
        sweep   = max(0, min(self.radar.nsweeps - 1, self.sweep_index.get()))
        s_start = int(self.radar.sweep_start_ray_index['data'][sweep])
        s_end   = int(self.radar.sweep_end_ray_index['data'][sweep])
        az_data  = self.radar.azimuth['data'][s_start:s_end + 1]
        rng_data = self.radar.range['data'] / 1000.0

        az_rad = np.deg2rad(az_data)
        sin_az = np.sin(az_rad)[:, np.newaxis]
        cos_az = np.cos(az_rad)[:, np.newaxis]
        gx = rng_data[np.newaxis, :] * sin_az
        gy = rng_data[np.newaxis, :] * cos_az

        nrays, ngates = gx.shape
        pts_flat = np.column_stack([gx.ravel(), gy.ravel()])
        inside = poly_path.contains_points(pts_flat).reshape(nrays, ngates)
        hit_indices = np.argwhere(inside)
        gates = [(s_start + int(r), int(g)) for r, g in hit_indices]

        if not gates:
            self.status_var.set("No gates inside boundary.")
            self.after(2000, lambda: self.status_var.set(""))
            self._boundary_clear()
            return

        # Snapshot for undo
        self._push_undo([fname])

        action = self._boundary_action.get()
        field  = self.radar.fields[fname]
        data   = field['data']
        bad    = field.get('_FillValue', None)
        changed = False

        if action == "erase":
            for ray_idx, gate_idx in gates:
                if ray_idx >= data.shape[0] or gate_idx >= data.shape[1]:
                    continue
                if np.ma.is_masked(data[ray_idx, gate_idx]):
                    continue
                if isinstance(data, np.ma.MaskedArray):
                    data[ray_idx, gate_idx] = np.ma.masked
                else:
                    data[ray_idx, gate_idx] = np.nan
                changed = True

        elif action == "unfold":
            nyqv = self._brush_nyquist.get()
            if nyqv <= 0:
                ip = getattr(self.radar, 'instrument_parameters', None) or {}
                nv = ip.get('nyquist_velocity')
                if nv is not None:
                    try:
                        nyqv = float(np.ma.filled(nv['data'], np.nan)[s_start])
                    except Exception:
                        nyqv = 0
            if nyqv <= 0:
                messagebox.showwarning("Boundary Unfold",
                    "Could not determine Nyquist velocity.\n"
                    "Set it in the Editor Panel → Unfold Brush Settings.")
                self._boundary_clear()
                return
            ctr  = self._brush_center.get()
            nyqi = 2.0 * nyqv
            rcp  = 1.0 / nyqi
            for ray_idx, gate_idx in gates:
                if ray_idx >= data.shape[0] or gate_idx >= data.shape[1]:
                    continue
                if np.ma.is_masked(data[ray_idx, gate_idx]):
                    continue
                v = float(data[ray_idx, gate_idx])
                if np.isnan(v):
                    continue
                diff = ctr - v
                if abs(diff) > nyqv:
                    nn = int(diff * rcp + (-0.5 if diff < 0 else 0.5))
                    data[ray_idx, gate_idx] = v + nn * nyqi
                    changed = True

        elif action == "deglitch":
            short_nyq = self._get_deglitch_nyquist()
            if short_nyq is None or short_nyq <= 0:
                messagebox.showwarning("Boundary Deglitch",
                    "Could not determine short Nyquist.\n"
                    "Set it in the Editor Panel → Deglitch Settings.")
                self._boundary_clear()
                return
            tolerance = self._deglitch_tolerance.get()
            step = 2.0 * short_nyq
            valid_vals = []
            for ray_idx, gate_idx in gates:
                if ray_idx >= data.shape[0] or gate_idx >= data.shape[1]:
                    continue
                if np.ma.is_masked(data[ray_idx, gate_idx]):
                    continue
                v = float(data[ray_idx, gate_idx])
                if not np.isnan(v):
                    valid_vals.append(v)
            if len(valid_vals) < 3:
                self._boundary_clear()
                return
            median_val = float(np.median(valid_vals))
            for ray_idx, gate_idx in gates:
                if ray_idx >= data.shape[0] or gate_idx >= data.shape[1]:
                    continue
                if np.ma.is_masked(data[ray_idx, gate_idx]):
                    continue
                v = float(data[ray_idx, gate_idx])
                if np.isnan(v):
                    continue
                diff = v - median_val
                if abs(diff) > tolerance:
                    n_steps = int(round(diff / step))
                    if n_steps != 0:
                        new_v = v - n_steps * step
                        if abs(new_v - median_val) < abs(diff):
                            data[ray_idx, gate_idx] = new_v
                            changed = True

        if changed:
            self.radar.fields[fname]['data'] = data

        self._boundary_clear()
        self.refresh_plot()
        n_gates = len(gates)
        self.status_var.set(
            f"Boundary {action}: applied to {n_gates} gate{'s' if n_gates!=1 else ''}.")
        self.after(2500, lambda: self.status_var.set(""))

    # ══════════════════════════════════════════════════════════════════════
    # Colormap offset / scale sliders (per field, right-click menu)
    # ══════════════════════════════════════════════════════════════════════

    def _open_cmap_slider_window(self, panel_idx, anchor_x, anchor_y):
        """Open (or refresh) a floating window with offset/scale sliders
        for the field shown in *panel_idx*."""
        if panel_idx >= len(self.panel_fields) or not self.panel_fields[panel_idx]:
            return
        fname, flabel, vmin, vmax, _ = self.panel_fields[panel_idx]

        # Close any existing slider window
        if self._cmap_slider_win is not None:
            try:
                self._cmap_slider_win.destroy()
            except Exception:
                pass
            self._cmap_slider_win = None

        adj = self._cmap_adjustments.setdefault(fname, [0.0, 1.0])

        win = tk.Toplevel(self)
        self._cmap_slider_win = win
        win.title(f"Colormap — {flabel}")
        win.configure(bg="#191970")
        win.resizable(False, False)
        win.transient(self)
        win.protocol("WM_DELETE_WINDOW", lambda: self._close_cmap_slider(win))

        BG, FG = "#191970", "white"

        hdr = tk.Frame(win, bg="#0d0d3a")
        hdr.pack(fill="x")
        tk.Label(hdr, text=f"Colormap  ·  {flabel}",
                 bg="#0d0d3a", fg=FG,
                 font=("TkDefaultFont", 10, "bold"), pady=5).pack()

        body = tk.Frame(win, bg=BG)
        body.pack(fill="both", expand=True, padx=12, pady=8)

        data_range = vmax - vmin

        # ── Offset slider ─────────────────────────────────────────────────
        tk.Label(body, text="Offset", bg=BG, fg=FG,
                 font=("TkDefaultFont", 9, "bold")).pack(anchor="w")

        off_val_var = tk.StringVar(value=f"{adj[0]:.2f}")
        off_lbl = tk.Label(body, textvariable=off_val_var, bg=BG, fg="#aaddff",
                           font=("TkDefaultFont", 9), width=8, anchor="center")
        off_lbl.pack()

        off_slider = tk.Scale(
            body, from_=-data_range, to=data_range,
            resolution=data_range / 200,
            orient=tk.VERTICAL, length=160,
            bg=BG, fg=FG, troughcolor="#333366",
            highlightbackground=BG, activebackground="#4444aa",
            showvalue=False, bd=0)
        off_slider.set(adj[0])
        off_slider.pack(pady=(0, 4))

        # ── Scale slider ──────────────────────────────────────────────────
        tk.Label(body, text="Scale", bg=BG, fg=FG,
                 font=("TkDefaultFont", 9, "bold")).pack(anchor="w")

        sc_val_var = tk.StringVar(value=f"{adj[1]:.2f}")
        sc_lbl = tk.Label(body, textvariable=sc_val_var, bg=BG, fg="#aaddff",
                          font=("TkDefaultFont", 9), width=8, anchor="center")
        sc_lbl.pack()

        sc_slider = tk.Scale(
            body, from_=0.1, to=4.0,
            resolution=0.01,
            orient=tk.VERTICAL, length=160,
            bg=BG, fg=FG, troughcolor="#333366",
            highlightbackground=BG, activebackground="#4444aa",
            showvalue=False, bd=0)
        sc_slider.set(adj[1])
        sc_slider.pack(pady=(0, 8))

        def _on_change(*_):
            o = off_slider.get()
            s = sc_slider.get()
            off_val_var.set(f"{o:.2f}")
            sc_val_var.set(f"{s:.2f}")
            self._cmap_adjustments[fname] = [o, s]
            self.refresh_plot()

        off_slider.config(command=_on_change)
        sc_slider.config(command=_on_change)

        tk.Button(win, text="Reset",
                  command=lambda: self._reset_cmap_adjustment(fname,
                                                               off_slider, sc_slider,
                                                               off_val_var, sc_val_var),
                  bg=BG, fg=FG, relief="raised", padx=8).pack(fill="x", padx=12, pady=2)
        tk.Button(win, text="Close",
                  command=lambda: self._close_cmap_slider(win),
                  bg=BG, fg=FG, relief="raised", padx=8).pack(fill="x", padx=12, pady=(2, 10))

        win.update_idletasks()
        win.geometry(f"+{anchor_x}+{anchor_y}")

    def _reset_cmap_adjustment(self, fname, off_slider, sc_slider,
                                off_val_var, sc_val_var):
        self._cmap_adjustments[fname] = [0.0, 1.0]
        off_slider.set(0.0)
        sc_slider.set(1.0)
        off_val_var.set("0.00")
        sc_val_var.set("1.00")
        self.refresh_plot()

    def _close_cmap_slider(self, win):
        try:
            win.destroy()
        except Exception:
            pass
        self._cmap_slider_win = None

    # ── Eraser brush — mask gates (set to missing) ────────────────────────────

    def _apply_eraser_brush(self, ax, x_km, y_km):
        """Erase (mask) all gates within the brush circle.

        Gates are set to masked/NaN so they appear as missing data.
        Works on the field currently displayed in the clicked panel.
        """
        if self.radar is None:
            return

        pidx = self._panel_index_for_ax(ax)
        if pidx is None or pidx >= len(self.panel_fields):
            return
        fname = self.panel_fields[pidx][0]
        if fname not in self.radar.fields:
            return

        field = self.radar.fields[fname]
        data  = field['data']

        gates = self._xy_to_gates_in_radius(x_km, y_km, self._brush_radius_km.get())
        if not gates:
            return

        changed = False
        for ray_idx, gate_idx in gates:
            if ray_idx >= data.shape[0] or gate_idx >= data.shape[1]:
                continue
            if np.ma.is_masked(data[ray_idx, gate_idx]):
                continue   # already masked — skip
            if isinstance(data, np.ma.MaskedArray):
                data[ray_idx, gate_idx] = np.ma.masked
            else:
                data[ray_idx, gate_idx] = np.nan
            changed = True

        if changed:
            self.radar.fields[fname]['data'] = data
            self._fast_refresh_field(fname)

    # ══════════════════════════════════════════════════════════════════════
    # Offset Elevations
    # ══════════════════════════════════════════════════════════════════════

    def _offset_elevations_dialog(self):
        """Open a dialog to add a fixed degree offset to all elevation values."""
        if self.radar is None:
            messagebox.showinfo("Offset Elevations", "No radar file loaded.")
            return

        win = tk.Toplevel(self)
        win.title("Offset Elevations")
        win.resizable(False, False)
        win.grab_set()
        win.configure(bg="#191970")
        win.transient(self)

        hdr = tk.Frame(win, bg="#0d0d3a")
        hdr.pack(fill="x")
        tk.Label(hdr, text="Offset Elevations",
                 bg="#0d0d3a", fg="white",
                 font=("TkDefaultFont", 11, "bold"), pady=6).pack()

        body = tk.Frame(win, bg="#191970")
        body.pack(fill="both", expand=True, padx=16, pady=10)

        tk.Label(body, text="Offset (degrees, + = up):",
                 bg="#191970", fg="white",
                 font=("TkDefaultFont", 9)).grid(row=0, column=0, sticky="w", pady=4)
        offset_var = tk.StringVar(value=f"{self._el_offset_deg:.4f}")
        entry = tk.Entry(body, textvariable=offset_var, width=12,
                         bg="#ffffff", fg="#000", relief="sunken")
        entry.grid(row=0, column=1, sticky="w", padx=8, pady=4)
        entry.focus_set()

        tk.Label(body,
            text="Adds the offset to every ray's elevation angle.\n"
                 "Cumulative: applied on top of any previous offset.\n"
                 "Use Edit → Undo to revert.",
            bg="#191970", fg="#aaddff",
            font=("TkDefaultFont", 8), justify="left", wraplength=280
        ).grid(row=1, column=0, columnspan=2, sticky="w", pady=(4, 8))

        btn_frame = tk.Frame(win, bg="#191970")
        btn_frame.pack(pady=(0, 10))

        def _apply(event=None):
            try:
                offset = float(offset_var.get())
            except ValueError:
                messagebox.showerror("Offset Elevations",
                                     "Invalid offset — enter a number.", parent=win)
                return
            self._apply_el_offset(offset)
            win.destroy()

        tk.Button(btn_frame, text="Apply", command=_apply,
                  bg="#191970", fg="white", relief="raised", padx=14).pack(side="left", padx=6)
        tk.Button(btn_frame, text="Cancel", command=win.destroy,
                  bg="#191970", fg="white", relief="raised", padx=10).pack(side="left", padx=6)
        entry.bind("<Return>", _apply)

    def _apply_el_offset(self, offset_deg):
        """Add *offset_deg* to every elevation value in self.radar."""
        if self.radar is None:
            return
        el = self.radar.elevation['data']
        self.radar.elevation['data'] = (el + offset_deg).astype(el.dtype)
        # Also update fixed_angle so sweep labels stay consistent
        if hasattr(self.radar, 'fixed_angle') and self.radar.fixed_angle is not None:
            fa = self.radar.fixed_angle['data']
            self.radar.fixed_angle['data'] = (fa + offset_deg).astype(fa.dtype)
        self._el_offset_deg += offset_deg
        self.status_var.set(
            f"Elevations offset by {offset_deg:+.4f}°  "
            f"(cumulative: {self._el_offset_deg:+.4f}°)")
        self.after(3000, lambda: self.status_var.set(""))
        self.refresh_plot()

    # ══════════════════════════════════════════════════════════════════════
    # Set Geolocation
    # ══════════════════════════════════════════════════════════════════════

    def _set_geolocation_dialog(self):
        """Open a dialog to set the radar's latitude, longitude, and altitude."""
        if self.radar is None:
            messagebox.showinfo("Set Geolocation", "No radar file loaded.")
            return

        # Read current values
        try:
            cur_lat = float(np.ma.filled(self.radar.latitude['data'],  0.0).flat[0])
            cur_lon = float(np.ma.filled(self.radar.longitude['data'], 0.0).flat[0])
            cur_alt = float(np.ma.filled(self.radar.altitude['data'],  0.0).flat[0])
        except Exception:
            cur_lat, cur_lon, cur_alt = 0.0, 0.0, 0.0

        win = tk.Toplevel(self)
        win.title("Set Geolocation")
        win.resizable(False, False)
        win.grab_set()
        win.configure(bg="#191970")
        win.transient(self)

        hdr = tk.Frame(win, bg="#0d0d3a")
        hdr.pack(fill="x")
        tk.Label(hdr, text="Set Radar Geolocation",
                 bg="#0d0d3a", fg="white",
                 font=("TkDefaultFont", 11, "bold"), pady=6).pack()

        body = tk.Frame(win, bg="#191970")
        body.pack(fill="both", expand=True, padx=16, pady=10)

        fields = [
            ("Latitude (°, N positive):",  cur_lat),
            ("Longitude (°, E positive):", cur_lon),
            ("Altitude (m MSL):",          cur_alt),
        ]
        evars = []
        for row, (label, default) in enumerate(fields):
            tk.Label(body, text=label, bg="#191970", fg="white",
                     font=("TkDefaultFont", 9)).grid(row=row, column=0, sticky="w", pady=4)
            var = tk.StringVar(value=f"{default:.6f}" if row < 2 else f"{default:.1f}")
            tk.Entry(body, textvariable=var, width=16,
                     bg="#ffffff", fg="#000", relief="sunken"
                     ).grid(row=row, column=1, sticky="w", padx=8, pady=4)
            evars.append(var)

        tk.Label(body,
            text="Updates the radar origin used for map overlays and exports.\n"
                 "Does not affect range/azimuth data.",
            bg="#191970", fg="#aaddff",
            font=("TkDefaultFont", 8), justify="left", wraplength=300
        ).grid(row=len(fields), column=0, columnspan=2, sticky="w", pady=(4, 8))

        btn_frame = tk.Frame(win, bg="#191970")
        btn_frame.pack(pady=(0, 10))

        def _apply(event=None):
            try:
                lat = float(evars[0].get())
                lon = float(evars[1].get())
                alt = float(evars[2].get())
            except ValueError:
                messagebox.showerror("Set Geolocation",
                                     "Invalid value — all fields must be numbers.", parent=win)
                return
            if not (-90 <= lat <= 90):
                messagebox.showerror("Set Geolocation",
                                     "Latitude must be between -90 and 90.", parent=win)
                return
            if not (-180 <= lon <= 180):
                messagebox.showerror("Set Geolocation",
                                     "Longitude must be between -180 and 180.", parent=win)
                return
            self._apply_geolocation(lat, lon, alt)
            win.destroy()

        tk.Button(btn_frame, text="Apply", command=_apply,
                  bg="#191970", fg="white", relief="raised", padx=14).pack(side="left", padx=6)
        tk.Button(btn_frame, text="Cancel", command=win.destroy,
                  bg="#191970", fg="white", relief="raised", padx=10).pack(side="left", padx=6)

    def _apply_geolocation(self, lat, lon, alt_m):
        """Write lat/lon/alt into the radar object."""
        if self.radar is None:
            return
        def _set(field, val):
            d = self.radar.__dict__.get(field, {})
            if isinstance(d, dict) and 'data' in d:
                arr = d['data']
                d['data'] = np.array([val], dtype=arr.dtype if hasattr(arr, 'dtype') else float)
            else:
                self.radar.__dict__[field] = {'data': np.array([val], dtype=float)}
        _set('latitude',  lat)
        _set('longitude', lon)
        _set('altitude',  alt_m)
        self.status_var.set(
            f"Geolocation set: {lat:.6f}°N  {lon:.6f}°E  {alt_m:.1f} m MSL")
        self.after(3000, lambda: self.status_var.set(""))
        self.refresh_plot()

    # ══════════════════════════════════════════════════════════════════════
    # Edit Range / Gate Spacing
    # ══════════════════════════════════════════════════════════════════════

    def _edit_range_dialog(self):
        """Open a dialog to change range-to-first-gate and gate spacing."""
        if self.radar is None:
            messagebox.showinfo("Edit Range", "No radar file loaded.")
            return

        rng = self.radar.range['data']   # metres
        cur_r0      = float(rng[0])
        cur_spacing = float(rng[1] - rng[0]) if len(rng) > 1 else float(rng[0])
        ngates      = len(rng)

        win = tk.Toplevel(self)
        win.title("Edit Range / Gate Spacing")
        win.resizable(False, False)
        win.grab_set()
        win.configure(bg="#191970")
        win.transient(self)

        hdr = tk.Frame(win, bg="#0d0d3a")
        hdr.pack(fill="x")
        tk.Label(hdr, text="Edit Range / Gate Spacing",
                 bg="#0d0d3a", fg="white",
                 font=("TkDefaultFont", 11, "bold"), pady=6).pack()

        body = tk.Frame(win, bg="#191970")
        body.pack(fill="both", expand=True, padx=16, pady=10)

        tk.Label(body, text=f"Number of gates: {ngates}  (read-only)",
                 bg="#191970", fg="#aaddff",
                 font=("TkDefaultFont", 8)).grid(row=0, column=0, columnspan=2, sticky="w", pady=2)

        labels  = ["Range to first gate (m):", "Gate spacing (m):"]
        defaults = [cur_r0, cur_spacing]
        evars   = []
        for i, (lbl, dflt) in enumerate(zip(labels, defaults)):
            tk.Label(body, text=lbl, bg="#191970", fg="white",
                     font=("TkDefaultFont", 9)).grid(row=i+1, column=0, sticky="w", pady=4)
            var = tk.StringVar(value=f"{dflt:.2f}")
            tk.Entry(body, textvariable=var, width=14,
                     bg="#ffffff", fg="#000", relief="sunken"
                     ).grid(row=i+1, column=1, sticky="w", padx=8, pady=4)
            evars.append(var)

        tk.Label(body,
            text="Range array is rebuilt from these values.\n"
                 "All sweeps are updated simultaneously.",
            bg="#191970", fg="#aaddff",
            font=("TkDefaultFont", 8), justify="left", wraplength=280
        ).grid(row=3, column=0, columnspan=2, sticky="w", pady=(4, 8))

        btn_frame = tk.Frame(win, bg="#191970")
        btn_frame.pack(pady=(0, 10))

        def _apply(event=None):
            try:
                r0      = float(evars[0].get())
                spacing = float(evars[1].get())
            except ValueError:
                messagebox.showerror("Edit Range",
                                     "Invalid value — both fields must be numbers.", parent=win)
                return
            if spacing <= 0:
                messagebox.showerror("Edit Range",
                                     "Gate spacing must be positive.", parent=win)
                return
            self._apply_range_edit(r0, spacing)
            win.destroy()

        tk.Button(btn_frame, text="Apply", command=_apply,
                  bg="#191970", fg="white", relief="raised", padx=14).pack(side="left", padx=6)
        tk.Button(btn_frame, text="Cancel", command=win.destroy,
                  bg="#191970", fg="white", relief="raised", padx=10).pack(side="left", padx=6)

    def _apply_range_edit(self, r0_m, spacing_m):
        """Rebuild the range array with new first-gate range and gate spacing."""
        if self.radar is None:
            return
        ngates = len(self.radar.range['data'])
        new_range = r0_m + np.arange(ngates, dtype=float) * spacing_m
        self.radar.range['data'] = new_range.astype(self.radar.range['data'].dtype)
        # Propagate to instrument_parameters if present
        ip = getattr(self.radar, 'instrument_parameters', None) or {}
        for key in ('range_gate_length', 'gate_spacing'):
            if key in ip and isinstance(ip[key], dict) and 'data' in ip[key]:
                ip[key]['data'][:] = spacing_m
        self.status_var.set(
            f"Range updated: first gate {r0_m:.1f} m, spacing {spacing_m:.1f} m  "
            f"({ngates} gates → max {r0_m + (ngates-1)*spacing_m:.1f} m)")
        self.after(3000, lambda: self.status_var.set(""))
        self.refresh_plot()

    def _apply_zoom_only(self):
        """Update axis limits only — no fig.clear(), no plot_ppi(), no colorbar rebuild.

        Called by rubber-band zoom, reset zoom, center-on-click, and zoom-level
        changes.  Falls back to a full refresh_plot if the axes cache is stale.
        """
        if not getattr(self, '_axes', None):
            self.refresh_plot()
            return

        fig_w_px, fig_h_px = self.fig.get_size_inches() * self.fig.dpi
        mr = self.max_range.get()

        for idx, ax in enumerate(self._axes):
            lims = self.panel_limits[idx] if idx < len(self.panel_limits) else None

            if lims:
                # Rubber-band zoom or center-on-click: derive from stored lims
                xl0, xl1, yl0, yl1 = lims
                cx   = (xl0 + xl1) / 2.0
                cy   = (yl0 + yl1) / 2.0
                half = max(xl1 - xl0, yl1 - yl0) / 2.0
            else:
                # Reset / full-range: always go back to radar origin
                cx, cy, half = 0.0, 0.0, mr

            # Keep _panel_meta in sync so _reposition_axes stays consistent
            if idx < len(self._panel_meta):
                self._panel_meta[idx] = (cx, cy, half)

            pos = ax.get_position()
            cell_w_px = pos.width  * fig_w_px
            cell_h_px = pos.height * fig_h_px
            aspect = (cell_w_px / cell_h_px) if cell_h_px > 0 else 1.0
            if aspect >= 1.0:
                x_half, y_half = half * aspect, half
            else:
                x_half, y_half = half, half / aspect
            ax.set_xlim(cx - x_half, cx + x_half)
            ax.set_ylim(cy - y_half, cy + y_half)

        self.canvas.draw_idle()

    def _fast_refresh_field(self, fname):
        """Update only the pcolormesh colour data for panels showing *fname*.

        Called by the unfold brush during active dragging so we avoid a full
        fig.clear() + plot_ppi() round-trip on every mouse-move event.
        Falls back to a full refresh_plot if the mesh cache entry is missing.
        """
        sweep   = max(0, min(self.radar.nsweeps - 1, self.sweep_index.get()))
        s_start = int(self.radar.sweep_start_ray_index['data'][sweep])
        s_end   = int(self.radar.sweep_end_ray_index['data'][sweep])
        field_data = self.radar.fields[fname]['data']
        sweep_data = field_data[s_start : s_end + 1, :]

        updated = False
        for idx, fi in enumerate(self.panel_fields):
            if fi is None or fi[0] != fname:
                continue
            mesh = self._ppi_meshes.get((idx, fname))
            if mesh is None:
                # Cache miss — fall back to full redraw and bail
                self.refresh_plot()
                return
            flat = np.ma.filled(sweep_data, np.nan).ravel()
            mesh.set_array(flat)
            updated = True

        if updated:
            self.canvas.draw_idle()

    def _reset_panel_zoom(self, idx):
        self.panel_limits = [None] * self.n_panels.get()
        self._apply_zoom_only()

    def _reset_all_zoom(self):
        self.panel_limits = [None] * self.n_panels.get()
        self._apply_zoom_only()

    def _zoom_all(self, factor):
        if factor == "data":
            self.max_range.set(150)
        elif isinstance(factor, float):
            self.max_range.set(self.max_range.get() * factor)
        else:
            self.max_range.set(float(factor))
        self.panel_limits = [None] * self.n_panels.get()
        self._apply_zoom_only()

    def _set_max_range(self):
        def _on_set():
            self.panel_limits = [None] * self.n_panels.get()
            self._apply_zoom_only()
        self._ask_float("Max range (km):", self.max_range, _on_set)

    def _center_on_click(self):
        lc = getattr(self, '_last_click', None)
        if lc is None:
            return
        cx, cy = lc
        mr = self.max_range.get()
        new = [cx-mr, cx+mr, cy-mr, cy+mr]
        self.panel_limits = [new] * self.n_panels.get()
        self._apply_zoom_only()

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

    # ══════════════════════════════════════════════════════════════════════════════
    # CfRadial Export
    # ══════════════════════════════════════════════════════════════════════════════

    def _set_cfrad_export_dir(self):
        """Let the user pre-set the CfRadial output directory without starting an export."""
        d = filedialog.askdirectory(title="Set CfRadial export directory")
        if d:
            self._cfrad_export_dir = d
            messagebox.showinfo("CfRadial Export Directory",
                                f"Export directory set to:\n{d}")

    def _cfrad_default_name(self, radar=None):
        """Build a default CfRadial filename from the given (or current) radar."""
        r = radar if radar is not None else self.radar
        if r is None:
            return "cfrad_export.nc"
        try:
            t_str = r.time['units'].split('since')[-1].strip()
            for fmt in ('%Y-%m-%dT%H:%M:%SZ', '%Y/%m/%d %H:%M:%S',
                        '%Y-%m-%dT%H:%M:%S', '%Y-%m-%d %H:%M:%S'):
                try:
                    scan_dt = datetime.strptime(t_str, fmt)
                    break
                except ValueError:
                    continue
            else:
                scan_dt = datetime.utcnow()
        except Exception:
            scan_dt = datetime.utcnow()
        ts   = scan_dt.strftime('%Y%m%d_%H%M%S')
        site = r.metadata.get('instrument_name', '') or r.metadata.get('site_name', 'UNKN')
        site = str(site).strip().replace(' ', '_') or 'UNKN'
        try:
            sm_raw = r.sweep_mode['data'][0]
            if isinstance(sm_raw, (bytes, np.bytes_)):
                sm_raw = sm_raw.decode(errors='ignore')
            sm_raw = str(sm_raw).strip().lower()
            # Follows Radx::sweepModeToShortStr() exactly
            _sm_short = {
                'not_set':                  'XXX',
                'sector':                   'PPI',
                'coplane':                  'COP',
                'rhi':                      'RHI',
                'vertical_pointing':        'VER',
                'idle':                     'IDL',
                'azimuth_surveillance':     'SUR',
                'elevation_surveillance':   'AIR',
                'sunscan':                  'SUN',
                'sunscan_rhi':              'SRH',
                'calibration':              'CAL',
                'pointing':                 'MAN',
                'manual_ppi':               'MAN',
                'manual_rhi':               'MAN',
                'doppler_beam_swinging':    'DBS',
                'complex_trajectory':       'TRJ',
                'electronic_steering':      'PAR',
                'apar_fore_doppler':        'RHI',
                'apar_aft_doppler':         'RHI',
                'apar_dualpol_rhi':         'RHI',
                'apar_sector_ppi':          'PPI',
            }
            scan_type = _sm_short.get(sm_raw, 'SUR')
        except Exception:
            scan_type = 'SUR'
        return f"cfrad.{ts}_{site}_{scan_type}.nc"

    def export_cfradial(self, scope="sweep"):
        """Open the CfRadial export dialog.

        scope : "sweep"  – export the current sweep only (default / right-click)
                "volume" – export the whole currently-loaded file
                "all"    – batch-export every file in the loaded folder
        """
        if not PYART_AVAILABLE:
            messagebox.showerror("Export CfRadial", "pyart is required for CfRadial export.")
            return
        if self.radar is None and scope != "all":
            messagebox.showinfo("Export CfRadial", "No radar file loaded.")
            return
        CfRadialExportDialog(self, scope)

    def _do_cfradial_export(self, out_dir, scope):
        """Worker called by CfRadialExportDialog once the user confirms."""
        if not PYART_AVAILABLE:
            return
        os.makedirs(out_dir, exist_ok=True)

        # True if the currently-loaded file is already a CfRadial file
        _current_is_cfrad = False
        if self.file_list and self.file_index < len(self.file_list):
            _p = self.file_list[self.file_index]
            _current_is_cfrad = _p.lower().endswith('.nc') or _p.lower().endswith('.h5') or _p.lower().endswith('.hdf5')

        def _ensure_sweep_mode(radar, src_is_cfradial):
            """Guarantee sweep_mode contains valid CfRadial strings.

            For CfRadial inputs the existing value is kept as-is.
            For all other formats (DORADE, NEXRAD, etc.) we default to
            'azimuth_surveillance' following the Radx convention.
            """
            nsweeps = radar.nsweeps
            if src_is_cfradial:
                if not hasattr(radar, 'sweep_mode') or radar.sweep_mode is None:
                    radar.sweep_mode = {
                        'data': np.array(
                            ['azimuth_surveillance'.ljust(32)[:32]] * nsweeps,
                            dtype='S32')}
            else:
                # Non-CfRadial source: keep DORADE-derived value if present and
                # non-empty, otherwise fall back to azimuth_surveillance.
                try:
                    existing = radar.sweep_mode['data']
                    decoded = [
                        (e.decode(errors='ignore') if isinstance(e, (bytes, np.bytes_)) else str(e)).strip()
                        for e in existing
                    ]
                    if all(d for d in decoded):
                        return   # already valid
                except Exception:
                    pass
                radar.sweep_mode = {
                    'data': np.array(
                        ['azimuth_surveillance'.ljust(32)[:32]] * nsweeps,
                        dtype='S32')}

        def _write(radar, out_dir, src_is_cfradial=False):
            _ensure_sweep_mode(radar, src_is_cfradial)
            name = self._cfrad_default_name(radar)
            dest = os.path.join(out_dir, name)
            pyart.io.write_cfradial(dest, radar)
            return dest

        try:
            if scope == "sweep":
                # Extract only the currently displayed sweep
                sweep_idx = self.sweep_index.get()
                r = self.radar.extract_sweeps([sweep_idx])
                dest = _write(r, out_dir, _current_is_cfrad)
                messagebox.showinfo("Export CfRadial", f"Saved sweep {sweep_idx}:\n{dest}")

            elif scope == "volume":
                dest = _write(self.radar, out_dir, _current_is_cfrad)
                messagebox.showinfo("Export CfRadial", f"Saved volume:\n{dest}")

            elif scope == "all":
                if not self.file_list:
                    messagebox.showwarning("Export CfRadial", "No files loaded.")
                    return
                orig_idx = self.file_index
                saved = []
                errors = []
                for i, fpath in enumerate(self.file_list):
                    try:
                        self.status_var.set(
                            f"Exporting {i+1}/{len(self.file_list)}: {os.path.basename(fpath)}…")
                        self.update_idletasks()
                        _fext = fpath.lower()
                        _fis_cfrad = (_fext.endswith('.nc') or _fext.endswith('.h5')
                                      or _fext.endswith('.hdf5'))
                        if _is_dorade(fpath):
                            df = DoradeFile(fpath)
                            r  = dorade_to_pyart_radar(df)
                            _fis_cfrad = False
                        else:
                            r = pyart.io.read(fpath, linear_interp=False)
                        dest = _write(r, out_dir, _fis_cfrad)
                        saved.append(dest)
                    except Exception as e:
                        errors.append(f"{os.path.basename(fpath)}: {e}")
                # Restore display
                self.file_index = orig_idx
                self.load_current_file()
                msg = f"Exported {len(saved)} of {len(self.file_list)} files to:\n{out_dir}"
                if errors:
                    msg += f"\n\nErrors ({len(errors)}):\n" + "\n".join(errors[:5])
                    if len(errors) > 5:
                        msg += f"\n… and {len(errors)-5} more"
                messagebox.showinfo("Export CfRadial — Done", msg)
        except Exception as e:
            messagebox.showerror("Export CfRadial", f"Export failed:\n{e}")
        finally:
            self.status_var.set("")

    def _on_mouse_move(self, event):
        if event.inaxes:
            x, y = event.xdata, event.ydata
            az   = (np.degrees(np.arctan2(x, y)) + 360) % 360
            rng  = np.sqrt(x**2 + y**2)
            mode = self._editor_mode.get()
            if mode == "unfold_brush":
                mode_hint = "  [UNFOLD BRUSH]"
            elif mode == "deglitch_brush":
                mode_hint = "  [DEGLITCH BRUSH]"
            elif mode == "eraser_brush":
                mode_hint = "  [ERASER BRUSH]"
            elif mode == "boundary":
                n_pts = len(self._boundary_poly_pts)
                mode_hint = f"  [BOUNDARY — {n_pts} pts — dbl-click or Enter to apply]"
            else:
                mode_hint = ""
            self.cursor_var.set(
                f"X: {x:.1f} km   Y: {y:.1f} km   "
                f"Az: {az:.1f}°   Range: {rng:.1f} km{mode_hint}")
            self._last_click = (x, y)
            # Move brush preview circle
            if mode in ("unfold_brush", "deglitch_brush", "eraser_brush"):
                self._update_brush_overlay(event.inaxes, x, y)
        else:
            self.cursor_var.set("Cursor: —")
            self._remove_brush_overlay()

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
            "GURTv3.3 — Graphic Utility Radar Toolkit\n"
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