
"""
MANTIS Main GUI V12B - Reference HWP Lead-In + Tangent Circle
Author: Sikandar Ali / MANTIS project support

Purpose:
- One GUI to manually test the main instruments before full automation:
  1) Aerotech XYZ stage using Automation1 Python API
  2) Thorlabs KDC101 + PRM1-Z8 waveplate rotation mount using Kinesis DLL
  3) Thorlabs SC10 shutter controller through serial RS232/USB adapter
  4) Camera preview/capture placeholder for ThorCam/manual test

IMPORTANT:
- Keep shutter CLOSED by default.
- Use very small Z steps.
- Confirm real stage limits with your supervisor before experiments.
- Confirm SC10 serial commands from the SC10 manual before enabling laser exposure.
"""

import os
import sys
import time
import csv
from datetime import datetime

import numpy as np
import imageio.v2 as imageio
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import threading
import queue
import subprocess
import math
import tkinter as tk
from tkinter import messagebox, filedialog
from ctypes import *

# Optional imports: GUI should still open even if hardware packages are missing
try:
    import automation1 as a1
except Exception:
    a1 = None

try:
    import serial
except Exception:
    serial = None

try:
    from pylablib.devices import uc480
except Exception:
    uc480 = None


# ============================================================
# 1) AEROTECH STAGE DEVICE CLASS
# ============================================================

class AerotechStage:
    def __init__(self, log_func):
        self.log = log_func
        self.controller = None
        self.axes = ["X", "Y", "Z"]

        # Ask supervisor to confirm these real safe limits
        self.limits = {
            "X": (-50.0, 50.0),
            "Y": (-50.0, 50.0),
            "Z": (0.0, 5.0)
        }

    def connect(self):
        if a1 is None:
            raise RuntimeError("automation1 package is not available in this Python environment.")
        self.controller = a1.Controller.connect()
        self.log("Aerotech: connected to Automation1 controller")

    def check_connected(self):
        if self.controller is None:
            raise RuntimeError("Aerotech stage is not connected.")

    def enable_all(self):
        self.check_connected()
        self.controller.runtime.commands.motion.enable(self.axes)
        self.log("Aerotech: enabled X, Y, Z")

    def stop(self):
        self.check_connected()
        self.controller.runtime.commands.motion.abort(self.axes)
        self.log("Aerotech: STOP sent to X, Y, Z")

    def clear_faults(self):
        self.check_connected()
        self.controller.runtime.commands.motion.abort(self.axes)
        self.controller.runtime.commands.motion.waitformotiondone(self.axes)
        self.controller.runtime.commands.motion.enable(self.axes)
        self.log("Aerotech: clear fault attempt done")

    def get_axis_data(self):
        self.check_connected()

        config = a1.StatusItemConfiguration()
        for axis in self.axes:
            config.axis.add(a1.AxisStatusItem.ProgramPositionFeedback, axis)
            config.axis.add(a1.AxisStatusItem.AxisStatus, axis)
            config.axis.add(a1.AxisStatusItem.DriveStatus, axis)
            config.axis.add(a1.AxisStatusItem.AxisFault, axis)

        results = self.controller.runtime.status.get_status_items(config)
        data = {}

        for axis in self.axes:
            pos = results.axis.get(a1.AxisStatusItem.ProgramPositionFeedback, axis).value
            axis_status = int(results.axis.get(a1.AxisStatusItem.AxisStatus, axis).value)
            drive_status = int(results.axis.get(a1.AxisStatusItem.DriveStatus, axis).value)
            axis_fault = int(results.axis.get(a1.AxisStatusItem.AxisFault, axis).value)

            enabled = (drive_status & a1.DriveStatus.Enabled) == a1.DriveStatus.Enabled
            homed = (axis_status & a1.AxisStatus.Homed) == a1.AxisStatus.Homed
            faulted = axis_fault != 0

            data[axis] = {
                "position": float(pos),
                "enabled": bool(enabled),
                "homed": bool(homed),
                "faulted": bool(faulted),
                "fault_code": axis_fault
            }

        return data

    def check_target_safe(self, axis, target):
        data = self.get_axis_data()
        d = data[axis]

        if d["faulted"]:
            raise RuntimeError(f"{axis} has fault code {d['fault_code']}")

        if not d["enabled"]:
            raise RuntimeError(f"{axis} is not enabled")

        low, high = self.limits[axis]
        if not (low <= target <= high):
            raise RuntimeError(f"{axis} target {target:.6f} mm outside limits {low} to {high}")

        if not d["homed"]:
            self.log(f"Aerotech warning: {axis} is not homed")

    def move_relative(self, axis, distance, speed):
        self.check_connected()
        current = self.get_axis_data()[axis]["position"]
        target = current + distance
        self.check_target_safe(axis, target)

        self.controller.runtime.commands.motion.moveincremental([axis], [distance], [speed])
        self.controller.runtime.commands.motion.waitformotiondone([axis])
        self.log(f"Aerotech: moved {axis} relatively by {distance:.6f} mm")

    def move_absolute(self, axis, target, speed):
        self.check_connected()
        self.check_target_safe(axis, target)

        self.controller.runtime.commands.motion.moveabsolute([axis], [target], [speed])
        self.controller.runtime.commands.motion.waitformotiondone([axis])
        self.log(f"Aerotech: moved {axis} to absolute {target:.6f} mm")

    def move_linear_absolute(self, axes, targets, speed):
        """
        V12B coordinated/simultaneous linear move.
        This avoids the old X-then-Y staircase path during circle writing.
        """
        self.check_connected()
        axes = list(axes)
        targets = [float(t) for t in targets]
        speed = float(speed)

        if len(axes) != len(targets):
            raise ValueError("axes and targets must have same length")

        for axis, target in zip(axes, targets):
            self.check_target_safe(axis, target)

        cmds = self.controller.runtime.commands.motion

        # Try common Automation1 Python signatures.
        try:
            cmds.movelinear(axes, targets, speed)
        except TypeError:
            try:
                cmds.movelinear(axes, targets, [speed] * len(axes))
            except TypeError:
                cmds.movelinear(axes, targets)

        cmds.waitformotiondone(axes)
        joined = ", ".join(f"{a}={t:.6f}" for a, t in zip(axes, targets))
        self.log(f"Aerotech V12B: coordinated XY/linear move to {joined} at {speed:.6f} mm/s")


# ============================================================
# 2) THORLABS WAVEPLATE / KDC101 DEVICE CLASS
# ============================================================

class ThorlabsWaveplate:
    def __init__(self, log_func):
        self.log = log_func
        self.lib = None
        self.connected = False

        self.kinesis_path = r"C:\Program Files\Thorlabs\Kinesis"
        self.serial_num = c_char_p(b"83846283")  # Change if needed
        self.steps_per_rev = c_double(1919.64186)

    def load_library(self):
        if self.lib is not None:
            return

        if not os.path.isdir(self.kinesis_path):
            raise RuntimeError(f"Kinesis path not found: {self.kinesis_path}")

        if sys.version_info >= (3, 8):
            os.add_dll_directory(self.kinesis_path)
        else:
            os.chdir(self.kinesis_path)

        dll_path = os.path.join(self.kinesis_path, "Thorlabs.MotionControl.KCube.DCServo.dll")
        self.lib = cdll.LoadLibrary(dll_path)
        self.log("Waveplate: Kinesis DLL loaded")

    def connect(self):
        self.load_library()
        self.lib.TLI_BuildDeviceList()

        result = self.lib.CC_Open(self.serial_num)
        if result != 0:
            raise RuntimeError(f"Waveplate: could not open device. CC_Open returned {result}")

        self.lib.CC_StartPolling(self.serial_num, c_int(200))
        time.sleep(0.5)
        self.lib.CC_SetMotorParamsExt(self.serial_num, self.steps_per_rev, c_double(1.0), c_double(1.0))
        self.connected = True
        self.log("Waveplate: connected and polling started")

    def check_connected(self):
        if not self.connected or self.lib is None:
            raise RuntimeError("Waveplate is not connected.")

    def home(self):
        self.check_connected()
        self.lib.CC_Home(self.serial_num)
        self.log("Waveplate: homing command sent. Wait until homing finishes before moving.")

    def move_to_angle(self, angle_deg):
        self.check_connected()

        target_pos_dev = c_int()
        self.lib.CC_GetDeviceUnitFromRealValue(
            self.serial_num,
            c_double(float(angle_deg)),
            byref(target_pos_dev),
            0
        )

        self.lib.CC_SetMoveAbsolutePosition(self.serial_num, target_pos_dev)
        self.lib.CC_MoveAbsolute(self.serial_num)
        self.log(f"Waveplate: moving to {angle_deg:.3f} degrees")

    def stop(self):
        self.check_connected()
        self.lib.CC_StopImmediate(self.serial_num)
        self.log("Waveplate: STOP sent")

    def close(self):
        if self.connected and self.lib is not None:
            try:
                self.lib.CC_StopPolling(self.serial_num)
                self.lib.CC_Close(self.serial_num)
                self.log("Waveplate: closed")
            except Exception as e:
                self.log(f"Waveplate close warning: {e}")
        self.connected = False


# ============================================================
# 3) THORLABS SC10 SHUTTER DEVICE CLASS
# ============================================================

class SC10Shutter:
    """
    Basic serial interface shell.

    Before using this with a real laser:
    - Confirm the SC10 command set from the manual.
    - Confirm COM port in Windows Device Manager.
    - Test with no laser/sample first.

    Many SC10 setups use serial/RS232 or TTL triggering. This class is intentionally conservative.
    """

    def __init__(self, log_func):
        self.log = log_func
        self.ser = None

    def connect(self, port, baudrate=9600):
        if serial is None:
            raise RuntimeError("pyserial is not installed. Install with: pip install pyserial")

        self.ser = serial.Serial(
            port=port,
            baudrate=int(baudrate),
            bytesize=8,
            parity="N",
            stopbits=1,
            timeout=1
        )
        self.log(f"Shutter SC10: connected on {port} at {baudrate} baud")
        self.log("Shutter SC10: keep shutter CLOSED until command set is verified.")

    def check_connected(self):
        if self.ser is None or not self.ser.is_open:
            raise RuntimeError("SC10 shutter is not connected.")

    def send_raw(self, cmd):
        self.check_connected()
        if not cmd.endswith("\r"):
            cmd = cmd + "\r"
        self.ser.write(cmd.encode("ascii"))
        time.sleep(0.1)
        reply = self.ser.read_all().decode(errors="ignore").strip()
        self.log(f"SC10 raw command sent: {cmd.strip()} | reply: {reply}")
        return reply

    def open_manual_command(self, open_cmd):
        self.send_raw(open_cmd)
        self.log("SC10: OPEN command sent")

    def close_manual_command(self, close_cmd):
        self.send_raw(close_cmd)
        self.log("SC10: CLOSE command sent")

    def pulse_manual(self, open_cmd, close_cmd, exposure_ms):
        self.close_manual_command(close_cmd)
        time.sleep(0.1)
        self.open_manual_command(open_cmd)
        time.sleep(float(exposure_ms) / 1000.0)
        self.close_manual_command(close_cmd)
        self.log(f"SC10: pulse done for {exposure_ms} ms")

    def close(self):
        if self.ser is not None and self.ser.is_open:
            self.ser.close()
            self.log("SC10: serial port closed")


# ============================================================
# 4) CAMERA PLACEHOLDER CLASS
# ============================================================

class CameraManual:
    """
    Version 3A camera control using pylablib + uc480.

    This uses the same working camera interface as your previous Z-scan code:
        from pylablib.devices import uc480
        camera = uc480.UC480Camera()
        camera.start_acquisition()
        img = camera.snap()
    """

    def __init__(self, log_func):
        self.log = log_func
        self.camera = None
        self.connected = False

    def connect_camera(self):
        if uc480 is None:
            raise RuntimeError(
                "pylablib uc480 is not available in this Python environment. "
                "Use the same Anaconda/Spyder environment where your old camera code worked."
            )

        cams = uc480.list_cameras()
        self.log(f"Camera: cameras found: {cams}")

        if not cams:
            raise RuntimeError("No uc480/Thorlabs camera found.")

        self.camera = uc480.UC480Camera()
        self.camera.start_acquisition()
        self.connected = True
        self.log("Camera: connected and acquisition started")

    def check_connected(self):
        if self.camera is None or not self.connected:
            raise RuntimeError("Camera is not connected.")

    def set_exposure_ms(self, exposure_ms):
        self.check_connected()
        try:
            self.camera.set_exposure(float(exposure_ms) / 1000.0)
            self.log(f"Camera: exposure set to {exposure_ms} ms")
        except Exception as e:
            self.log(f"Camera exposure warning: {e}")

    def capture_array(self):
        self.check_connected()
        img = self.camera.snap()

        if isinstance(img, tuple):
            img = img[0]

        return np.asarray(img)

    def capture_image(self, save_folder, filename_prefix="capture", exposure_ms=None):
        self.check_connected()
        os.makedirs(save_folder, exist_ok=True)

        if exposure_ms is not None:
            self.set_exposure_ms(exposure_ms)

        img = self.capture_array()

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        filename = os.path.join(save_folder, f"{filename_prefix}_{timestamp}.png")

        imageio.imwrite(filename, img)

        self.log(
            f"Camera: image saved: {filename} | shape={img.shape} | "
            f"min={np.min(img)} max={np.max(img)} mean={np.mean(img):.3f}"
        )

        return filename

    def open_thorcam(self, exe_path):
        if not exe_path or not os.path.isfile(exe_path):
            raise RuntimeError("ThorCam executable path is invalid.")
        subprocess.Popen([exe_path])
        self.log("Camera: ThorCam opened")

    def manual_capture_note(self):
        self.log("Camera: V3A uses pylablib/uc480 for real capture. Use Connect Camera before V2 sequence.")

    def close(self):
        try:
            if self.camera is not None:
                self.camera.close()
                self.log("Camera: closed")
        except Exception as e:
            self.log(f"Camera close warning: {e}")
        self.camera = None
        self.connected = False


# ============================================================
# 5) MAIN GUI
# ============================================================

class MantisMainGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("MANTIS Automation - Main GUI V12B Reference HWP Lead-In + Tangent Circle")
        try:
            self.root.state("zoomed")
        except Exception:
            self.root.geometry("1400x900")

        self.log_queue = queue.Queue()

        self.stage = AerotechStage(self.log)
        self.waveplate = ThorlabsWaveplate(self.log)
        self.shutter = SC10Shutter(self.log)
        self.camera = CameraManual(self.log)

        self.build_gui()
        self.create_external_log_window()
        self.process_log_queue()
        self.auto_update_stage_positions()

    def create_external_log_window(self):
        """
        Opens a separate always-visible experiment log window.
        This fixes the issue where the log is hidden at the bottom on small screens.
        """
        self.log_win = tk.Toplevel(self.root)
        self.log_win.title("MANTIS Experiment Log - Always Visible")
        self.log_win.geometry("900x350+50+50")
        self.log_win.attributes("-topmost", True)

        tk.Label(
            self.log_win,
            text="Experiment Log",
            font=("Arial", 13, "bold")
        ).pack(anchor="w", padx=8, pady=4)

        self.external_log_box = tk.Text(self.log_win, height=16, width=120)
        self.external_log_box.pack(fill=tk.BOTH, expand=True, padx=8, pady=4)

        tk.Button(
            self.log_win,
            text="Hide Topmost / Keep Window",
            command=lambda: self.log_win.attributes("-topmost", False)
        ).pack(pady=4)

    def build_gui(self):
        title = tk.Label(
            self.root,
            text="MANTIS Main GUI V12B - Reference HWP Lead-In + Tangent Circle",
            font=("Arial", 18, "bold")
        )
        title.pack(pady=8)

        main = tk.Frame(self.root)
        main.pack(fill=tk.BOTH, expand=True, padx=10)

        left = tk.Frame(main)
        left.grid(row=0, column=0, sticky="n")

        right = tk.Frame(main)
        right.grid(row=0, column=1, sticky="n", padx=15)

        # ---------------- Stage frame ----------------
        stage_frame = tk.LabelFrame(left, text="1) Aerotech XYZ Stage", font=("Arial", 12, "bold"), padx=8, pady=8)
        stage_frame.grid(row=0, column=0, sticky="we", pady=5)

        tk.Button(stage_frame, text="Connect Stage", width=18, command=self.safe_stage_connect).grid(row=0, column=0, padx=4, pady=3)
        tk.Button(stage_frame, text="Enable XYZ", width=18, command=self.safe_stage_enable).grid(row=0, column=1, padx=4, pady=3)
        tk.Button(stage_frame, text="Clear Faults", width=18, command=self.safe_stage_clear_faults).grid(row=0, column=2, padx=4, pady=3)
        tk.Button(stage_frame, text="STAGE STOP", width=18, bg="red", fg="white", command=self.safe_stage_stop).grid(row=0, column=3, padx=4, pady=3)

        tk.Label(stage_frame, text="Step (mm):").grid(row=1, column=0)
        self.stage_step_entry = tk.Entry(stage_frame, width=10)
        self.stage_step_entry.insert(0, "0.01")
        self.stage_step_entry.grid(row=1, column=1)

        tk.Label(stage_frame, text="Speed (mm/s):").grid(row=1, column=2)
        self.stage_speed_entry = tk.Entry(stage_frame, width=10)
        self.stage_speed_entry.insert(0, "0.2")
        self.stage_speed_entry.grid(row=1, column=3)

        self.stage_pos_labels = {}
        for i, axis in enumerate(["X", "Y", "Z"]):
            self.stage_pos_labels[axis] = tk.Label(stage_frame, text=f"{axis} = --- mm", font=("Arial", 11))
            self.stage_pos_labels[axis].grid(row=2 + i, column=0, columnspan=2, sticky="w")

            tk.Button(stage_frame, text=f"{axis} -", width=10, command=lambda a=axis: self.safe_stage_jog(a, -1)).grid(row=2 + i, column=2)
            tk.Button(stage_frame, text=f"{axis} +", width=10, command=lambda a=axis: self.safe_stage_jog(a, 1)).grid(row=2 + i, column=3)

        tk.Label(stage_frame, text="Absolute targets (mm):").grid(row=5, column=0, sticky="w", pady=(8, 0))
        self.abs_entries = {}
        for i, axis in enumerate(["X", "Y", "Z"]):
            tk.Label(stage_frame, text=f"{axis}:").grid(row=6 + i, column=0)
            ent = tk.Entry(stage_frame, width=10)
            ent.grid(row=6 + i, column=1)
            self.abs_entries[axis] = ent
            tk.Button(stage_frame, text=f"Move {axis}", width=10, command=lambda a=axis: self.safe_stage_abs(a)).grid(row=6 + i, column=2)

        tk.Button(stage_frame, text="Read Stage Position", width=20, command=self.update_stage_positions).grid(row=9, column=0, columnspan=2, pady=5)

        # ---------------- Waveplate frame ----------------
        wp_frame = tk.LabelFrame(left, text="2) Thorlabs HWP Rotation Stage", font=("Arial", 12, "bold"), padx=8, pady=8)
        wp_frame.grid(row=1, column=0, sticky="we", pady=5)

        tk.Button(wp_frame, text="Connect HWP", width=18, command=self.safe_wp_connect).grid(row=0, column=0, padx=4, pady=3)
        tk.Button(wp_frame, text="Home HWP", width=18, command=self.safe_wp_home).grid(row=0, column=1, padx=4, pady=3)
        tk.Button(wp_frame, text="HWP STOP", width=18, bg="orange", command=self.safe_wp_stop).grid(row=0, column=2, padx=4, pady=3)

        tk.Label(wp_frame, text="Angle (deg):").grid(row=1, column=0)
        self.wp_angle_entry = tk.Entry(wp_frame, width=12)
        self.wp_angle_entry.insert(0, "0.0")
        self.wp_angle_entry.grid(row=1, column=1)
        tk.Button(wp_frame, text="Move HWP", width=18, command=self.safe_wp_move).grid(row=1, column=2, padx=4, pady=3)

        # ---------------- Shutter frame ----------------
        sh_frame = tk.LabelFrame(right, text="3) Thorlabs SC10 Shutter", font=("Arial", 12, "bold"), padx=8, pady=8)
        sh_frame.grid(row=0, column=0, sticky="we", pady=5)

        tk.Label(sh_frame, text="COM port:").grid(row=0, column=0)
        self.shutter_port_entry = tk.Entry(sh_frame, width=10)
        self.shutter_port_entry.insert(0, "COM3")
        self.shutter_port_entry.grid(row=0, column=1)

        tk.Label(sh_frame, text="Baud:").grid(row=0, column=2)
        self.shutter_baud_entry = tk.Entry(sh_frame, width=10)
        self.shutter_baud_entry.insert(0, "9600")
        self.shutter_baud_entry.grid(row=0, column=3)

        tk.Button(sh_frame, text="Connect Shutter", width=18, command=self.safe_shutter_connect).grid(row=0, column=4, padx=4)

        tk.Label(sh_frame, text="OPEN cmd:").grid(row=1, column=0)
        self.shutter_open_cmd_entry = tk.Entry(sh_frame, width=10)
        self.shutter_open_cmd_entry.insert(0, "ens")
        self.shutter_open_cmd_entry.grid(row=1, column=1)

        tk.Label(sh_frame, text="CLOSE cmd:").grid(row=1, column=2)
        self.shutter_close_cmd_entry = tk.Entry(sh_frame, width=10)
        self.shutter_close_cmd_entry.insert(0, "ens")
        self.shutter_close_cmd_entry.grid(row=1, column=3)

        tk.Button(sh_frame, text="Send OPEN", width=18, command=self.safe_shutter_open).grid(row=2, column=0, columnspan=2, pady=4)
        tk.Button(sh_frame, text="Send CLOSE", width=18, command=self.safe_shutter_close).grid(row=2, column=2, columnspan=2, pady=4)

        tk.Label(sh_frame, text="Pulse time (ms):").grid(row=3, column=0)
        self.shutter_pulse_entry = tk.Entry(sh_frame, width=10)
        self.shutter_pulse_entry.insert(0, "100")
        self.shutter_pulse_entry.grid(row=3, column=1)
        tk.Button(sh_frame, text="Pulse", width=18, command=self.safe_shutter_pulse).grid(row=3, column=2, columnspan=2, pady=4)

        tk.Label(
            sh_frame,
            text="SC10 commands: ens toggles enable; use ens?, closed?, id? for status queries.",
            fg="red"
        ).grid(row=4, column=0, columnspan=5, sticky="w")

        # ---------------- Camera frame ----------------
        cam_frame = tk.LabelFrame(right, text="4) Camera / ThorCam", font=("Arial", 12, "bold"), padx=8, pady=8)
        cam_frame.grid(row=1, column=0, sticky="we", pady=5)

        tk.Label(cam_frame, text="ThorCam exe:").grid(row=0, column=0)
        self.thorcam_path_entry = tk.Entry(cam_frame, width=55)
        self.thorcam_path_entry.insert(0, r"C:\Program Files\Thorlabs\Scientific Imaging\ThorCam\ThorCam.exe")
        self.thorcam_path_entry.grid(row=0, column=1, columnspan=3)

        tk.Button(cam_frame, text="Browse", width=12, command=self.browse_thorcam).grid(row=0, column=4, padx=3)
        tk.Button(cam_frame, text="Connect Camera", width=18, command=self.safe_connect_camera).grid(row=1, column=0, pady=4)
        tk.Button(cam_frame, text="Save Test Image", width=18, command=self.safe_save_test_image).grid(row=1, column=1, pady=4)
        tk.Button(cam_frame, text="Open ThorCam", width=18, command=self.safe_open_thorcam).grid(row=1, column=2, pady=4)
        tk.Button(cam_frame, text="Camera Note", width=18, command=self.camera.manual_capture_note).grid(row=1, column=3, pady=4)

        # ---------------- Manual sequence frame ----------------
        seq_frame = tk.LabelFrame(right, text="5) Manual Test Sequence Checklist", font=("Arial", 12, "bold"), padx=8, pady=8)
        seq_frame.grid(row=2, column=0, sticky="we", pady=5)

        checklist = (
            "Recommended V1 test:\n"
            "1. Connect Stage → Enable XYZ → move X/Y by small step.\n"
            "2. Connect HWP → Home → move to 0°, 10°, 20°.\n"
            "3. Keep shutter CLOSED. Verify SC10 command set before laser tests.\n"
            "4. Connect Camera, then Save Test Image.\n"
            "5. After all devices work alone, test: move stage → move HWP → shutter pulse → camera image."
        )
        tk.Label(seq_frame, text=checklist, justify="left").pack(anchor="w")

        # ---------------- V2 synchronized sequence frame ----------------
        v2_frame = tk.LabelFrame(right, text="6) V2 Synchronized Test Sequence", font=("Arial", 12, "bold"), padx=8, pady=8)
        v2_frame.grid(row=3, column=0, sticky="we", pady=5)

        tk.Label(v2_frame, text="X target (mm):").grid(row=0, column=0, sticky="w")
        self.seq_x_entry = tk.Entry(v2_frame, width=10)
        self.seq_x_entry.insert(0, "0.0")
        self.seq_x_entry.grid(row=0, column=1)

        tk.Label(v2_frame, text="Y target (mm):").grid(row=0, column=2, sticky="w")
        self.seq_y_entry = tk.Entry(v2_frame, width=10)
        self.seq_y_entry.insert(0, "0.0")
        self.seq_y_entry.grid(row=0, column=3)

        tk.Label(v2_frame, text="Z target (mm):").grid(row=1, column=0, sticky="w")
        self.seq_z_entry = tk.Entry(v2_frame, width=10)
        self.seq_z_entry.insert(0, "0.0")
        self.seq_z_entry.grid(row=1, column=1)

        tk.Label(v2_frame, text="Reference/Fixed Lead-in reference HWP angle (deg):").grid(row=1, column=2, sticky="w")
        self.seq_hwp_entry = tk.Entry(v2_frame, width=10)
        self.seq_hwp_entry.insert(0, "0.0")
        self.seq_hwp_entry.grid(row=1, column=3)

        tk.Label(v2_frame, text="Exposure (ms):").grid(row=2, column=0, sticky="w")
        self.seq_exp_entry = tk.Entry(v2_frame, width=10)
        self.seq_exp_entry.insert(0, "100")
        self.seq_exp_entry.grid(row=2, column=1)

        tk.Label(v2_frame, text="Speed (mm/s):").grid(row=2, column=2, sticky="w")
        self.seq_speed_entry = tk.Entry(v2_frame, width=10)
        self.seq_speed_entry.insert(0, "0.2")
        self.seq_speed_entry.grid(row=2, column=3)

        tk.Label(v2_frame, text="Image/log folder:").grid(row=3, column=0, sticky="w")
        self.seq_folder_entry = tk.Entry(v2_frame, width=45)
        self.seq_folder_entry.insert(0, os.path.join(os.getcwd(), "MANTIS_V2_Data"))
        self.seq_folder_entry.grid(row=3, column=1, columnspan=3, sticky="we")
        tk.Button(v2_frame, text="Browse", width=10, command=self.browse_seq_folder).grid(row=3, column=4, padx=3)

        tk.Button(v2_frame, text="RUN V2 SEQUENCE", width=22, bg="lightgreen",
                  command=self.safe_run_v2_sequence).grid(row=4, column=0, columnspan=2, pady=6)

        tk.Button(v2_frame, text="Close Shutter Now", width=22, bg="orange",
                  command=self.safe_v2_close_shutter).grid(row=4, column=2, columnspan=2, pady=6)

        tk.Label(
            v2_frame,
            text="Sequence: close shutter → move HWP → move stage → open shutter → wait → close shutter → camera capture → save log",
            fg="blue",
            wraplength=600,
            justify="left"
        ).grid(row=5, column=0, columnspan=5, sticky="w")

        # ---------------- V3B multi-point line scan frame ----------------
        v3b_frame = tk.LabelFrame(right, text="7) V3B Multi-Point Line Scan", font=("Arial", 12, "bold"), padx=8, pady=8)
        v3b_frame.grid(row=4, column=0, sticky="we", pady=5)

        tk.Label(v3b_frame, text="Start X (mm):").grid(row=0, column=0, sticky="w")
        self.line_x0_entry = tk.Entry(v3b_frame, width=10)
        self.line_x0_entry.insert(0, "0.0")
        self.line_x0_entry.grid(row=0, column=1)

        tk.Label(v3b_frame, text="Start Y (mm):").grid(row=0, column=2, sticky="w")
        self.line_y0_entry = tk.Entry(v3b_frame, width=10)
        self.line_y0_entry.insert(0, "0.0")
        self.line_y0_entry.grid(row=0, column=3)

        tk.Label(v3b_frame, text="Z (mm):").grid(row=1, column=0, sticky="w")
        self.line_z_entry = tk.Entry(v3b_frame, width=10)
        self.line_z_entry.insert(0, "0.0")
        self.line_z_entry.grid(row=1, column=1)

        tk.Label(v3b_frame, text="Lead-in reference HWP angle (deg):").grid(row=1, column=2, sticky="w")
        self.line_hwp_entry = tk.Entry(v3b_frame, width=10)
        self.line_hwp_entry.insert(0, "0.0")
        self.line_hwp_entry.grid(row=1, column=3)

        tk.Label(v3b_frame, text="Direction:").grid(row=2, column=0, sticky="w")
        self.line_direction_var = tk.StringVar(value="X")
        tk.OptionMenu(v3b_frame, self.line_direction_var, "X", "Y", "Diagonal").grid(row=2, column=1, sticky="we")

        tk.Label(v3b_frame, text="Step (mm):").grid(row=2, column=2, sticky="w")
        self.line_step_entry = tk.Entry(v3b_frame, width=10)
        self.line_step_entry.insert(0, "0.01")
        self.line_step_entry.grid(row=2, column=3)

        tk.Label(v3b_frame, text="Number of points:").grid(row=3, column=0, sticky="w")
        self.line_n_entry = tk.Entry(v3b_frame, width=10)
        self.line_n_entry.insert(0, "5")
        self.line_n_entry.grid(row=3, column=1)

        tk.Label(v3b_frame, text="Exposure (ms):").grid(row=3, column=2, sticky="w")
        self.line_exp_entry = tk.Entry(v3b_frame, width=10)
        self.line_exp_entry.insert(0, "100")
        self.line_exp_entry.grid(row=3, column=3)

        tk.Label(v3b_frame, text="Speed (mm/s):").grid(row=4, column=0, sticky="w")
        self.line_speed_entry = tk.Entry(v3b_frame, width=10)
        self.line_speed_entry.insert(0, "0.2")
        self.line_speed_entry.grid(row=4, column=1)

        tk.Label(v3b_frame, text="Wait after move (s):").grid(row=4, column=2, sticky="w")
        self.line_wait_entry = tk.Entry(v3b_frame, width=10)
        self.line_wait_entry.insert(0, "0.2")
        self.line_wait_entry.grid(row=4, column=3)

        tk.Button(v3b_frame, text="RUN LINE SCAN", width=22, bg="lightblue",
                  command=self.safe_run_line_scan).grid(row=5, column=0, columnspan=2, pady=6)

        tk.Button(v3b_frame, text="STOP / Close Shutter", width=22, bg="orange",
                  command=self.safe_v2_close_shutter).grid(row=5, column=2, columnspan=2, pady=6)

        tk.Label(
            v3b_frame,
            text="Line scan: each point → close shutter → move HWP → move stage → open shutter → wait → close shutter → capture image → log.",
            fg="blue",
            wraplength=600,
            justify="left"
        ).grid(row=6, column=0, columnspan=5, sticky="w")

        # ---------------- V4 shape writing test frame ----------------
        v4_frame = tk.LabelFrame(right, text="8) V4 Shape Writing Test", font=("Arial", 12, "bold"), padx=8, pady=8)
        v4_frame.grid(row=5, column=0, sticky="we", pady=5)

        tk.Label(v4_frame, text="Shape:").grid(row=0, column=0, sticky="w")
        self.shape_type_var = tk.StringVar(value="Rectangle")
        tk.OptionMenu(v4_frame, self.shape_type_var, "Line", "Rectangle", "Circle", "Grid").grid(row=0, column=1, sticky="we")

        tk.Label(v4_frame, text="Center/Start X (mm):").grid(row=0, column=2, sticky="w")
        self.shape_x_entry = tk.Entry(v4_frame, width=10)
        self.shape_x_entry.insert(0, "0.0")
        self.shape_x_entry.grid(row=0, column=3)

        tk.Label(v4_frame, text="Center/Start Y (mm):").grid(row=1, column=0, sticky="w")
        self.shape_y_entry = tk.Entry(v4_frame, width=10)
        self.shape_y_entry.insert(0, "0.0")
        self.shape_y_entry.grid(row=1, column=1)

        tk.Label(v4_frame, text="Z (mm):").grid(row=1, column=2, sticky="w")
        self.shape_z_entry = tk.Entry(v4_frame, width=10)
        self.shape_z_entry.insert(0, "0.0")
        self.shape_z_entry.grid(row=1, column=3)

        tk.Label(v4_frame, text="Width/Line length (mm):").grid(row=2, column=0, sticky="w")
        self.shape_width_entry = tk.Entry(v4_frame, width=10)
        self.shape_width_entry.insert(0, "0.04")
        self.shape_width_entry.grid(row=2, column=1)

        tk.Label(v4_frame, text="Height (mm):").grid(row=2, column=2, sticky="w")
        self.shape_height_entry = tk.Entry(v4_frame, width=10)
        self.shape_height_entry.insert(0, "0.04")
        self.shape_height_entry.grid(row=2, column=3)

        tk.Label(v4_frame, text="Radius (mm):").grid(row=3, column=0, sticky="w")
        self.shape_radius_entry = tk.Entry(v4_frame, width=10)
        self.shape_radius_entry.insert(0, "0.02")
        self.shape_radius_entry.grid(row=3, column=1)

        tk.Label(v4_frame, text="Step (mm):").grid(row=3, column=2, sticky="w")
        self.shape_step_entry = tk.Entry(v4_frame, width=10)
        self.shape_step_entry.insert(0, "0.01")
        self.shape_step_entry.grid(row=3, column=3)

        tk.Label(v4_frame, text="Lead-in reference HWP angle (deg):").grid(row=4, column=0, sticky="w")
        self.shape_hwp_entry = tk.Entry(v4_frame, width=10)
        self.shape_hwp_entry.insert(0, "0.0")
        self.shape_hwp_entry.grid(row=4, column=1)

        tk.Label(v4_frame, text="Exposure (ms):").grid(row=4, column=2, sticky="w")
        self.shape_exp_entry = tk.Entry(v4_frame, width=10)
        self.shape_exp_entry.insert(0, "100")
        self.shape_exp_entry.grid(row=4, column=3)

        # V5 automatic polarization / HWP controls
        self.auto_hwp_var = tk.BooleanVar(value=False)
        tk.Checkbutton(
            v4_frame,
            text="V5 Auto HWP: polarization perpendicular to motion",
            variable=self.auto_hwp_var
        ).grid(row=5, column=0, columnspan=4, sticky="w")

        tk.Label(v4_frame, text="Offset calibration (ignored for circle):").grid(row=6, column=0, sticky="w")
        self.hwp_offset_entry = tk.Entry(v4_frame, width=10)
        self.hwp_offset_entry.insert(0, "0.0")
        self.hwp_offset_entry.grid(row=6, column=1)

        tk.Label(v4_frame, text="HWP settle time (s):").grid(row=6, column=2, sticky="w")
        self.hwp_settle_entry = tk.Entry(v4_frame, width=10)
        self.hwp_settle_entry.insert(0, "0.3")
        self.hwp_settle_entry.grid(row=6, column=3)

        # V8 continuous circle controls
        self.continuous_circle_var = tk.BooleanVar(value=True)
        tk.Checkbutton(
            v4_frame,
            text="V12B Continuous Circle: open shutter once, write full circle, close at end",
            variable=self.continuous_circle_var
        ).grid(row=7, column=0, columnspan=4, sticky="w")

        tk.Label(v4_frame, text="Circle HWP update wait (s):").grid(row=8, column=0, sticky="w")
        self.circle_hwp_wait_entry = tk.Entry(v4_frame, width=10)
        self.circle_hwp_wait_entry.insert(0, "0.0")
        self.circle_hwp_wait_entry.grid(row=8, column=1)

        self.capture_after_circle_var = tk.BooleanVar(value=True)
        tk.Checkbutton(
            v4_frame,
            text="Capture image after continuous circle",
            variable=self.capture_after_circle_var
        ).grid(row=8, column=2, columnspan=2, sticky="w")

        self.closed_circle_var = tk.BooleanVar(value=True)
        tk.Checkbutton(
            v4_frame,
            text="Closed circle: return exactly to start point",
            variable=self.closed_circle_var
        ).grid(row=9, column=0, columnspan=4, sticky="w")

        tk.Label(v4_frame, text="Lead-in length (mm):").grid(row=10, column=0, sticky="w")
        self.lead_in_entry = tk.Entry(v4_frame, width=10)
        self.lead_in_entry.insert(0, "0.005")
        self.lead_in_entry.grid(row=10, column=1)

        tk.Label(v4_frame, text="Lead-out length (mm):").grid(row=10, column=2, sticky="w")
        self.lead_out_entry = tk.Entry(v4_frame, width=10)
        self.lead_out_entry.insert(0, "0.005")
        self.lead_out_entry.grid(row=10, column=3)

        tk.Label(v4_frame, text="Speed (mm/s):").grid(row=11, column=0, sticky="w")
        self.shape_speed_entry = tk.Entry(v4_frame, width=10)
        self.shape_speed_entry.insert(0, "0.2")
        self.shape_speed_entry.grid(row=11, column=1)

        tk.Label(v4_frame, text="Wait after move (s):").grid(row=11, column=2, sticky="w")
        self.shape_wait_entry = tk.Entry(v4_frame, width=10)
        self.shape_wait_entry.insert(0, "0.2")
        self.shape_wait_entry.grid(row=11, column=3)

        tk.Button(v4_frame, text="PREVIEW SHAPE", width=22, bg="lightblue",
                  command=self.preview_shape_v6).grid(row=12, column=0, columnspan=1, pady=6)

        tk.Button(v4_frame, text="RUN SHAPE TEST", width=22, bg="lightgreen",
                  command=self.safe_run_shape_test).grid(row=12, column=1, columnspan=1, pady=6)

        tk.Button(v4_frame, text="STOP / Close Shutter", width=22, bg="orange",
                  command=self.safe_v2_close_shutter).grid(row=12, column=2, columnspan=2, pady=6)

        tk.Label(
            v4_frame,
            text="V12B: Circle uses coordinated XY moves plus lead-in/lead-out. Shutter opens once before lead-in and closes after lead-out.",
            fg="blue",
            wraplength=620,
            justify="left"
        ).grid(row=13, column=0, columnspan=5, sticky="w")

        # ---------------- Log frame ----------------
        log_frame = tk.LabelFrame(self.root, text="Experiment Log", font=("Arial", 12, "bold"), padx=8, pady=8)
        log_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=8)

        self.log_box = tk.Text(log_frame, height=12, width=135)
        self.log_box.pack(fill=tk.BOTH, expand=True)

        bottom = tk.Frame(self.root)
        bottom.pack(fill=tk.X, padx=10, pady=4)

        tk.Button(bottom, text="Open / Reopen Experiment Log Window",
                  bg="lightblue", font=("Arial", 10, "bold"),
                  command=self.create_external_log_window).pack(fill=tk.X, pady=2)

        tk.Button(bottom, text="GLOBAL SAFE STOP: close shutter command + stop stage + stop HWP",
                  bg="red", fg="white", font=("Arial", 11, "bold"),
                  command=self.global_safe_stop).pack(fill=tk.X)

    # ---------------- Logging ----------------

    def log(self, message):
        timestamp = time.strftime("%H:%M:%S")
        self.log_queue.put(f"[{timestamp}] {message}")

    def process_log_queue(self):
        while not self.log_queue.empty():
            msg = self.log_queue.get()
            try:
                self.log_box.insert(tk.END, msg + "\n")
                self.log_box.see(tk.END)
            except Exception:
                pass

            try:
                self.external_log_box.insert(tk.END, msg + "\n")
                self.external_log_box.see(tk.END)
            except Exception:
                pass
        self.root.after(100, self.process_log_queue)

    def run_threaded(self, func, error_title="Error"):
        def worker():
            try:
                func()
            except Exception as e:
                self.log(f"{error_title}: {e}")
                self.root.after(0, lambda: messagebox.showerror(error_title, str(e)))
        threading.Thread(target=worker, daemon=True).start()

    # ---------------- Stage GUI callbacks ----------------

    def safe_stage_connect(self):
        self.run_threaded(self.stage.connect, "Stage connection error")

    def safe_stage_enable(self):
        self.run_threaded(self.stage.enable_all, "Stage enable error")

    def safe_stage_clear_faults(self):
        self.run_threaded(self.stage.clear_faults, "Stage clear fault error")

    def safe_stage_stop(self):
        self.run_threaded(self.stage.stop, "Stage stop error")

    def get_stage_step_speed(self):
        step = float(self.stage_step_entry.get())
        speed = float(self.stage_speed_entry.get())
        if step <= 0 or speed <= 0:
            raise ValueError("Step and speed must be positive.")
        return step, speed

    def safe_stage_jog(self, axis, direction):
        def job():
            step, speed = self.get_stage_step_speed()
            if axis == "Z" and step > 0.01:
                ok = messagebox.askyesno("Z Safety", f"Z step is {step} mm. Continue?")
                if not ok:
                    self.log("Z jog cancelled")
                    return
            self.stage.move_relative(axis, direction * step, speed)
            self.update_stage_positions()
        self.run_threaded(job, "Stage jog error")

    def safe_stage_abs(self, axis):
        def job():
            target = float(self.abs_entries[axis].get())
            speed = float(self.stage_speed_entry.get())

            if axis == "Z":
                ok = messagebox.askyesno("Z Safety", f"Move Z to {target} mm? Continue?")
                if not ok:
                    self.log("Z absolute move cancelled")
                    return

            self.stage.move_absolute(axis, target, speed)
            self.update_stage_positions()
        self.run_threaded(job, "Stage absolute move error")

    def update_stage_positions(self):
        try:
            data = self.stage.get_axis_data()
            for axis, d in data.items():
                status = "FAULT" if d["faulted"] else f"Enabled={d['enabled']} Homed={d['homed']}"
                self.stage_pos_labels[axis].config(
                    text=f"{axis} = {d['position']:.6f} mm | {status}"
                )
        except Exception as e:
            self.log(f"Stage status read: {e}")

    def auto_update_stage_positions(self):
        if self.stage.controller is not None:
            self.update_stage_positions()
        self.root.after(1500, self.auto_update_stage_positions)

    # ---------------- Waveplate GUI callbacks ----------------

    def safe_wp_connect(self):
        self.run_threaded(self.waveplate.connect, "Waveplate connection error")

    def safe_wp_home(self):
        self.run_threaded(self.waveplate.home, "Waveplate home error")

    def safe_wp_move(self):
        def job():
            angle = float(self.wp_angle_entry.get())
            self.waveplate.move_to_angle(angle)
        self.run_threaded(job, "Waveplate move error")

    def safe_wp_stop(self):
        self.run_threaded(self.waveplate.stop, "Waveplate stop error")

    # ---------------- Shutter GUI callbacks ----------------

    def safe_shutter_connect(self):
        def job():
            port = self.shutter_port_entry.get().strip()
            baud = int(self.shutter_baud_entry.get())
            self.shutter.connect(port, baud)
        self.run_threaded(job, "Shutter connection error")

    def safe_shutter_open(self):
        def job():
            cmd = self.shutter_open_cmd_entry.get().strip()
            if "?" in cmd:
                raise RuntimeError("Please replace OPEN? with the real SC10 open command first.")
            self.shutter.open_manual_command(cmd)
        self.run_threaded(job, "Shutter open error")

    def safe_shutter_close(self):
        def job():
            cmd = self.shutter_close_cmd_entry.get().strip()
            if "?" in cmd:
                raise RuntimeError("Please replace CLOSE? with the real SC10 close command first.")
            self.shutter.close_manual_command(cmd)
        self.run_threaded(job, "Shutter close error")

    def safe_shutter_pulse(self):
        def job():
            open_cmd = self.shutter_open_cmd_entry.get().strip()
            close_cmd = self.shutter_close_cmd_entry.get().strip()
            exposure_ms = float(self.shutter_pulse_entry.get())
            if "?" in open_cmd or "?" in close_cmd:
                raise RuntimeError("Please replace OPEN?/CLOSE? with real SC10 commands first.")
            self.shutter.pulse_manual(open_cmd, close_cmd, exposure_ms)
        self.run_threaded(job, "Shutter pulse error")

    # ---------------- Camera GUI callbacks ----------------

    def safe_connect_camera(self):
        self.run_threaded(self.camera.connect_camera, "Camera connection error")

    def safe_save_test_image(self):
        def job():
            folder = self.seq_folder_entry.get().strip() if hasattr(self, "seq_folder_entry") else os.getcwd()
            exposure_ms = None
            if hasattr(self, "seq_exp_entry"):
                try:
                    exposure_ms = float(self.seq_exp_entry.get())
                except Exception:
                    exposure_ms = None
            self.camera.capture_image(folder, filename_prefix="manual_test_image", exposure_ms=exposure_ms)
        self.run_threaded(job, "Camera save image error")

    def browse_thorcam(self):
        path = filedialog.askopenfilename(
            title="Select ThorCam.exe",
            filetypes=[("Executable", "*.exe"), ("All files", "*.*")]
        )
        if path:
            self.thorcam_path_entry.delete(0, tk.END)
            self.thorcam_path_entry.insert(0, path)

    def safe_open_thorcam(self):
        def job():
            path = self.thorcam_path_entry.get().strip()
            self.camera.open_thorcam(path)
        self.run_threaded(job, "Camera error")


    # ---------------- V2 synchronized sequence callbacks ----------------

    def browse_seq_folder(self):
        folder = filedialog.askdirectory(title="Select V2 data folder")
        if folder:
            self.seq_folder_entry.delete(0, tk.END)
            self.seq_folder_entry.insert(0, folder)

    def _send_sc10_command_and_get_reply(self, cmd):
        self.shutter.check_connected()
        reply = self.shutter.send_raw(cmd)
        return reply

    def _get_sc10_enable_state(self):
        reply = self._send_sc10_command_and_get_reply("ens?")
        try:
            cleaned = "".join(ch for ch in reply if ch in "01")
            if cleaned:
                return int(cleaned[-1])
        except Exception:
            pass
        return None

    def _ensure_shutter_closed_v2(self):
        state = self._get_sc10_enable_state()
        self.log(f"V2: SC10 enable state before closing = {state}")
        if state == 1:
            self._send_sc10_command_and_get_reply("ens")
            time.sleep(0.2)
            self.log("V2: shutter close/disable command sent")
        elif state == 0:
            self.log("V2: shutter already disabled/closed")
        else:
            self.log("V2 warning: could not parse ens? state. Not toggling shutter automatically.")

    def _open_shutter_v2(self):
        state = self._get_sc10_enable_state()
        self.log(f"V2: SC10 enable state before opening = {state}")
        if state == 0:
            self._send_sc10_command_and_get_reply("ens")
            time.sleep(0.1)
            self.log("V2: shutter open/enable command sent")
        elif state == 1:
            self.log("V2: shutter already enabled/open")
        else:
            raise RuntimeError("Cannot determine shutter state from ens?. Aborting before exposure.")

    def _close_shutter_v2(self):
        state = self._get_sc10_enable_state()
        self.log(f"V2: SC10 enable state before final closing = {state}")
        if state == 1:
            self._send_sc10_command_and_get_reply("ens")
            time.sleep(0.1)
            self.log("V2: shutter close/disable command sent")
        elif state == 0:
            self.log("V2: shutter already disabled/closed")
        else:
            raise RuntimeError("Cannot determine shutter state from ens?. Please close shutter manually.")

    def safe_v2_close_shutter(self):
        self.run_threaded(self._close_shutter_v2, "V2 close shutter error")

    def safe_run_v2_sequence(self):
        answer = messagebox.askyesno(
            "Run V2 Sequence",
            "This will move HWP, move the stage, open the shutter for the exposure time, close it, and save a log.\n\n"
            "Confirm laser is blocked/safe for this first test.\n\nContinue?"
        )
        if not answer:
            self.log("V2 sequence cancelled by user")
            return
        self.run_threaded(self.run_v2_sequence, "V2 sequence error")

    def run_v2_sequence(self):
        x = float(self.seq_x_entry.get())
        y = float(self.seq_y_entry.get())
        z = float(self.seq_z_entry.get())
        hwp_angle = float(self.seq_hwp_entry.get())
        exposure_ms = float(self.seq_exp_entry.get())
        speed = float(self.seq_speed_entry.get())
        save_folder = self.seq_folder_entry.get().strip()

        if exposure_ms <= 0:
            raise ValueError("Exposure time must be positive.")
        if speed <= 0:
            raise ValueError("Speed must be positive.")

        os.makedirs(save_folder, exist_ok=True)

        self.log("========== V2 SYNCHRONIZED SEQUENCE START ==========")
        self.log(f"V2 targets: X={x} mm, Y={y} mm, Z={z} mm, HWP={hwp_angle} deg, exposure={exposure_ms} ms")

        self.log("V2 step 1: close shutter first")
        self._ensure_shutter_closed_v2()

        self.log("V2 step 2: move HWP")
        self.waveplate.move_to_angle(hwp_angle)
        time.sleep(2.0)

        self.log("V2 step 3: move stage X, Y, Z")
        self.stage.move_absolute("X", x, speed)
        self.stage.move_absolute("Y", y, speed)

        if z > 5.0:
            raise RuntimeError("Z target above temporary safety limit 5.0 mm.")
        self.stage.move_absolute("Z", z, speed)

        self.update_stage_positions()
        self.log("V2 step 4: stage motion complete")

        self.log("V2 step 5: open shutter")
        self._open_shutter_v2()

        self.log(f"V2 step 6: exposure wait {exposure_ms} ms")
        time.sleep(exposure_ms / 1000.0)

        self.log("V2 step 7: close shutter")
        self._close_shutter_v2()

        self.log("V2 step 8: camera capture")
        capture_file = self.camera.capture_image(save_folder, filename_prefix="v2_capture", exposure_ms=exposure_ms)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_file = os.path.join(save_folder, f"v2_sequence_log_{timestamp}.csv")
        with open(csv_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["timestamp", "x_mm", "y_mm", "z_mm", "hwp_deg", "exposure_ms", "speed_mm_s", "capture_file"])
            writer.writerow([timestamp, x, y, z, hwp_angle, exposure_ms, speed, capture_file])

        self.log(f"V2: sequence CSV log saved: {csv_file}")
        self.log("========== V2 SYNCHRONIZED SEQUENCE COMPLETE ==========")


    # ---------------- V3B multi-point line scan ----------------

    def _generate_line_points(self, x0, y0, z, step, n_points, direction):
        points = []
        for i in range(n_points):
            if direction == "X":
                x = x0 + i * step
                y = y0
            elif direction == "Y":
                x = x0
                y = y0 + i * step
            else:
                x = x0 + i * step
                y = y0 + i * step
            points.append((float(x), float(y), float(z)))
        return points

    def safe_run_line_scan(self):
        answer = messagebox.askyesno(
            "Run V3B Line Scan",
            "This will run repeated stage moves and repeated shutter exposures.\\n\\n"
            "For first test, keep laser blocked/safe and use small steps.\\n\\nContinue?"
        )
        if not answer:
            self.log("V3B line scan cancelled by user")
            return
        self.run_threaded(self.run_line_scan, "V3B line scan error")

    def run_line_scan(self):
        x0 = float(self.line_x0_entry.get())
        y0 = float(self.line_y0_entry.get())
        z = float(self.line_z_entry.get())
        hwp_angle = float(self.line_hwp_entry.get())
        step = float(self.line_step_entry.get())
        n_points = int(float(self.line_n_entry.get()))
        exposure_ms = float(self.line_exp_entry.get())
        speed = float(self.line_speed_entry.get())
        wait_after_move = float(self.line_wait_entry.get())
        direction = self.line_direction_var.get()
        save_folder = self.seq_folder_entry.get().strip()

        if n_points <= 0:
            raise ValueError("Number of points must be positive.")
        if step <= 0:
            raise ValueError("Step must be positive.")
        if exposure_ms <= 0:
            raise ValueError("Exposure time must be positive.")
        if speed <= 0:
            raise ValueError("Speed must be positive.")
        if z > 5.0:
            raise RuntimeError("Z target above temporary safety limit 5.0 mm.")

        os.makedirs(save_folder, exist_ok=True)
        scan_folder = os.path.join(save_folder, datetime.now().strftime("V3B_line_scan_%Y%m%d_%H%M%S"))
        os.makedirs(scan_folder, exist_ok=True)

        points = self._generate_line_points(x0, y0, z, step, n_points, direction)
        csv_file = os.path.join(scan_folder, "line_scan_log.csv")

        with open(csv_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "point_index", "timestamp", "x_mm", "y_mm", "z_mm",
                "hwp_deg", "exposure_ms", "speed_mm_s", "wait_after_move_s",
                "direction", "image_file"
            ])

        self.log("========== V3B LINE SCAN START ==========")
        self.log(f"V3B: folder = {scan_folder}")
        self.log(f"V3B: points = {n_points}, direction = {direction}, step = {step} mm")

        self._ensure_shutter_closed_v2()

        self.log(f"V3B: moving HWP to {hwp_angle} deg")
        self.waveplate.move_to_angle(hwp_angle)
        time.sleep(2.0)

        for idx, (x, y, zpos) in enumerate(points):
            self.log(f"--- V3B point {idx+1}/{n_points}: X={x:.6f}, Y={y:.6f}, Z={zpos:.6f} ---")

            self._ensure_shutter_closed_v2()

            self.stage.move_absolute("X", x, speed)
            self.stage.move_absolute("Y", y, speed)
            self.stage.move_absolute("Z", zpos, speed)
            self.update_stage_positions()
            self.log(f"V3B: stage reached point {idx+1}")

            if wait_after_move > 0:
                time.sleep(wait_after_move)

            self._open_shutter_v2()
            time.sleep(exposure_ms / 1000.0)
            self._close_shutter_v2()

            image_file = self.camera.capture_image(
                scan_folder,
                filename_prefix=f"point_{idx:04d}_X_{x:.6f}_Y_{y:.6f}_Z_{zpos:.6f}",
                exposure_ms=exposure_ms
            )

            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
            with open(csv_file, "a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([
                    idx, timestamp, x, y, zpos, hwp_angle,
                    exposure_ms, speed, wait_after_move, direction, image_file
                ])

            self.log(f"V3B: point {idx+1} complete, image saved")

        self._ensure_shutter_closed_v2()
        self.log(f"V3B: CSV log saved: {csv_file}")
        self.log("========== V3B LINE SCAN COMPLETE ==========")


    # ---------------- V4 shape writing test ----------------

    def _build_circle_leadin_leadout_path(self, circle_points, lead_in_len, lead_out_len, closed_circle=True):
        """
        Build tangent lead-in and lead-out for circle.
        Path: lead-in -> circle -> exact start point -> lead-out.
        """
        if len(circle_points) < 3:
            return circle_points

        x0, y0, z0 = circle_points[0]
        x1, y1, _ = circle_points[1]
        dx = x1 - x0
        dy = y1 - y0
        norm = (dx**2 + dy**2) ** 0.5
        if norm <= 1e-12:
            return circle_points

        tx = dx / norm
        ty = dy / norm
        lead_in_len = max(0.0, float(lead_in_len))
        lead_out_len = max(0.0, float(lead_out_len))

        path = []
        if lead_in_len > 0:
            path.append((x0 - lead_in_len * tx, y0 - lead_in_len * ty, z0))
        path.extend(circle_points)
        if closed_circle:
            path.append(circle_points[0])
        if lead_out_len > 0:
            path.append((x0 + lead_out_len * tx, y0 + lead_out_len * ty, z0))
        return path

    def _safe_count_limit(self, points, max_points=5000):
        if len(points) > max_points:
            raise RuntimeError(
                f"Generated {len(points)} points, above safety limit {max_points}. "
                "Increase step size or reduce shape size."
            )

    def _generate_line_shape_points(self, x0, y0, z, length, step):
        n = max(1, int(math.floor(length / step)) + 1)
        return [(x0 + i * step, y0, z) for i in range(n)]

    def _generate_rectangle_points(self, cx, cy, z, width, height, step):
        x_min = cx - width / 2.0
        x_max = cx + width / 2.0
        y_min = cy - height / 2.0
        y_max = cy + height / 2.0
        points = []
        nx = max(1, int(math.floor(width / step)))
        ny = max(1, int(math.floor(height / step)))

        for i in range(nx + 1):
            x = x_min + i * width / nx
            points.append((x, y_min, z))
        for i in range(1, ny + 1):
            y = y_min + i * height / ny
            points.append((x_max, y, z))
        for i in range(1, nx + 1):
            x = x_max - i * width / nx
            points.append((x, y_max, z))
        for i in range(1, ny):
            y = y_max - i * height / ny
            points.append((x_min, y, z))
        return points

    def _generate_circle_points(self, cx, cy, z, radius, step):
        circumference = 2.0 * math.pi * radius
        n = max(8, int(math.ceil(circumference / step)))
        points = []
        for i in range(n):
            theta = 2.0 * math.pi * i / n
            x = cx + radius * math.cos(theta)
            y = cy + radius * math.sin(theta)
            points.append((x, y, z))
        return points

    def _generate_grid_points(self, cx, cy, z, width, height, step):
        x_min = cx - width / 2.0
        y_min = cy - height / 2.0
        nx = max(1, int(math.floor(width / step)))
        ny = max(1, int(math.floor(height / step)))
        xs = [x_min + i * width / nx for i in range(nx + 1)]
        ys = [y_min + j * height / ny for j in range(ny + 1)]
        points = []
        for j, yy in enumerate(ys):
            row = xs if j % 2 == 0 else list(reversed(xs))
            for xx in row:
                points.append((xx, yy, z))
        return points

    def _generate_shape_points_v4(self, shape, x, y, z, width, height, radius, step):
        if shape == "Line":
            return self._generate_line_shape_points(x, y, z, width, step)
        if shape == "Rectangle":
            return self._generate_rectangle_points(x, y, z, width, height, step)
        if shape == "Circle":
            return self._generate_circle_points(x, y, z, radius, step)
        if shape == "Grid":
            return self._generate_grid_points(x, y, z, width, height, step)
        raise ValueError(f"Unknown shape: {shape}")

    def _read_shape_parameters_v6(self):
        shape = self.shape_type_var.get()
        x = float(self.shape_x_entry.get())
        y = float(self.shape_y_entry.get())
        z = float(self.shape_z_entry.get())
        width = float(self.shape_width_entry.get())
        height = float(self.shape_height_entry.get())
        radius = float(self.shape_radius_entry.get())
        step = float(self.shape_step_entry.get())
        hwp_angle = float(self.shape_hwp_entry.get())
        exposure_ms = float(self.shape_exp_entry.get())
        speed = float(self.shape_speed_entry.get())
        wait_after_move = float(self.shape_wait_entry.get())
        auto_hwp = bool(self.auto_hwp_var.get())
        hwp_offset = float(self.hwp_offset_entry.get())

        if step <= 0:
            raise ValueError("Step must be positive.")
        if width <= 0 or height <= 0 or radius <= 0:
            raise ValueError("Width, height, and radius must be positive.")

        points = self._generate_shape_points_v4(shape, x, y, z, width, height, radius, step)
        self._safe_count_limit(points, max_points=5000)

        return {
            "shape": shape,
            "x": x,
            "y": y,
            "z": z,
            "width": width,
            "height": height,
            "radius": radius,
            "step": step,
            "hwp_angle": hwp_angle,
            "exposure_ms": exposure_ms,
            "speed": speed,
            "wait_after_move": wait_after_move,
            "auto_hwp": auto_hwp,
            "hwp_offset": hwp_offset,
            "points": points,
        }

    def preview_shape_v6(self):
        try:
            params = self._read_shape_parameters_v6()
            points = params["points"]

            try:
                if params["shape"] == "Circle" and hasattr(self, "continuous_circle_var") and self.continuous_circle_var.get():
                    closed_circle = bool(self.closed_circle_var.get()) if hasattr(self, "closed_circle_var") else True
                    lead_in_len = float(self.lead_in_entry.get()) if hasattr(self, "lead_in_entry") else 0.0
                    lead_out_len = float(self.lead_out_entry.get()) if hasattr(self, "lead_out_entry") else 0.0
                    points = self._build_circle_leadin_leadout_path(points, lead_in_len, lead_out_len, closed_circle)
            except Exception as e:
                self.log(f"V12B preview lead-in/out warning: {e}")

            xs = [p[0] for p in points]
            ys = [p[1] for p in points]

            win = tk.Toplevel(self.root)
            win.title("V12B Shape Preview - Check Before Scan")
            win.geometry("850x700")

            info = (
                f"Shape: {params['shape']} | Points: {len(points)} | "
                f"Step: {params['step']} mm | Z: {params['z']} mm | "
                f"Auto HWP: {params['auto_hwp']}"
            )

            tk.Label(win, text=info, font=("Arial", 12, "bold")).pack(pady=5)

            fig, ax = plt.subplots(figsize=(7, 6))

            ax.plot(xs, ys, marker="o", linewidth=1)
            ax.scatter(xs[0], ys[0], s=100, marker="o", label="Start")
            ax.scatter(xs[-1], ys[-1], s=100, marker="x", label="End")

            # Direction arrows, not every point if too many
            step_arrow = max(1, len(points) // 20)
            for i in range(0, len(points) - 1, step_arrow):
                dx = xs[i + 1] - xs[i]
                dy = ys[i + 1] - ys[i]
                ax.arrow(
                    xs[i], ys[i], dx, dy,
                    head_width=max(params["step"] * 0.25, 0.001),
                    length_includes_head=True
                )

            ax.set_title("Preview of Generated XY Structure")
            ax.set_xlabel("X position (mm)")
            ax.set_ylabel("Y position (mm)")
            ax.axis("equal")
            ax.grid(True)
            ax.legend()

            text_lines = [
                f"Width/Length = {params['width']} mm",
                f"Height = {params['height']} mm",
                f"Radius = {params['radius']} mm",
                f"Exposure = {params['exposure_ms']} ms",
                f"Speed = {params['speed']} mm/s",
                f"HWP fixed angle = {params['hwp_angle']} deg",
                f"Offset ignored for V12B circle = {params['hwp_offset']} deg",
            ]

            if params["auto_hwp"]:
                # Show first few predicted HWP values
                predicted = []
                for idx, (px, py, pz) in enumerate(points[:min(8, len(points))]):
                    if idx == 0 and len(points) > 1:
                        nx, ny, _ = points[1]
                        motion_angle = self._angle_deg_from_segment(px, py, nx, ny)
                    elif idx > 0:
                        prevx, prevy, _ = points[idx - 1]
                        motion_angle = self._angle_deg_from_segment(prevx, prevy, px, py)
                    else:
                        motion_angle = 0.0

                    if motion_angle is None:
                        motion_angle = 0.0
                    hwp, pol = self._hwp_for_perpendicular_polarization(motion_angle, params["hwp_offset"])
                    predicted.append(f"pt {idx}: motion={motion_angle:.1f}°, pol={pol:.1f}°, HWP={hwp:.1f}°")

                text_lines.append("")
                text_lines.append("First predicted Auto-HWP values:")
                text_lines.extend(predicted)

            ax.text(
                1.02, 0.98,
                "\\n".join(text_lines),
                transform=ax.transAxes,
                va="top",
                fontsize=9,
                bbox=dict(boxstyle="round", alpha=0.2)
            )

            canvas = FigureCanvasTkAgg(fig, master=win)
            canvas.draw()
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

            tk.Label(
                win,
                text="If this preview looks correct, close this window and click RUN SHAPE TEST.",
                fg="blue",
                font=("Arial", 11, "bold")
            ).pack(pady=5)

            self.log(f"V6 preview generated: shape={params['shape']}, points={len(points)}")

        except Exception as e:
            messagebox.showerror("Preview Error", str(e))
            self.log(f"V6 preview error: {e}")

    def _angle_deg_from_segment(self, x_prev, y_prev, x_now, y_now):
        dx = x_now - x_prev
        dy = y_now - y_prev
        if abs(dx) < 1e-12 and abs(dy) < 1e-12:
            return None
        return math.degrees(math.atan2(dy, dx))

    def _wrap_angle_pm180(self, angle_deg):
        """Wrap angle difference to -180..+180 degrees."""
        return (float(angle_deg) + 180.0) % 360.0 - 180.0

    def _hwp_from_reference_tangent(self, motion_angle_deg, reference_motion_angle_deg, reference_hwp_deg):
        """
        Professor-reference mode: no separate offset/calibration is used.

        The HWP value typed in the GUI is treated as the reference HWP angle for
        the lead-in direction. Example: if 34 deg gives max power/perpendicular
        polarization on the lead-in line, then lead-in uses exactly 34 deg.

        For later circle segments, polarization should remain perpendicular to
        the local tangent. A half-wave plate changes polarization by 2*HWP, so
        when the tangent angle changes by delta, HWP changes by delta/2.

        HWP = reference_hwp + (current_tangent - reference_tangent)/2
        wrapped to 0..180 deg.
        """
        delta_tangent = self._wrap_angle_pm180(float(motion_angle_deg) - float(reference_motion_angle_deg))
        hwp_deg = (float(reference_hwp_deg) + 0.5 * delta_tangent) % 180.0
        desired_pol_deg = (float(motion_angle_deg) + 90.0) % 180.0
        return hwp_deg, desired_pol_deg

    # Backward-compatible name used by older point-by-point modes.
    def _hwp_for_perpendicular_polarization(self, motion_angle_deg, hwp_offset_deg=0.0):
        return self._hwp_from_reference_tangent(motion_angle_deg, 0.0, hwp_offset_deg)

    def run_continuous_circle_v8(self, shape, points, shape_folder, csv_file,
                                 hwp_angle, auto_hwp, hwp_offset, hwp_settle,
                                 circle_hwp_wait, exposure_ms, speed,
                                 wait_after_move, capture_after_circle):
        """
        V8 continuous circle mode:
        - Move to first circle point with shutter closed.
        - Open shutter once.
        - Move through all circle points while shutter remains open.
        - Close shutter once after the complete circle.
        - Capture one final image and log all points.
        """
        if shape != "Circle":
            raise RuntimeError("V8 continuous mode is only for Shape = Circle.")

        self.log("========== V12B CONTINUOUS CIRCLE START ==========")
        self.log(f"V12B: path points={len(points)}")
        self.log("V12B: shutter opens once before lead-in and closes once after lead-out")

        self._ensure_shutter_closed_v2()

        # Move to starting point with shutter CLOSED
        x0, y0, z0 = points[0]
        self.log(f"V12B step 1: move to circle start with shutter CLOSED: X={x0:.6f}, Y={y0:.6f}, Z={z0:.6f}")
        # V12B: move X and Y together to avoid X-then-Y staircase motion
        self.stage.move_linear_absolute(["X", "Y"], [x0, y0], speed)
        self.stage.move_absolute("Z", z0, speed)
        self.update_stage_positions()
        if wait_after_move > 0:
            time.sleep(wait_after_move)

        # Prepare initial HWP before opening shutter.
        # V12B professor-reference mode:
        # - The typed HWP angle is the reference angle for the lead-in line.
        # - No separate offset/calibration is applied during circle writing.
        motion_angle = 0.0
        reference_motion_angle = 0.0
        desired_pol = None
        this_hwp = hwp_angle

        if len(points) > 1:
            nx, ny, _ = points[1]
            reference_motion_angle = self._angle_deg_from_segment(x0, y0, nx, ny)
            if reference_motion_angle is None:
                reference_motion_angle = 0.0
            motion_angle = reference_motion_angle

        if auto_hwp:
            this_hwp = hwp_angle  # lead-in reference HWP, e.g. 34 deg
            desired_pol = (motion_angle + 90.0) % 180.0
            self.log(
                f"V12B step 2: lead-in/reference HWP before opening shutter: "
                f"reference tangent={reference_motion_angle:.3f} deg, desired pol={desired_pol:.3f} deg, "
                f"HWP={this_hwp:.3f} deg (no offset calibration)"
            )
        else:
            self.log(f"V12B step 2: Auto HWP OFF, fixed HWP={hwp_angle:.3f} deg")

        self.waveplate.move_to_angle(this_hwp)
        if hwp_settle > 0:
            time.sleep(hwp_settle)

        with open(csv_file, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                0, datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f"),
                shape, x0, y0, z0, motion_angle, desired_pol, this_hwp,
                auto_hwp, hwp_offset, exposure_ms, speed, wait_after_move,
                "V12B_START_SHUTTER_CLOSED", ""
            ])

        # Open shutter ONCE
        self.log("V12B step 3: OPEN SHUTTER ONCE — continuous circle writing begins")
        self.log("V12B SAFETY: shutter will remain open until circle path is complete; use GLOBAL SAFE STOP if needed.")
        self._open_shutter_v2()

        try:
            # Move around circle while shutter remains open
            for idx in range(1, len(points)):
                px, py, pz = points[idx]
                prevx, prevy, _ = points[idx - 1]

                motion_angle = self._angle_deg_from_segment(prevx, prevy, px, py)
                if motion_angle is None:
                    motion_angle = 0.0

                desired_pol = None
                this_hwp = hwp_angle

                if auto_hwp:
                    this_hwp, desired_pol = self._hwp_from_reference_tangent(motion_angle, reference_motion_angle, hwp_angle)
                    self.log(
                        f"V12B circle point {idx+1}/{len(points)}: "
                        f"tangent={motion_angle:.3f} deg, desired pol={desired_pol:.3f} deg, HWP={this_hwp:.3f} deg"
                    )
                    self.waveplate.move_to_angle(this_hwp)
                    # V9 laser-test safety: default circle_hwp_wait is 0.
                    # Avoid extra dwell/exposure at fixed positions while shutter is open.
                    if circle_hwp_wait > 0:
                        self.log(f"WARNING: circle_hwp_wait={circle_hwp_wait}s while shutter is open")
                        time.sleep(circle_hwp_wait)
                else:
                    self.log(f"V12B circle point {idx+1}/{len(points)}: fixed HWP={hwp_angle:.3f} deg")

                # V12B: coordinated XY move for each circle segment, not X then Y
                self.stage.move_linear_absolute(["X", "Y"], [px, py], speed)
                self.stage.move_absolute("Z", pz, speed)
                self.update_stage_positions()

                if wait_after_move > 0:
                    time.sleep(wait_after_move)

                with open(csv_file, "a", newline="", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        idx, datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f"),
                        shape, px, py, pz, motion_angle, desired_pol, this_hwp,
                        auto_hwp, hwp_offset, exposure_ms, speed, wait_after_move,
                        "V12B_SHUTTER_OPEN_CONTINUOUS", ""
                    ])

        finally:
            self.log("V12B step 5: CLOSE SHUTTER — continuous circle finished")
            self._close_shutter_v2()

        image_file = ""
        if capture_after_circle:
            self.log("V12B step 6: capture final camera image after circle")
            image_file = self.camera.capture_image(
                shape_folder,
                filename_prefix="v8_continuous_circle_final_capture",
                exposure_ms=exposure_ms
            )

        with open(csv_file, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                len(points), datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f"),
                shape, x0, y0, z0, "", "", "", auto_hwp, hwp_offset,
                exposure_ms, speed, wait_after_move, "V12B_END_SHUTTER_CLOSED", image_file
            ])

        self.log("========== V12B CONTINUOUS CIRCLE COMPLETE ==========")

    def safe_run_shape_test(self):
        answer = messagebox.askyesno(
            "Run V12B Shape Test",
            "This will move through a generated shape and trigger shutter/camera at every point.\n\n"
            "For LASER test: use LOW power first. For continuous circle, shutter will stay OPEN during the whole circle.\n\nUse small radius, safe Z, and confirm beam path is safe. Continue?"
        )
        if not answer:
            self.log("V12B shape test cancelled by user")
            return
        self.run_threaded(self.run_shape_test, "V12B shape test error")

    def run_shape_test(self):
        shape = self.shape_type_var.get()
        x = float(self.shape_x_entry.get())
        y = float(self.shape_y_entry.get())
        z = float(self.shape_z_entry.get())
        width = float(self.shape_width_entry.get())
        height = float(self.shape_height_entry.get())
        radius = float(self.shape_radius_entry.get())
        step = float(self.shape_step_entry.get())
        hwp_angle = float(self.shape_hwp_entry.get())
        exposure_ms = float(self.shape_exp_entry.get())
        speed = float(self.shape_speed_entry.get())
        wait_after_move = float(self.shape_wait_entry.get())
        save_folder = self.seq_folder_entry.get().strip()

        auto_hwp = bool(self.auto_hwp_var.get())
        hwp_offset = float(self.hwp_offset_entry.get())
        hwp_settle = float(self.hwp_settle_entry.get())
        continuous_circle = bool(self.continuous_circle_var.get())
        circle_hwp_wait = float(self.circle_hwp_wait_entry.get())
        capture_after_circle = bool(self.capture_after_circle_var.get())
        closed_circle = bool(self.closed_circle_var.get()) if hasattr(self, "closed_circle_var") else True
        lead_in_len = float(self.lead_in_entry.get()) if hasattr(self, "lead_in_entry") else 0.0
        lead_out_len = float(self.lead_out_entry.get()) if hasattr(self, "lead_out_entry") else 0.0

        if step <= 0:
            raise ValueError("Step must be positive.")
        if exposure_ms <= 0:
            raise ValueError("Exposure time must be positive.")
        if speed <= 0:
            raise ValueError("Speed must be positive.")
        if z > 5.0:
            raise RuntimeError("Z target above temporary safety limit 5.0 mm.")
        if width <= 0 or height <= 0 or radius <= 0:
            raise ValueError("Width, height, and radius must be positive.")

        points = self._generate_shape_points_v4(shape, x, y, z, width, height, radius, step)
        self._safe_count_limit(points, max_points=5000)

        os.makedirs(save_folder, exist_ok=True)
        shape_folder = os.path.join(save_folder, datetime.now().strftime(f"V4_{shape}_test_%Y%m%d_%H%M%S"))
        os.makedirs(shape_folder, exist_ok=True)

        csv_file = os.path.join(shape_folder, f"{shape.lower()}_shape_log.csv")
        with open(csv_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "point_index", "timestamp", "shape", "x_mm", "y_mm", "z_mm",
                "motion_angle_deg", "desired_pol_deg", "hwp_deg", "auto_hwp",
                "hwp_offset_deg", "exposure_ms", "speed_mm_s", "wait_after_move_s", "mode", "image_file"
            ])

        if shape == "Circle" and continuous_circle:
            self.log("V12B selected: Shape=Circle and Continuous Circle is ON")
            self.log(f"V12B: closed_circle={closed_circle}, lead_in={lead_in_len} mm, lead_out={lead_out_len} mm")
            points = self._build_circle_leadin_leadout_path(points, lead_in_len, lead_out_len, closed_circle)
            self._safe_count_limit(points, max_points=5000)
            self.log(f"V12B: total path points after lead-in/lead-out = {len(points)}")

            self.run_continuous_circle_v8(
                shape, points, shape_folder, csv_file,
                hwp_angle, auto_hwp, hwp_offset, hwp_settle,
                circle_hwp_wait, exposure_ms, speed,
                wait_after_move, capture_after_circle
            )
            return

        self.log("========== V9 POINT-BY-POINT SHAPE TEST START ==========")
        self.log(f"V5: shape={shape}, points={len(points)}, folder={shape_folder}")

        self._ensure_shutter_closed_v2()

        if not auto_hwp:
            self.log(f"V5: Auto HWP OFF. Moving HWP once to fixed angle {hwp_angle} deg")
            self.waveplate.move_to_angle(hwp_angle)
            time.sleep(2.0)
        else:
            self.log(f"V5: Auto HWP ON. HWP offset/calibration = {hwp_offset} deg")

        previous_xy = None

        for idx, (px, py, pz) in enumerate(points):
            self.log(f"--- V5 {shape} point {idx+1}/{len(points)}: X={px:.6f}, Y={py:.6f}, Z={pz:.6f} ---")

            motion_angle = None
            desired_pol = None
            this_hwp = hwp_angle

            if auto_hwp:
                if previous_xy is None:
                    # For first point, use direction toward the second point if available.
                    if len(points) > 1:
                        nx, ny, _ = points[1]
                        motion_angle = self._angle_deg_from_segment(px, py, nx, ny)
                    else:
                        motion_angle = 0.0
                else:
                    motion_angle = self._angle_deg_from_segment(previous_xy[0], previous_xy[1], px, py)

                if motion_angle is None:
                    motion_angle = 0.0

                this_hwp, desired_pol = self._hwp_from_reference_tangent(motion_angle, reference_motion_angle, hwp_angle)
                self.log(
                    f"V5 Auto HWP: motion angle={motion_angle:.3f} deg, "
                    f"desired polarization={desired_pol:.3f} deg, HWP={this_hwp:.3f} deg"
                )

                self.waveplate.move_to_angle(this_hwp)
                if hwp_settle > 0:
                    time.sleep(hwp_settle)

            self._ensure_shutter_closed_v2()
            self.stage.move_absolute("X", px, speed)
            self.stage.move_absolute("Y", py, speed)
            self.stage.move_absolute("Z", pz, speed)
            self.update_stage_positions()

            if wait_after_move > 0:
                time.sleep(wait_after_move)

            self._open_shutter_v2()
            time.sleep(exposure_ms / 1000.0)
            self._close_shutter_v2()

            image_file = self.camera.capture_image(
                shape_folder,
                filename_prefix=f"{shape.lower()}_{idx:04d}_X_{px:.6f}_Y_{py:.6f}_Z_{pz:.6f}",
                exposure_ms=exposure_ms
            )

            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
            with open(csv_file, "a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([
                    idx, timestamp, shape, px, py, pz,
                    motion_angle, desired_pol, this_hwp, auto_hwp,
                    hwp_offset, exposure_ms, speed, wait_after_move, "POINT_BY_POINT", image_file
                ])

            previous_xy = (px, py)
            self.log(f"V5: {shape} point {idx+1} complete")

        self._ensure_shutter_closed_v2()
        self.log(f"V8: CSV log saved: {csv_file}")
        self.log("========== V9 POINT-BY-POINT SHAPE TEST COMPLETE ==========")

    # ---------------- Global stop ----------------

    def global_safe_stop(self):
        self.log("GLOBAL SAFE STOP requested")

        # Try shutter close first if command is configured
        try:
            close_cmd = self.shutter_close_cmd_entry.get().strip()
            if self.shutter.ser is not None and "?" not in close_cmd:
                self.shutter.close_manual_command(close_cmd)
            else:
                self.log("Global stop: shutter close command not configured or shutter not connected")
        except Exception as e:
            self.log(f"Global stop shutter warning: {e}")

        # Stop stage
        try:
            if self.stage.controller is not None:
                self.stage.stop()
        except Exception as e:
            self.log(f"Global stop stage warning: {e}")

        # Stop waveplate
        try:
            if self.waveplate.connected:
                self.waveplate.stop()
        except Exception as e:
            self.log(f"Global stop waveplate warning: {e}")

    def on_closing(self):
        try:
            self.global_safe_stop()
        except Exception:
            pass

        try:
            self.waveplate.close()
        except Exception:
            pass

        try:
            self.shutter.close()
        except Exception:
            pass

        try:
            self.camera.close()
        except Exception:
            pass

        self.root.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    app = MantisMainGUI(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()
