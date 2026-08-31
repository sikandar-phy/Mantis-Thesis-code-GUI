
import sys
import numpy as np
from PIL import Image

try:
    from scipy.special import jv
    from scipy.ndimage import maximum_filter
    SCIPY_AVAILABLE = True
except Exception:
    SCIPY_AVAILABLE = False

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QLabel, QLineEdit, QPushButton, QComboBox, QCheckBox, QFileDialog,
    QTextEdit, QGroupBox, QMessageBox, QTabWidget, QSplitter, QSlider
)
from PyQt5.QtCore import Qt

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure


def symmetric_orders(n, include_zero_for_even=False):
    if n < 1:
        raise ValueError("Number of beams must be >= 1.")
    if n == 1:
        return [0]
    if n % 2 == 1:
        h = n // 2
        return list(range(-h, h + 1))
    h = n // 2
    if include_zero_for_even:
        # For even n, this returns n+1 including zero; useful for testing zero-order cases.
        return list(range(-h, h + 1))
    return list(range(-h, 0)) + list(range(1, h + 1))


def build_orders(shape, n_beams, rows, cols, include_zero_for_even=False, line_direction="Horizontal"):
    """
    Build diffraction orders.

    For Line:
      Horizontal -> line along x direction
      Vertical   -> line along y direction
      Custom angle is handled later using a rotated coordinate.
    """
    if shape == "Line":
        line_orders = symmetric_orders(n_beams, include_zero_for_even)
        if line_direction == "Vertical":
            return [0], line_orders
        return line_orders, [0]

    if shape == "Rectangle":
        return symmetric_orders(cols, include_zero_for_even), symmetric_orders(rows, include_zero_for_even)

    if shape == "Square":
        return symmetric_orders(cols, include_zero_for_even), symmetric_orders(cols, include_zero_for_even)

    raise ValueError("Unknown beam shape.")


def period_pixels_from_spacing(f_obj_m, wavelength_m, spacing_um, pixel_m):
    return f_obj_m * wavelength_m / ((spacing_um * 1e-6) * pixel_m)


def multi_order_phase_1d(coord, orders, period_pixels, pixel_m):
    if len(orders) == 1 and orders[0] == 0:
        return np.zeros_like(coord)
    period_m = period_pixels * pixel_m
    field = np.zeros_like(coord, dtype=np.complex128)
    for m in orders:
        field += np.exp(1j * 2 * np.pi * m * coord / period_m)
    return np.angle(field)


def gaussian_amplitude(R, diameter_m):
    w0 = diameter_m / 2.0
    return np.exp(-(R ** 2) / (w0 ** 2))


def order_angle_data(orders_x, orders_y, wavelength_m, period_m):
    data = []
    for oy in orders_y:
        for ox in orders_x:
            sx = ox * wavelength_m / period_m
            sy = oy * wavelength_m / period_m
            st = np.sqrt(sx ** 2 + sy ** 2)
            data.append({
                "mx": ox,
                "my": oy,
                "sin_x": sx,
                "sin_y": sy,
                "sin_total": st,
                "theta_deg": np.rad2deg(np.arcsin(min(st, 0.999999)))
            })
    return data


def beam_positions_um(order_data, f_obj_m):
    pts = []
    for od in order_data:
        pts.append((f_obj_m * od["sin_x"] * 1e6, f_obj_m * od["sin_y"] * 1e6))
    return np.array(pts, dtype=float)


def nearest_spacing(pts):
    if len(pts) < 2:
        return 0.0
    vals = []
    for i in range(len(pts)):
        for j in range(i + 1, len(pts)):
            d = np.linalg.norm(pts[i] - pts[j])
            if d > 1e-12:
                vals.append(d)
    return float(min(vals)) if vals else 0.0


def fft_marker_positions(orders_x, orders_y, slm_w, slm_h, period_pixels):
    dx = slm_w / period_pixels
    dy = slm_h / period_pixels
    pts = []
    for oy in orders_y:
        for ox in orders_x:
            pts.append((ox * dx, oy * dy))
    return np.array(pts, dtype=float)


def detect_fft_peaks(fftI, crop=700, threshold_rel=0.20, min_distance=12, max_peaks=50):
    """
    Detect local maxima in central FFT crop.
    Returns peak coordinates in FFT-index coordinates centered at zero.
    """
    h, w = fftI.shape
    crop = min(crop, h, w)
    y0 = h // 2 - crop // 2
    x0 = w // 2 - crop // 2
    img = fftI[y0:y0 + crop, x0:x0 + crop].copy()

    # suppress the broad low background using relative threshold
    mx = img.max()
    if mx <= 0:
        return np.empty((0, 2)), np.array([])

    if SCIPY_AVAILABLE:
        local = maximum_filter(img, size=min_distance) == img
        mask = local & (img > threshold_rel * mx)
        ys, xs = np.where(mask)
    else:
        # fallback: simple top brightest points, less robust
        flat = img.ravel()
        idx = np.argsort(flat)[-max_peaks:]
        ys, xs = np.unravel_index(idx, img.shape)

    vals = img[ys, xs]
    order = np.argsort(vals)[::-1]
    ys = ys[order][:max_peaks]
    xs = xs[order][:max_peaks]
    vals = vals[order][:max_peaks]

    # convert to FFT-index coordinates centered at zero
    coords = np.column_stack((xs - crop / 2, ys - crop / 2))
    return coords, vals


def match_expected_to_detected(expected, detected, tolerance_px=18):
    if len(expected) == 0 or len(detected) == 0:
        return 0, []
    matches = []
    used = set()
    for i, e in enumerate(expected):
        dists = np.linalg.norm(detected - e, axis=1)
        j = int(np.argmin(dists))
        if dists[j] <= tolerance_px and j not in used:
            matches.append((i, j, float(dists[j])))
            used.add(j)
    return len(matches), matches


def angular_spectrum_propagate(field, dx_m, wavelength_m, z_m, NA=None):
    ny, nx = field.shape
    k = 2 * np.pi / wavelength_m
    fx = np.fft.fftfreq(nx, d=dx_m)
    fy = np.fft.fftfreq(ny, d=dx_m)
    FX, FY = np.meshgrid(fx, fy)
    kx = 2 * np.pi * FX
    ky = 2 * np.pi * FY
    kz2 = k ** 2 - kx ** 2 - ky ** 2
    kz = np.zeros_like(kx, dtype=np.complex128)
    prop = kz2 >= 0
    kz[prop] = np.sqrt(kz2[prop])
    kz[~prop] = 1j * np.sqrt(-kz2[~prop])
    H = np.exp(1j * kz * z_m)
    if NA is not None:
        H *= (np.sqrt(FX ** 2 + FY ** 2) <= (NA / wavelength_m))
    return np.fft.ifft2(np.fft.fft2(field) * H)


def build_sample_field(positions_um, order_data, p):
    wavelength_m = p["wavelength_nm"] * 1e-9
    NA = p["objective_NA"]
    core_um = max(p["bessel_diameter_um"], 0.1)
    window_um = p["sample_window_um"]
    npx = int(p["sample_pixels"])

    x_um = np.linspace(-window_um / 2, window_um / 2, npx)
    y_um = np.linspace(-window_um / 2, window_um / 2, npx)
    X_um, Y_um = np.meshgrid(x_um, y_um)
    X_m = X_um * 1e-6
    Y_m = Y_um * 1e-6

    field = np.zeros_like(X_um, dtype=np.complex128)
    accepted = []

    kr_um = 2.0 * 2.4048 / core_um
    env_radius_um = p["bessel_envelope_factor"] * core_um

    for pos, od in zip(positions_um, order_data):
        ok = od["sin_total"] <= NA
        accepted.append(ok)
        if not ok:
            continue

        dx = X_um - pos[0]
        dy = Y_um - pos[1]
        r = np.sqrt(dx ** 2 + dy ** 2)
        phi = np.arctan2(dy, dx)

        ell = int(p["vortex_charge"]) if p["use_vortex"] else 0

        if SCIPY_AVAILABLE:
            radial = jv(abs(ell), kr_um * r)
        else:
            sigma = core_um / 2.355
            radial = np.exp(-(r ** 2) / (2 * sigma ** 2))

        if ell != 0:
            radial = radial * np.exp(1j * ell * phi)

        envelope = np.exp(-(r / env_radius_um) ** 2)
        k = 2 * np.pi / wavelength_m
        tilt = np.exp(1j * k * (od["sin_x"] * X_m + od["sin_y"] * Y_m))

        field += radial * envelope * tilt

    dx_sample_m = (x_um[1] - x_um[0]) * 1e-6
    return x_um, y_um, field, dx_sample_m, np.array(accepted, dtype=bool)


def generate_design(p, z_um=0.0):
    sw = int(p["slm_width"])
    sh = int(p["slm_height"])
    pixel_m = p["pixel_size_um"] * 1e-6
    wavelength_m = p["wavelength_nm"] * 1e-9
    beam_diam_m = p["beam_diameter_mm"] * 1e-3
    f_obj_m = p["objective_focal_mm"] * 1e-3

    x = (np.arange(sw) - sw / 2) * pixel_m
    y = (np.arange(sh) - sh / 2) * pixel_m
    X, Y = np.meshgrid(x, y)
    R = np.sqrt(X ** 2 + Y ** 2)
    PHI = np.arctan2(Y, X)

    gaussian = gaussian_amplitude(R, beam_diam_m)

    alpha = np.deg2rad(p["axicon_angle_deg"])
    cone_angle = (p["axicon_n"] - 1.0) * alpha
    axicon_phase = (2 * np.pi / wavelength_m) * R * np.sin(cone_angle)

    orders_x, orders_y = build_orders(
        p["beam_shape"], int(p["number_of_beams"]),
        int(p["rows"]), int(p["cols"]),
        include_zero_for_even=p["include_zero_for_even"],
        line_direction=p["line_direction"]
    )

    period_pixels = period_pixels_from_spacing(
        f_obj_m, wavelength_m, p["spacing_um"], pixel_m
    )
    period_m = period_pixels * pixel_m

    # Multi-beam phase.
    # For a custom-angle line, rotate the coordinate:
    # U = X cos(theta) + Y sin(theta)
    if p["beam_shape"] == "Line" and p["line_direction"] == "Custom angle":
        theta = np.deg2rad(p["line_angle_deg"])
        U = X * np.cos(theta) + Y * np.sin(theta)
        line_orders = symmetric_orders(int(p["number_of_beams"]), p["include_zero_for_even"])
        cgh_phase = multi_order_phase_1d(U, line_orders, period_pixels, pixel_m)
    else:
        phase_x = multi_order_phase_1d(X, orders_x, period_pixels, pixel_m)
        phase_y = multi_order_phase_1d(Y, orders_y, period_pixels, pixel_m)
        cgh_phase = np.angle(np.exp(1j * (phase_x + phase_y)))

    vortex_phase = int(p["vortex_charge"]) * PHI if p["use_vortex"] else np.zeros_like(PHI)

    total_phase = axicon_phase + cgh_phase + vortex_phase
    wrapped = np.mod(total_phase, 2 * np.pi)
    mask8 = np.uint8(255 * wrapped / (2 * np.pi))

    field_slm = gaussian * np.exp(1j * wrapped)
    fft = np.fft.fftshift(np.fft.fft2(field_slm))
    fftI = np.abs(fft) ** 2
    if np.max(fftI) > 0:
        fftI /= np.max(fftI)

    if p["beam_shape"] == "Line" and p["line_direction"] == "Custom angle":
        theta = np.deg2rad(p["line_angle_deg"])
        line_orders = symmetric_orders(int(p["number_of_beams"]), p["include_zero_for_even"])
        base = sw / period_pixels
        expected_markers = np.array([
            [m * base * np.cos(theta), m * base * np.sin(theta)]
            for m in line_orders
        ], dtype=float)
    else:
        expected_markers = fft_marker_positions(orders_x, orders_y, sw, sh, period_pixels)

    detected_peaks, detected_vals = detect_fft_peaks(
        fftI,
        crop=700,
        threshold_rel=p["fft_peak_threshold"],
        min_distance=int(p["fft_peak_distance"]),
        max_peaks=80
    )
    matched_count, matches = match_expected_to_detected(
        expected_markers, detected_peaks, tolerance_px=p["fft_match_tolerance"]
    )

    if p["beam_shape"] == "Line" and p["line_direction"] == "Custom angle":
        theta = np.deg2rad(p["line_angle_deg"])
        line_orders = symmetric_orders(int(p["number_of_beams"]), p["include_zero_for_even"])
        order_data = []
        for m in line_orders:
            sx = m * wavelength_m / period_m * np.cos(theta)
            sy = m * wavelength_m / period_m * np.sin(theta)
            st = float(np.sqrt(sx*sx + sy*sy))
            order_data.append({
                "mx": m,
                "my": 0,
                "sin_x": sx,
                "sin_y": sy,
                "sin_total": st,
                "theta_deg": np.rad2deg(np.arcsin(min(st, 0.999999)))
            })
        positions = beam_positions_um(order_data, f_obj_m)
    else:
        order_data = order_angle_data(orders_x, orders_y, wavelength_m, period_m)
        positions = beam_positions_um(order_data, f_obj_m)

    sx, sy, field0, dx_sample_m, accepted = build_sample_field(positions, order_data, p)
    fieldz = angular_spectrum_propagate(
        field0, dx_sample_m, wavelength_m, z_um * 1e-6, NA=p["objective_NA"]
    )

    z_env = np.exp(-(z_um / max(p["zmax_um"], 1.0)) ** 2)
    fieldz *= z_env

    sampleI = np.abs(fieldz) ** 2
    if np.max(sampleI) > 0:
        sampleI /= np.max(sampleI)

    return {
        "p": p,
        "z_um": z_um,
        "gaussian": gaussian,
        "axicon_phase": axicon_phase,
        "cgh_phase": cgh_phase,
        "mask8": mask8,
        "fft": fftI,
        "expected_markers": expected_markers,
        "detected_peaks": detected_peaks,
        "detected_vals": detected_vals,
        "matched_count": matched_count,
        "matches": matches,
        "sample_x": sx,
        "sample_y": sy,
        "sampleI": sampleI,
        "positions": positions,
        "orders_x": orders_x,
        "orders_y": orders_y,
        "order_data": order_data,
        "accepted": accepted,
        "period_pixels": period_pixels,
        "period_um": period_m * 1e6,
        "nearest_spacing_um": nearest_spacing(positions),
        "beam_count": len(positions),
        "accepted_count": int(np.sum(accepted)),
        "cone_angle_deg": np.rad2deg(cone_angle),
    }


class PlotCanvas(FigureCanvas):
    def __init__(self):
        self.fig = Figure(figsize=(9, 6), dpi=100)
        super().__init__(self.fig)

    def show_results(self, r):
        self.fig.clear()

        ax1 = self.fig.add_subplot(2, 3, 1)
        ax1.imshow(r["gaussian"], cmap="gray")
        ax1.set_title("Gaussian on SLM")
        ax1.axis("off")

        ax2 = self.fig.add_subplot(2, 3, 2)
        ax2.imshow(np.mod(r["axicon_phase"], 2 * np.pi), cmap="gray")
        ax2.set_title("Axicon phase")
        ax2.axis("off")

        ax3 = self.fig.add_subplot(2, 3, 3)
        ax3.imshow(np.mod(r["cgh_phase"], 2 * np.pi), cmap="gray")
        ax3.set_title("Multi-beam CGH")
        ax3.axis("off")

        ax4 = self.fig.add_subplot(2, 3, 4)
        ax4.imshow(r["mask8"], cmap="gray")
        ax4.set_title("Final SLM mask")
        ax4.axis("off")

        fft = r["fft"]
        h, w = fft.shape
        crop = min(700, h, w)
        y0 = h // 2 - crop // 2
        x0 = w // 2 - crop // 2
        fft_crop = fft[y0:y0 + crop, x0:x0 + crop]
        extent = [-crop / 2, crop / 2, -crop / 2, crop / 2]

        ax5 = self.fig.add_subplot(2, 3, 5)
        ax5.imshow(np.log10(fft_crop + 1e-8), cmap="inferno", origin="lower", extent=extent)
        if len(r["expected_markers"]):
            m = r["expected_markers"]
            keep = (np.abs(m[:, 0]) < crop / 2) & (np.abs(m[:, 1]) < crop / 2)
            ax5.scatter(m[keep, 0], m[keep, 1], s=60, facecolors="none", edgecolors="cyan", label="expected")
        if len(r["detected_peaks"]):
            d = r["detected_peaks"]
            keep = (np.abs(d[:, 0]) < crop / 2) & (np.abs(d[:, 1]) < crop / 2)
            ax5.scatter(d[keep, 0], d[keep, 1], s=20, marker="x", color="lime", label="detected")
        ax5.set_title("FFT: expected vs detected peaks")
        ax5.set_xlabel("FFT index")
        ax5.set_ylabel("FFT index")

        ax6 = self.fig.add_subplot(2, 3, 6)
        ext = [r["sample_x"][0], r["sample_x"][-1], r["sample_y"][0], r["sample_y"][-1]]
        ax6.imshow(r["sampleI"], cmap="inferno", origin="lower", extent=ext)
        if len(r["positions"]):
            acc = r["accepted"]
            ax6.scatter(r["positions"][acc, 0], r["positions"][acc, 1],
                        s=35, facecolors="none", edgecolors="cyan")
            if np.any(~acc):
                ax6.scatter(r["positions"][~acc, 0], r["positions"][~acc, 1],
                            s=45, marker="x", color="red")
        ax6.set_title(f"Sample field, z={r['z_um']:.0f} µm")
        ax6.set_xlabel("x (µm)")
        ax6.set_ylabel("y (µm)")

        self.fig.tight_layout()
        self.draw()


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("MANTIS Multi-Bessel Research Suite V6.1")
        self.resize(1450, 880)
        self.result = None

        central = QWidget()
        self.setCentralWidget(central)
        layout = QHBoxLayout(central)
        splitter = QSplitter(Qt.Horizontal)
        layout.addWidget(splitter)

        self.left = QWidget()
        self.left_layout = QVBoxLayout(self.left)
        self.tabs = QTabWidget()
        self.left_layout.addWidget(self.tabs)
        self.plot = PlotCanvas()

        splitter.addWidget(self.left)
        splitter.addWidget(self.plot)
        splitter.setSizes([455, 995])

        self.fields = {}
        self.build_design_tab()
        self.build_report_tab()

    def add_field(self, grid, row, label, default):
        grid.addWidget(QLabel(label), row, 0)
        e = QLineEdit(str(default))
        grid.addWidget(e, row, 1)
        self.fields[label] = e

    def build_design_tab(self):
        tab = QWidget()
        lay = QVBoxLayout(tab)

        exp = QGroupBox("Experimental parameters")
        g = QGridLayout(exp)
        fields = [
            ("SLM width px", 1920),
            ("SLM height px", 1080),
            ("SLM pixel size µm", 8),
            ("Wavelength nm", 1030),
            ("Gaussian beam diameter mm", 4),
            ("Objective focal length mm", 10),
            ("Objective NA", 0.40),
            ("Axicon angle deg", 0.5),
            ("Axicon refractive index", 1.45),
            ("Expected Bessel diameter µm", 2.0),
            ("Bessel zmax estimate µm", 80.0),
            ("Bessel envelope factor", 4.5),
            ("Sample window µm", 50.0),
            ("Sample simulation pixels", 512),
        ]
        for i, (lab, val) in enumerate(fields):
            self.add_field(g, i, lab, val)
        lay.addWidget(exp)

        arr = QGroupBox("Beam array")
        ag = QGridLayout(arr)
        ag.addWidget(QLabel("Beam shape"), 0, 0)
        self.shape_box = QComboBox()
        self.shape_box.addItems(["Line", "Rectangle", "Square"])
        ag.addWidget(self.shape_box, 0, 1)

        ag.addWidget(QLabel("Line direction"), 1, 0)
        self.line_direction_box = QComboBox()
        self.line_direction_box.addItems(["Horizontal", "Vertical", "Custom angle"])
        ag.addWidget(self.line_direction_box, 1, 1)

        for i, (lab, val) in enumerate([
            ("Line angle deg", 0),
            ("Number of beams", 6),
            ("Rows", 2),
            ("Columns", 3),
            ("Center spacing µm", 4),
        ], start=2):
            self.add_field(ag, i, lab, val)

        self.include_zero_check = QCheckBox("Include zero order for even beam number")
        ag.addWidget(self.include_zero_check, 7, 0, 1, 2)
        lay.addWidget(arr)

        fftbox = QGroupBox("FFT peak detection")
        fg = QGridLayout(fftbox)
        self.add_field(fg, 0, "FFT peak threshold", 0.20)
        self.add_field(fg, 1, "FFT peak distance px", 12)
        self.add_field(fg, 2, "FFT match tolerance px", 18)
        lay.addWidget(fftbox)

        vort = QGroupBox("Optional vortex")
        vg = QGridLayout(vort)
        self.vortex_check = QCheckBox("Use vortex phase")
        vg.addWidget(self.vortex_check, 0, 0, 1, 2)
        self.add_field(vg, 1, "Vortex charge", 1)
        lay.addWidget(vort)

        zbox = QGroupBox("Z propagation preview")
        zl = QVBoxLayout(zbox)
        self.z_label = QLabel("z = 0 µm")
        self.z_slider = QSlider(Qt.Horizontal)
        self.z_slider.setMinimum(0)
        self.z_slider.setMaximum(200)
        self.z_slider.setValue(0)
        self.z_slider.valueChanged.connect(self.update_z)
        zl.addWidget(self.z_label)
        zl.addWidget(self.z_slider)
        lay.addWidget(zbox)

        btns = QHBoxLayout()
        b1 = QPushButton("Generate + Simulate")
        b1.clicked.connect(self.generate)
        b2 = QPushButton("Save SLM PNG")
        b2.clicked.connect(self.save_png)
        b3 = QPushButton("Save Report")
        b3.clicked.connect(self.save_report)
        btns.addWidget(b1)
        btns.addWidget(b2)
        btns.addWidget(b3)
        lay.addLayout(btns)

        lay.addStretch()
        self.tabs.addTab(tab, "Design")

    def build_report_tab(self):
        tab = QWidget()
        lay = QVBoxLayout(tab)
        self.report = QTextEdit()
        self.report.setReadOnly(True)
        self.report.setText(
            "MANTIS Multi-Bessel Research Suite V6.1\n\n"
            "This is the practical stop-point before experimental calibration.\n"
            "It detects FFT peaks and compares them with expected diffraction orders.\n"
        )
        lay.addWidget(self.report)
        self.tabs.addTab(tab, "Report")

    def rf(self, name):
        return float(self.fields[name].text())

    def ri(self, name):
        return int(float(self.fields[name].text()))

    def params(self):
        return {
            "slm_width": self.ri("SLM width px"),
            "slm_height": self.ri("SLM height px"),
            "pixel_size_um": self.rf("SLM pixel size µm"),
            "wavelength_nm": self.rf("Wavelength nm"),
            "beam_diameter_mm": self.rf("Gaussian beam diameter mm"),
            "objective_focal_mm": self.rf("Objective focal length mm"),
            "objective_NA": self.rf("Objective NA"),
            "axicon_angle_deg": self.rf("Axicon angle deg"),
            "axicon_n": self.rf("Axicon refractive index"),
            "bessel_diameter_um": self.rf("Expected Bessel diameter µm"),
            "zmax_um": self.rf("Bessel zmax estimate µm"),
            "bessel_envelope_factor": self.rf("Bessel envelope factor"),
            "sample_window_um": self.rf("Sample window µm"),
            "sample_pixels": self.ri("Sample simulation pixels"),
            "beam_shape": self.shape_box.currentText(),
            "line_direction": self.line_direction_box.currentText(),
            "line_angle_deg": self.rf("Line angle deg"),
            "number_of_beams": self.ri("Number of beams"),
            "rows": self.ri("Rows"),
            "cols": self.ri("Columns"),
            "spacing_um": self.rf("Center spacing µm"),
            "include_zero_for_even": self.include_zero_check.isChecked(),
            "fft_peak_threshold": self.rf("FFT peak threshold"),
            "fft_peak_distance": self.rf("FFT peak distance px"),
            "fft_match_tolerance": self.rf("FFT match tolerance px"),
            "use_vortex": self.vortex_check.isChecked(),
            "vortex_charge": self.ri("Vortex charge"),
        }

    def generate(self):
        try:
            self.result = generate_design(self.params(), float(self.z_slider.value()))
            self.plot.show_results(self.result)
            self.update_report()
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))

    def update_z(self):
        self.z_label.setText(f"z = {self.z_slider.value()} µm")
        if self.result is not None:
            try:
                self.result = generate_design(self.params(), float(self.z_slider.value()))
                self.plot.show_results(self.result)
                self.update_report()
            except Exception:
                pass

    def update_report(self):
        r = self.result
        p = r["p"]

        warnings = []
        if r["period_pixels"] < 4:
            warnings.append("Grating period is very small; SLM sampling may be poor.")
        if r["period_pixels"] > p["slm_width"]:
            warnings.append("Grating period is larger than SLM width; requested spacing may be too small.")
        if p["spacing_um"] < 1.5 * p["bessel_diameter_um"]:
            warnings.append("Center spacing is close to Bessel diameter; overlap is likely.")
        if r["accepted_count"] < r["beam_count"]:
            warnings.append("Some orders exceed objective NA and are clipped.")
        if r["matched_count"] < r["beam_count"]:
            warnings.append("Not all expected orders were matched to FFT peaks. Adjust threshold/distance or check CGH.")

        max_sin = max([d["sin_total"] for d in r["order_data"]]) if r["order_data"] else 0.0

        text = (
            "Generation successful.\n\n"
            f"Beam shape: {p['beam_shape']}\n"
            f"Line direction: {p['line_direction']}\n"
            f"Line angle: {p['line_angle_deg']:.2f}°\n"
            f"Orders X: {r['orders_x']}\n"
            f"Orders Y: {r['orders_y']}\n"
            f"Designed beam count: {r['beam_count']}\n"
            f"Objective-accepted beam count: {r['accepted_count']}\n"
            f"FFT detected peaks: {len(r['detected_peaks'])}\n"
            f"Expected orders matched to FFT peaks: {r['matched_count']} / {r['beam_count']}\n"
            f"Requested spacing: {p['spacing_um']:.3f} µm\n"
            f"Nearest computed spacing: {r['nearest_spacing_um']:.3f} µm\n"
            f"Grating period: {r['period_pixels']:.2f} pixels = {r['period_um']:.2f} µm\n"
            f"Max diffraction sin(theta): {max_sin:.5f}\n"
            f"Objective NA: {p['objective_NA']:.3f}\n"
            f"Gaussian beam on SLM: {p['beam_diameter_mm']:.2f} mm\n"
            f"Objective focal length: {p['objective_focal_mm']:.2f} mm\n"
            f"Axicon physical angle: {p['axicon_angle_deg']:.3f}°\n"
            f"Axicon cone angle: {r['cone_angle_deg']:.4f}°\n"
            f"Expected Bessel diameter: {p['bessel_diameter_um']:.2f} µm\n"
            f"z preview: {r['z_um']:.1f} µm\n"
            f"Vortex enabled: {p['use_vortex']}\n\n"
        )

        if warnings:
            text += "Warnings:\n" + "\n".join(f"- {w}" for w in warnings) + "\n\n"
        else:
            text += "No major warnings.\n\n"

        text += (
            "Stop criterion:\n"
            "If the saved SLM mask experimentally gives the correct number of beams, spacing, and approximate Bessel diameter, stop changing the generator. "
            "The next step is calibration against camera images, not more theoretical versions.\n\n"
            "Interpretation:\n"
            "- Cyan circles = expected diffraction orders.\n"
            "- Green x marks = FFT-detected peaks.\n"
            "- Sample panel = scalar ASM-style multi-Bessel field prediction.\n"
        )

        self.report.setText(text)

    def save_png(self):
        if self.result is None:
            QMessageBox.warning(self, "No result", "Generate first.")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Save SLM mask", "MANTIS_V6_1_SLM_mask.png", "PNG files (*.png)"
        )
        if path:
            Image.fromarray(self.result["mask8"]).save(path)
            QMessageBox.information(self, "Saved", f"Saved:\n{path}")

    def save_report(self):
        if self.result is None:
            QMessageBox.warning(self, "No result", "Generate first.")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Save report", "MANTIS_V6_1_report.txt", "Text files (*.txt)"
        )
        if path:
            with open(path, "w", encoding="utf-8") as f:
                f.write(self.report.toPlainText())
            QMessageBox.information(self, "Saved", f"Saved:\n{path}")


def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
