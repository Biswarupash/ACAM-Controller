#!/usr/bin/env python3

# This is our code (Lucas & Biswarup) for the project.
# This code is working and the outputs are stored in a different directory

import argparse
import json
import math
import os
import sys

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# =============================================================================
# CONFIGURATION
# =============================================================================
DEFAULT_CONFIG = {
    "data_mode": None,              # "random"/"mock" or "genuine"/"real"
    "xfoil_csv": "xfoil_polars.csv",
    "flight_csv": None,             # optional recorded flight: t,cas,h,n,W,phase
    "S": 16.0,                      # wing area, m^2
    "b": None,                      # span, m (needed for drag comparison)
    "W": 10000.0,                   # weight, N
    "dt": 0.1,                      # control period, s
    "tau": 1.0,                     # smoothing time constant, s
    "rate": 2.0,                    # max camber rate, % per second
    "deadband": 0.1,                # ignore changes smaller than this, %
    "cal_camber": [0, 2, 4, 6, 8],  # PLACEHOLDER calibration (Step 6)
    "cal_angle": [10, 35, 60, 85, 110],
    "serial_port": None,            # e.g. "COM5" or "/dev/ttyUSB0"
    "baud": 115200,
    "use_ai": False,
    "ai_samples": 100000,
    "ai_model_path": None,          # load an existing model instead of training
    "airfoil_thickness": 0.12,      # for the shape plot (NACA 4-digit style)
    "camber_position": 0.4,         # position of max camber, fraction of chord
    "output_dir": "acam_output",
    "verbose": False,               # print every M,x.xx message
}

MOCK_WORDS = {"random", "mock", "fake", "test", "placeholder", "simulated", "dummy"}
REAL_WORDS = {"genuine", "real", "proper", "xfoil", "actual", "measured"}

RHO0 = 1.225
K = {"takeoff": 1.2, "cruise": 1.3, "landing": 1.3}   # stall safety factors
RATE_PEN = 5e-4
SAFE_DEFAULT = 4.0       # % camber when inputs are lost > 2 s
STALE_LIMIT = 0.5        # s: input older than this is treated as bad
HOLD_LIMIT = 2.0         # s: hold last camber this long, then go to default
M_MIN, M_MAX = 0.0, 8.0


def normalise_mode(mode):
    """Return 'mock' or 'real' from whatever the user typed, or None."""
    if mode is None:
        return None
    word = str(mode).strip().lower()
    if word in MOCK_WORDS:
        return "mock"
    if word in REAL_WORDS:
        return "real"
    return None


def ask_mode():
    while True:
        ans = input("Is the wing data a RANDOM (mock) case or GENUINE (real XFOIL) data? "
                    "[random/genuine]: ")
        mode = normalise_mode(ans)
        if mode:
            return mode
        print("  Please type 'random' or 'genuine'.")


# =============================================================================
# STEP 1 - AIR DENSITY AND TRUE SPEED
# =============================================================================
def air_density(h):
    """ISA density (kg/m^3) for altitude h (m), valid below 11 km."""
    return RHO0 * (1 - 2.25577e-5 * h) ** 4.25588


def true_speed(cas, rho):
    """True airspeed from calibrated airspeed (valid below ~100 m/s)."""
    return cas * (RHO0 / rho) ** 0.5


# =============================================================================
# STEP 2 - REQUIRED LIFT COEFFICIENT
# =============================================================================
def required_cl(n, W, rho, V, S):
    return 2 * n * W / (rho * V ** 2 * S)


# =============================================================================
# STEP 3 - WING DATA (MOCK OR REAL) AND TABLE LOADER
# =============================================================================
def make_mock_polars():
    """FAKE polars. Never use results from these for real conclusions."""
    rows = []
    for m in [0, 2, 4, 6, 8]:
        clmax = 1.2 + 0.08 * m
        for a in np.arange(-4, 16.01, 0.5):
            cl = min(0.11 * (a + m), clmax)
            cd = 0.006 + 0.0003 * m + 0.008 * (cl - 0.1 * m) ** 2
            rows.append((m, a, cl, cd, 0.0))
    return pd.DataFrame(rows, columns=["m", "alpha_deg", "cl", "cd", "cm"])


def build_tables(df):
    """Per-camber tables, kept only up to the first point where lift stops rising."""
    tables = {}
    for m, g in df.groupby("m"):
        g = g.sort_values("alpha_deg")
        cl = g.cl.values
        drops = np.where(np.diff(cl) <= 0)[0]
        i_stall = int(drops[0]) if len(drops) else len(cl) - 1
        g = g.iloc[: i_stall + 1]
        if len(g) < 2:
            raise ValueError(f"Camber m={m}: fewer than 2 usable pre-stall points.")
        tables[float(m)] = {"cl": g.cl.values.astype(float),
                            "cd": g.cd.values.astype(float),
                            "cl_max": float(g.cl.values[-1])}
    if len(tables) < 2:
        raise ValueError("Need polars for at least 2 different cambers.")
    return tables


def load_real_polars(path, out_dir):
    """STEP 10 - load and validate the genuine XFOIL CSV."""
    if not os.path.isfile(path):
        sys.exit(f"ERROR: data mode is GENUINE but XFOIL file '{path}' was not found.\n"
                 f"       Provide it with --xfoil, or run with --data-mode random.")
    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]
    need = {"m", "alpha_deg", "cl", "cd"}
    missing = need - set(df.columns)
    if missing:
        sys.exit(f"ERROR: XFOIL file is missing columns {sorted(missing)}. "
                 f"Expected: m,alpha_deg,cl,cd,cm")
    if "cm" not in df.columns:
        df["cm"] = 0.0
    n_before = len(df)
    df = df.dropna(subset=list(need))
    if len(df) < n_before:
        print(f"  note: dropped {n_before - len(df)} rows with missing values")
    df = df[df.cd > 0]

    tables = build_tables(df)
    print("XFOIL data check:")
    warnings = []
    for m, t in sorted(tables.items()):
        inc = bool(np.all(np.diff(t["cl"]) > 0))
        print(f"  m={m:4.1f}: points={len(t['cl']):3d}  cl_max={t['cl_max']:.2f}  "
              f"cl strictly increasing: {inc}")
        if not (1.0 <= t["cl_max"] <= 2.2):
            warnings.append(f"m={m}: cl_max {t['cl_max']:.2f} outside 1.0-2.2 (ask Lucas)")
    for w in warnings:
        print("  WARNING:", w)

    fig, ax = plt.subplots(figsize=(7, 5))
    for m, t in sorted(tables.items()):
        ax.plot(t["cd"], t["cl"], marker=".", label=f"m = {m:g}%")
    ax.set_xlabel("cd"); ax.set_ylabel("cl"); ax.set_title("XFOIL polars (pre-stall)")
    ax.grid(alpha=0.3); ax.legend()
    fig.tight_layout(); fig.savefig(os.path.join(out_dir, "polars_check.png"), dpi=130)
    plt.close(fig)
    return tables


# =============================================================================
# STEP 4 - PICK THE CAMBER
# =============================================================================
def make_grid(tables):
    ms = sorted(tables)
    lo, hi = max(M_MIN, ms[0]), min(M_MAX, ms[-1])
    return np.round(np.arange(lo, hi + 1e-4, 0.1), 3)


def pick_camber(CL, phase, m_prev, tables, grid=None):
    ms = sorted(tables)
    grid = make_grid(tables) if grid is None else grid
    cd_at = [np.interp(CL, tables[m]["cl"], tables[m]["cd"]) for m in ms]
    clmax = [tables[m]["cl_max"] for m in ms]
    cd_g = np.interp(grid, ms, cd_at)
    clmax_g = np.interp(grid, ms, clmax)
    ok = CL <= clmax_g / K[phase] ** 2
    if not ok.any():
        return float(grid[np.argmax(clmax_g)]), "STALL_WARNING"
    cost = cd_g + RATE_PEN * ((grid - m_prev) / 8.0) ** 2
    cost[~ok] = np.inf
    return float(grid[np.argmin(cost)]), "OK"


# =============================================================================
# STEP 5 - SMOOTH AND LIMIT
# =============================================================================
def shape_command_ex(m_cmd, m_prev, dt, tau=1.0, rate=2.0, deadband=0.1):
    """Returns (new camber, rate_limited?)."""
    if abs(m_cmd - m_prev) < deadband:
        return m_prev, False
    beta = dt / (tau + dt)
    m_f = m_prev + beta * (m_cmd - m_prev)
    raw = m_f - m_prev
    step = np.clip(raw, -rate * dt, rate * dt)
    return float(np.clip(m_prev + step, M_MIN, M_MAX)), bool(abs(raw) > rate * dt)


def shape_command(m_cmd, m_prev, dt, tau=1.0, rate=2.0, deadband=0.1):
    return shape_command_ex(m_cmd, m_prev, dt, tau, rate, deadband)[0]


# =============================================================================
# STEP 6 / 13 - CAMBER TO SERVO AND MESSAGE LINK
# =============================================================================
class ServoLink:
    def __init__(self, cal_camber, cal_angle, port=None, baud=115200, verbose=False):
        order = np.argsort(cal_camber)
        self.cal_camber = np.asarray(cal_camber, float)[order]
        self.cal_angle = np.asarray(cal_angle, float)[order]
        self.verbose = verbose
        self.messages = []
        self.ser = None
        if port:
            try:
                import serial
            except ImportError:
                sys.exit("ERROR: pyserial not installed (pip install pyserial).")
            self.ser = serial.Serial(port, baud, timeout=1)
            print(f"Serial link open on {port} @ {baud}")

    def camber_to_servo_deg(self, m):
        return float(np.interp(m, self.cal_camber, self.cal_angle))

    @staticmethod
    def servo_deg_to_pulse_us(deg):
        return 500 + 2000 * deg / 180.0      # check the servo's datasheet

    def send_camber(self, m):
        msg = f"M,{m:.2f}"
        self.messages.append(msg)
        if self.ser is not None:
            self.ser.write((msg + "\n").encode())
        elif self.verbose:
            print(msg)

    def bench_sweep(self, dt=0.1, rate=0.5):
        """Step 13 first hardware test: slow sweep 0 -> 8 -> 0."""
        import time
        seq = list(np.arange(0, 8.0001, rate * dt)) + list(np.arange(8, -0.0001, -rate * dt))
        for m in seq:
            self.send_camber(float(m))
            time.sleep(dt)

    def close(self):
        if self.ser is not None:
            self.ser.close()


# =============================================================================
# STEP 7 / 11 - FLIGHT DATA SOURCES
# =============================================================================
def fake_flight(t, W=10000.0):
    if t < 60:
        return np.interp(t, [0, 60], [35, 60]), np.interp(t, [0, 60], [0, 500]), 1.0, W, "takeoff"
    if t < 240:
        return 65.0, 500.0, 1.0, W, "cruise"
    if t < 300:
        return np.interp(t, [240, 300], [65, 35]), np.interp(t, [240, 300], [500, 0]), 1.0, W, "landing"
    return None


def simulated_samples(dt, W):
    t = 0.0
    while True:
        d = fake_flight(t, W)
        if d is None:
            return
        cas, h, n, Wt, phase = d
        yield t, {"cas": cas, "h": h, "n": n, "W": Wt, "phase": phase, "time_measured": t}
        t = round(t + dt, 6)


def csv_samples(path, W_default):
    """read_flight_data(): recorded flight CSV in SI units.
    Required columns: t, cas (m/s), h (m), phase. Optional: n (g), W (N), time_measured (s)."""
    if not os.path.isfile(path):
        sys.exit(f"ERROR: flight CSV '{path}' not found.")
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    for c in ["t", "cas", "h", "phase"]:
        if c not in df.columns:
            sys.exit(f"ERROR: flight CSV needs column '{c}' (have {list(df.columns)})")
    for _, r in df.sort_values("t").iterrows():
        yield float(r.t), {
            "cas": r.cas, "h": r.h,
            "n": r.n if "n" in df.columns else 1.0,
            "W": r.W if "W" in df.columns else W_default,
            "phase": str(r.phase).strip().lower(),
            "time_measured": r.time_measured if "time_measured" in df.columns else r.t,
        }


# =============================================================================
# STEP 8 / 12 - AI PIPELINE
# =============================================================================
def make_training_data(tables, n_samples=100000, seed=0):
    rng = np.random.default_rng(seed)
    PHASES = ["takeoff", "cruise", "landing"]
    grid = make_grid(tables)
    X, y = [], []
    for _ in range(n_samples):
        CL = rng.uniform(0.1, 1.6)
        phase = PHASES[rng.integers(3)]
        m_prev = rng.uniform(0.0, 8.0)
        m_opt, _ = pick_camber(CL, phase, m_prev, tables, grid)
        X.append([CL, phase == "takeoff", phase == "cruise", phase == "landing", m_prev])
        y.append(m_opt)
    return np.array(X, dtype=float), np.array(y)


def train_ai(X, y):
    from sklearn.neural_network import MLPRegressor
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    Xtr, Xva, ytr, yva = train_test_split(X, y, test_size=0.2, random_state=0)
    scaler = StandardScaler().fit(Xtr)
    net = MLPRegressor(hidden_layer_sizes=(64, 64, 64), max_iter=500,
                       early_stopping=True, random_state=0)
    net.fit(scaler.transform(Xtr), ytr)
    err = float(np.abs(net.predict(scaler.transform(Xva)) - yva).mean())
    print(f"  AI mean validation error: {err:.3f} % camber (goal < 0.25)")
    return net, scaler, err


def ai_camber(CL, phase, m_prev, tables, net, scaler, grid=None):
    m_phys, flag = pick_camber(CL, phase, m_prev, tables, grid)   # always computed
    x = np.array([[CL, phase == "takeoff", phase == "cruise",
                   phase == "landing", m_prev]], dtype=float)
    m_ai = float(net.predict(scaler.transform(x))[0])
    in_range = 0.1 <= CL <= 1.6
    agrees = abs(m_ai - m_phys) < 1.5
    if in_range and agrees and flag == "OK":
        return float(np.clip(m_ai, M_MIN, M_MAX)), flag, "AI"
    return m_phys, flag, "FALLBACK"


# =============================================================================
# CONTROLLER (Steps 5 failure rules + Step 7 loop body)
# =============================================================================
class Controller:
    def __init__(self, tables, S, cfg, ai=None, m_init=SAFE_DEFAULT, t0=0.0):
        self.tables, self.S, self.cfg, self.ai = tables, S, cfg, ai
        self.grid = make_grid(tables)
        self.m_out = m_init
        self.last_good = t0

    @staticmethod
    def _valid(sample, t):
        if sample is None:
            return False
        try:
            vals = [float(sample[k]) for k in ("cas", "h", "n", "W")]
        except (KeyError, TypeError, ValueError):
            return False
        if not all(math.isfinite(v) for v in vals):
            return False
        if vals[0] <= 0 or vals[3] <= 0:
            return False
        if sample.get("phase") not in K:
            return False
        tm = sample.get("time_measured", t)
        if tm is None or not math.isfinite(float(tm)) or t - float(tm) > STALE_LIMIT:
            return False
        return True

    def step(self, t, sample, dt):
        row = {"t": t, "phase": None, "cas": np.nan, "h": np.nan, "n": np.nan, "W": np.nan,
               "rho": np.nan, "V": np.nan, "CL_req": np.nan}
        if self._valid(sample, t):
            cas, h, n, W = (float(sample[k]) for k in ("cas", "h", "n", "W"))
            phase = sample["phase"]
            rho = air_density(h)
            V = true_speed(cas, rho)
            CL = required_cl(n, W, rho, V, self.S)
            if self.ai is not None:
                m_cmd, flag, source = ai_camber(CL, phase, self.m_out, self.tables,
                                                self.ai[0], self.ai[1], self.grid)
            else:
                m_cmd, flag = pick_camber(CL, phase, self.m_out, self.tables, self.grid)
                source = "PHYSICS"
            self.last_good = t
            row.update(phase=phase, cas=cas, h=h, n=n, W=W, rho=rho, V=V, CL_req=CL)
        else:
            if sample is not None:
                row["phase"] = sample.get("phase")
            if t - self.last_good < HOLD_LIMIT:
                m_cmd, flag = self.m_out, "HOLD_BAD_INPUT"
            else:
                m_cmd, flag = SAFE_DEFAULT, "SAFE_DEFAULT"
            source = "NONE"

        m_new, limited = shape_command_ex(m_cmd, self.m_out, dt, self.cfg["tau"],
                                          self.cfg["rate"], self.cfg["deadband"])
        self.m_out = m_new
        row.update(m_cmd=m_cmd, m_out=m_new, flag=flag, source=source, rate_limited=limited)
        return row


def run_flight(tables, cfg, samples, servo, ai=None):
    ctrl = Controller(tables, cfg["S"], cfg, ai=ai)
    log, t_prev = [], None
    for t, sample in samples:
        dt = cfg["dt"] if t_prev is None else max(t - t_prev, 1e-3)
        row = ctrl.step(t, sample, dt)
        deg = servo.camber_to_servo_deg(row["m_out"])
        row["servo_deg"] = deg
        row["pulse_us"] = servo.servo_deg_to_pulse_us(deg)
        servo.send_camber(row["m_out"])
        log.append(row)
        t_prev = t
    return pd.DataFrame(log)


# =============================================================================
# CAMBER SHAPE (the physical output: what the wing section looks like)
# =============================================================================
def camber_shape(m_pct, p=0.4, t=0.12, n=121):
    """NACA 4-digit style section for camber m_pct (% chord).
    Returns x, camber line yc, upper (xu, yu), lower (xl, yl), all in chord units."""
    beta = np.linspace(0, np.pi, n)
    x = 0.5 * (1 - np.cos(beta))
    m = m_pct / 100.0
    yc = np.where(x < p, m / p ** 2 * (2 * p * x - x ** 2),
                  m / (1 - p) ** 2 * ((1 - 2 * p) + 2 * p * x - x ** 2))
    dyc = np.where(x < p, 2 * m / p ** 2 * (p - x), 2 * m / (1 - p) ** 2 * (p - x))
    yt = 5 * t * (0.2969 * np.sqrt(x) - 0.1260 * x - 0.3516 * x ** 2
                  + 0.2843 * x ** 3 - 0.1036 * x ** 4)
    th = np.arctan(dyc)
    return (x, yc, x - yt * np.sin(th), yc + yt * np.cos(th),
            x + yt * np.sin(th), yc - yt * np.cos(th))


def shape_outputs(log, cfg, out_dir, mode):
    good = log.dropna(subset=["phase"])
    shapes = {}
    for ph in ["takeoff", "cruise", "landing"]:
        sel = good[good.phase == ph]
        if len(sel):
            shapes[ph] = float(sel.m_out.median())
    shapes["final"] = float(log.m_out.iloc[-1])

    p, th = cfg["camber_position"], cfg["airfoil_thickness"]
    colors = {"takeoff": "tab:orange", "cruise": "tab:blue",
              "landing": "tab:green", "final": "tab:gray"}
    fig, ax = plt.subplots(figsize=(9, 4))
    coords = {}
    for name, m in shapes.items():
        x, yc, xu, yu, xl, yl = camber_shape(m, p, th)
        ls = "--" if name == "final" else "-"
        ax.plot(np.r_[xu[::-1], xl[1:]], np.r_[yu[::-1], yl[1:]], ls, color=colors[name],
                label=f"{name}: camber {m:.2f}%")
        ax.plot(x, yc, ":", color=colors[name], lw=1)
        coords[f"{name}_yc"], coords[f"{name}_yu"], coords[f"{name}_yl"] = yc, yu, yl
        coords["x"] = x
    ax.set_aspect("equal"); ax.grid(alpha=0.3)
    ax.set_xlabel("x / chord"); ax.set_ylabel("y / chord")
    ax.set_title(f"Wing section shape per flight phase ({mode.upper()} data)")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(out_dir, "airfoil_shapes.png"), dpi=140)
    plt.close(fig)
    cols = ["x"] + [k for k in coords if k != "x"]
    pd.DataFrame({k: coords[k] for k in cols}).to_csv(
        os.path.join(out_dir, "airfoil_shape_coords.csv"), index=False)
    return shapes


def plot_camber_time(log, out_dir, mode):
    fig, ax = plt.subplots(figsize=(10, 4.5))
    shade = {"takeoff": "tab:orange", "cruise": "tab:blue", "landing": "tab:green"}
    ph = log.phase.fillna("none").values
    t = log.t.values
    start = 0
    for i in range(1, len(ph) + 1):
        if i == len(ph) or ph[i] != ph[start]:
            if ph[start] in shade:
                ax.axvspan(t[start], t[i - 1], color=shade[ph[start]], alpha=0.12,
                           label=ph[start] if ph[start] not in ax.get_legend_handles_labels()[1] else None)
            start = i
    ax.plot(t, log.m_cmd, color="0.6", lw=1, label="raw command")
    ax.plot(t, log.m_out, color="k", lw=2, label="final camber (to servo)")
    bad = log[log.flag != "OK"]
    if len(bad):
        ax.scatter(bad.t, bad.m_out, color="red", s=8, label="non-OK flag", zorder=5)
    ax.set_xlabel("time (s)"); ax.set_ylabel("camber (% chord)"); ax.set_ylim(-0.2, 8.2)
    ax.set_title(f"ACAM camber vs time ({mode.upper()} data)")
    ax.grid(alpha=0.3); ax.legend(loc="upper center", ncol=6, fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(out_dir, "camber_vs_time.png"), dpi=140)
    plt.close(fig)


# =============================================================================
# STEP 14 - DRAG COMPARISON
# =============================================================================
def total_cd(CL, m, tables, AR, e=0.85):
    ms = sorted(tables)
    cd_at = [np.interp(CL, tables[k]["cl"], tables[k]["cd"]) for k in ms]
    return float(np.interp(m, ms, cd_at)) + CL ** 2 / (np.pi * e * AR)


def compare_fixed(log, tables, AR):
    good = log.dropna(subset=["CL_req"])
    CLs, ms, phases = good.CL_req.values, good.m_out.values, good.phase.values
    cd_ad = np.mean([total_cd(c, m, tables, AR) for c, m in zip(CLs, ms)])
    keys = sorted(tables)
    rows = []
    for mf in [0, 2, 4, 6, 8]:
        if mf < keys[0] or mf > keys[-1]:
            continue
        cd_fx = np.mean([total_cd(c, mf, tables, AR) for c in CLs])
        clmax_f = np.interp(mf, keys, [tables[k]["cl_max"] for k in keys])
        viol = int(sum(c > clmax_f / K[p] ** 2 for c, p in zip(CLs, phases)))
        rows.append({"fixed_camber": mf, "mean_CD_fixed": cd_fx, "mean_CD_adaptive": cd_ad,
                     "adaptive_saves_pct": 100 * (1 - cd_ad / cd_fx),
                     "stall_margin_violations": viol})
    return pd.DataFrame(rows)


# =============================================================================
# STEP 9 - TEST CHECKLIST
# =============================================================================
def run_checklist(tables, cfg, ai=None, ai_err=None, phys_log=None, ai_log=None):
    res = []

    def add(name, expected, got, ok):
        res.append({"test": name, "expected": expected, "got": got, "result": "PASS" if ok else "FAIL"})

    r0, r1 = air_density(0), air_density(3048)
    add("air_density(0)", "1.225", f"{r0:.4f}", abs(r0 - 1.225) < 1e-6)
    add("air_density(3048)", "~0.905", f"{r1:.4f}", abs(r1 - 0.905) < 0.005)
    v = true_speed(100, r1)
    add("true_speed(100, rho(3048))", "~116", f"{v:.2f}", abs(v - 116) < 1.0)
    c1, c2 = required_cl(1, 10000, 1.225, 50, 16), required_cl(2, 10000, 1.225, 50, 16)
    add("required_cl(1,...)", "0.408", f"{c1:.3f}", abs(c1 - 0.408) < 1e-3)
    add("required_cl(2,...)", "0.816", f"{c2:.3f}", abs(c2 - 0.816) < 1e-3)

    mono = True
    for ph in K:
        prev_m, prev_out = 0.0, -1
        for CL in np.linspace(0.2, 1.0, 81):
            m, _ = pick_camber(CL, ph, prev_m, tables)
            if m < prev_out - 1e-9:
                mono = False
            prev_out = prev_m = m
    add("pick_camber sweep CL 0.2->1.0", "never decreases", "monotonic" if mono else "decreases", mono)

    dt, m, maxstep = cfg["dt"], 0.0, 0.0
    for _ in range(200):
        new = shape_command(8.0, m, dt, cfg["tau"], cfg["rate"], cfg["deadband"])
        maxstep = max(maxstep, abs(new - m)); m = new
    add("step 0->8 through shape_command", f"<= {cfg['rate']*dt:.2f} per step",
        f"max {maxstep:.3f}", maxstep <= cfg["rate"] * dt + 1e-9)

    ctrl = Controller(tables, cfg["S"], cfg, m_init=6.0, t0=0.0)
    good = {"cas": 50, "h": 0, "n": 1, "W": 10000, "phase": "cruise", "time_measured": 0.0}
    ctrl.step(0.0, good, dt)
    m_hold = ctrl.m_out
    held, t = True, 0.0
    for _ in range(19):
        t = round(t + dt, 6)
        ctrl.step(t, dict(good, cas=float("nan"), time_measured=t), dt)
        held &= abs(ctrl.m_out - m_hold) < 1e-12
    for _ in range(300):
        t = round(t + dt, 6)
        r = ctrl.step(t, dict(good, cas=float("nan"), time_measured=t), dt)
    ok = held and r["flag"] == "SAFE_DEFAULT" and abs(ctrl.m_out - SAFE_DEFAULT) <= cfg["deadband"]
    add("NaN airspeed input", "held, then -> 4 after 2 s",
        f"held={held}, final={ctrl.m_out:.2f}", ok)

    if phys_log is not None:
        n_stall = int((phys_log.flag == "STALL_WARNING").sum())
        rmax = float(np.max(np.abs(np.diff(phys_log.m_out)) / np.diff(phys_log.t)))
        add("simulated flight", "no STALL_WARNING", f"{n_stall} warnings, max rate {rmax:.2f}%/s",
            n_stall == 0)
    if ai is not None and ai_log is not None and phys_log is not None:
        diff = float(np.abs(ai_log.m_out.values - phys_log.m_out.values).mean())
        add("AI loop", "err < 0.25, matches physics",
            f"train err {ai_err:.3f}, loop diff {diff:.3f}", ai_err < 0.25 and diff < 0.25)
    return pd.DataFrame(res)


# =============================================================================
# MAIN
# =============================================================================
def parse_args():
    ap = argparse.ArgumentParser(description="ACAM adaptive camber controller")
    ap.add_argument("--config", help="JSON config file (any keys of DEFAULT_CONFIG)")
    ap.add_argument("--data-mode", help="random/mock  or  genuine/real")
    ap.add_argument("--xfoil", help="XFOIL polars CSV (genuine mode)")
    ap.add_argument("--flight", help="recorded flight CSV (t,cas,h,n,W,phase)")
    ap.add_argument("--S", type=float); ap.add_argument("--b", type=float)
    ap.add_argument("--W", type=float)
    ap.add_argument("--ai", action="store_true", help="enable the neural-network controller")
    ap.add_argument("--ai-samples", type=int)
    ap.add_argument("--serial-port"); ap.add_argument("--bench-sweep", action="store_true")
    ap.add_argument("--out"); ap.add_argument("--verbose", action="store_true")
    return ap.parse_args()


def build_config(args):
    cfg = dict(DEFAULT_CONFIG)
    user_keys = set()
    if args.config:
        with open(args.config) as f:
            file_cfg = json.load(f)
        cfg.update(file_cfg); user_keys |= set(file_cfg)
    cli = {"data_mode": args.data_mode, "xfoil_csv": args.xfoil, "flight_csv": args.flight,
           "S": args.S, "b": args.b, "W": args.W, "ai_samples": args.ai_samples,
           "serial_port": args.serial_port, "output_dir": args.out}
    for k, v in cli.items():
        if v is not None:
            cfg[k] = v; user_keys.add(k)
    if args.ai:
        cfg["use_ai"] = True
    if args.verbose:
        cfg["verbose"] = True
    return cfg, user_keys


def main():
    args = parse_args()
    cfg, user_keys = build_config(args)

    # ------------------- DATA MODE CONDITION -------------------
    mode = normalise_mode(cfg["data_mode"])
    if cfg["data_mode"] is not None and mode is None:
        print(f"Unrecognised data_mode '{cfg['data_mode']}'.")
    if mode is None:
        mode = ask_mode()

    out_dir = cfg["output_dir"]
    os.makedirs(out_dir, exist_ok=True)
    print("=" * 66)
    if mode == "mock":
        print(" DATA MODE: RANDOM / MOCK  (placeholder polars, results NOT real)")
        print("=" * 66)
        tables = build_tables(make_mock_polars())
        if cfg["b"] is None:
            cfg["b"] = math.sqrt(7.5 * cfg["S"])     # assumed AR = 7.5 for mock only
    else:
        print(" DATA MODE: GENUINE / REAL  (XFOIL polars)")
        print("=" * 66)
        missing = [k for k in ("S", "W") if k not in user_keys]
        if missing:
            sys.exit(f"ERROR: genuine mode needs the real wing constants {missing} "
                     f"(set them in the config file or with --S/--W).")
        tables = load_real_polars(cfg["xfoil_csv"], out_dir)
        if cfg["cal_camber"] == DEFAULT_CONFIG["cal_camber"] and \
           cfg["cal_angle"] == DEFAULT_CONFIG["cal_angle"]:
            print("  NOTE: servo calibration is still the PLACEHOLDER from Step 6.")
    print(f"  cambers in table: {sorted(tables)}   S={cfg['S']} m^2  W={cfg['W']} N  "
          f"b={cfg['b']}")

    servo = ServoLink(cfg["cal_camber"], cfg["cal_angle"], cfg["serial_port"],
                      cfg["baud"], cfg["verbose"])
    if args.bench_sweep:
        print("Bench sweep 0 -> 8 -> 0 ...")
        servo.bench_sweep(); servo.close(); return

    def samples():
        if cfg["flight_csv"]:
            return csv_samples(cfg["flight_csv"], cfg["W"])
        if mode == "real":
            print("  NOTE: no flight CSV given, using the simulated flight profile.")
        return simulated_samples(cfg["dt"], cfg["W"])

    # ------------------- PHYSICS RUN -------------------
    print("\nRunning physics controller ...")
    phys_log = run_flight(tables, cfg, samples(), servo)
    log, ai, ai_err, ai_log = phys_log, None, None, None

    # ------------------- AI RUN -------------------
    if cfg["use_ai"]:
        import joblib
        if cfg["ai_model_path"] and os.path.isfile(cfg["ai_model_path"]):
            bundle = joblib.load(cfg["ai_model_path"])
            ai, ai_err = (bundle["net"], bundle["scaler"]), bundle.get("err", float("nan"))
            print(f"Loaded AI model from {cfg['ai_model_path']}")
        else:
            print(f"Training AI on {cfg['ai_samples']} samples ({mode} tables) ...")
            X, y = make_training_data(tables, cfg["ai_samples"])
            net, scaler, ai_err = train_ai(X, y)
            ai = (net, scaler)
            joblib.dump({"net": net, "scaler": scaler, "err": ai_err, "data_mode": mode},
                        os.path.join(out_dir, "acam_ai_model.joblib"))
        servo.messages.clear()
        print("Running AI controller (physics fallback active) ...")
        ai_log = run_flight(tables, cfg, samples(), servo, ai=ai)
        log = ai_log
    servo.close()

    # ------------------- OUTPUTS -------------------
    log.to_csv(os.path.join(out_dir, "acam_log.csv"), index=False)
    with open(os.path.join(out_dir, "servo_messages.txt"), "w") as f:
        f.write("\n".join(servo.messages) + "\n")
    plot_camber_time(log, out_dir, mode)
    shapes = shape_outputs(log, cfg, out_dir, mode)

    checklist = run_checklist(tables, cfg, ai, ai_err, phys_log, ai_log)
    checklist.to_csv(os.path.join(out_dir, "checklist.csv"), index=False)

    AR = cfg["b"] ** 2 / cfg["S"] if cfg["b"] else None
    drag = None
    if AR:
        drag = compare_fixed(log, tables, AR)
        drag.to_csv(os.path.join(out_dir, "drag_comparison.csv"), index=False)

    # ------------------- SUMMARY -------------------
    print("\n" + "=" * 66)
    print(" CAMBER SHAPE OUTPUT")
    print("=" * 66)
    for name, m in shapes.items():
        print(f"  {name:8s}: camber = {m:.2f} % chord   servo = "
              f"{servo.camber_to_servo_deg(m):6.1f} deg   message  M,{m:.2f}")
    print("\n  camber range per phase (min / median / max, % chord):")
    for ph, g in log.dropna(subset=["phase"]).groupby("phase", sort=False):
        print(f"    {ph:8s}: {g.m_out.min():.2f} / {g.m_out.median():.2f} / {g.m_out.max():.2f}")
    rate = np.abs(np.diff(log.m_out)) / np.diff(log.t)
    print(f"\n  steps: {len(log)}   max camber rate: {rate.max():.2f} %/s "
          f"(limit {cfg['rate']})")
    print(f"  flags: {log.flag.value_counts().to_dict()}")
    print(f"  sources: {log.source.value_counts().to_dict()}")
    print(f"  rate-limit hits: {int(log.rate_limited.sum())}")
    print("\n Step 9 checklist:")
    print(checklist.to_string(index=False))
    if drag is not None:
        tag = "  (MOCK numbers - not real)" if mode == "mock" else ""
        print(f"\n Drag comparison, AR = {AR:.2f}{tag}:")
        print(drag.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    else:
        print("\n Drag comparison skipped: span b not given.")
    print(f"\n All files written to: {os.path.abspath(out_dir)}")


if __name__ == "__main__":
    main()
