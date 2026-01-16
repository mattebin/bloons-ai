#!/usr/bin/env python3
"""BTD6 Shadow Player (Linux/X11) — hardened automation build.

Includes:
- OFF→ARMED→LIVE autonomous (F9 cycles)
- Window locking (g) + SHADOW window by name
- Window verification before click (anti-Chrome)
- Place→Select→Upgrade pipeline in autonomous
- Retry logic + reason codes + heartbeat
- Profile system for calibrations/config (p / P)
- Mapping drift detector (auto-pauses to ARMED)
- Debug ring buffer + dump on failures (frames + logs + matrices)
- Dry-run visualization markers in ARMED mode
"""

import cv2
import json
import numpy as np
from pathlib import Path
import subprocess
import time
import shutil
from collections import deque
import os

import btd6_read_hud as hud

# ======================
# Config / profiles
# ======================
CFG_ROOT = Path.home() / ".config" / "btd6-ai"
CFG_ROOT.mkdir(parents=True, exist_ok=True)
PROFILES_DIR = CFG_ROOT / "profiles"
PROFILES_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_PROFILE = "default"
# Legacy (non-profile) paths (kept for backward compatibility)
LEGACY_DIR = Path.home() / ".config" / "btd6-ai"
LEGACY_DIR.mkdir(parents=True, exist_ok=True)
LEGACY_CALIB_FILE = LEGACY_DIR / "calib.json"
LEGACY_SCREEN_MAP_FILE = LEGACY_DIR / "screen_map.json"
LEGACY_SPOTS_FILE = LEGACY_DIR / "build_spots.json"
LEGACY_STATS_FILE = LEGACY_DIR / "spot_stats.json"
LEGACY_BLOCKED_FILE = LEGACY_DIR / "blocked_polygons.json"
LEGACY_UI_MAP_FILE = LEGACY_DIR / "ui_map.json"
LEGACY_UI_DESKTOP_FILE = LEGACY_DIR / "ui_desktop.json"


# These are set by set_profile()
PROFILE = None
CFG = None
CALIB_FILE = None
SCREEN_MAP_FILE = None
SPOTS_FILE = None
STATS_FILE = None
BLOCKED_FILE = None
UI_MAP_FILE = None

def _safe_json_load(path: Path, default):
    try:
        if path.exists():
            return json.loads(path.read_text())
    except Exception as e:
        log(f"WARN: failed to load {path}: {e}")
    return default

def _safe_json_save(path: Path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2))
    tmp.replace(path)

def list_profiles():
    profs = []
    for p in sorted(PROFILES_DIR.iterdir()):
        if p.is_dir():
            profs.append(p.name)
    if DEFAULT_PROFILE not in profs:
        (PROFILES_DIR / DEFAULT_PROFILE).mkdir(parents=True, exist_ok=True)
        profs.insert(0, DEFAULT_PROFILE)
    return profs

def set_profile(name: str):
    """Switch active profile; updates file paths and reloads config."""
    global PROFILE, CFG, CALIB_FILE, SCREEN_MAP_FILE, SPOTS_FILE, STATS_FILE, BLOCKED_FILE, UI_MAP_FILE
    global H, S, build_spots, spot_stats, blocked_polygons, UPGRADE_UI

    PROFILE = name
    CFG = PROFILES_DIR / PROFILE
    CFG.mkdir(parents=True, exist_ok=True)

    UI_MAP_FILE = CFG / "ui_map.json"
    UI_DESKTOP_FILE = CFG / "ui_desktop.json"
    CALIB_FILE = CFG / "calib.json"
    SCREEN_MAP_FILE = CFG / "screen_map.json"
    SPOTS_FILE = CFG / "build_spots.json"
    STATS_FILE = CFG / "spot_stats.json"
    BLOCKED_FILE = CFG / "blocked_polygons.json"

    H = load_homography(CALIB_FILE, key="H")
    S = load_homography(SCREEN_MAP_FILE, key="S")
    build_spots = _safe_json_load(SPOTS_FILE, [])
    spot_stats = _safe_json_load(STATS_FILE, {})
    blocked_polygons = _safe_json_load(BLOCKED_FILE, [])
    global UI_DESKTOP
    UI_DESKTOP = _safe_json_load(UI_DESKTOP_FILE, _safe_json_load(LEGACY_UI_DESKTOP_FILE, {}))
    ui_map = _safe_json_load(UI_MAP_FILE, {})
    # Optional override; if empty, we rely on static OBS pixels below
    UPGRADE_UI.update(ui_map.get("UPGRADE_UI", {}))

    log(f"PROFILE: {PROFILE}")
    log(f"  H={'OK' if H is not None else 'None'} S={'OK' if S is not None else 'None'} spots={len(build_spots)} blocked={len(blocked_polygons)}")

# ======================
# Capture / grid settings
# ======================
CAM_INDEX = 0  # /dev/video0 (OBS Virtual Camera)
GRID_W, GRID_H = 32, 18  # policy grid (coarse)

# ======================
# Automation tuning
# ======================
ACTION_COOLDOWN = 0.55
STAGE_COOLDOWN = {
    "PLACE": 0.0,
    "SELECT": 0.10,
    "UPGRADE": 0.15,
}
CLICK_RETRIES = 2
CLICK_RETRY_DELAY = 0.12
HEARTBEAT_INTERVAL = 0.50
DRIFT_CHECK_INTERVAL = 5.0

# Debug ring buffer
RING_FPS = 10
RING_SECONDS = 30
RING_MAX = RING_FPS * RING_SECONDS
DEBUG_DUMPS_DIR = Path.home() / "btd6-ai" / "debug_dumps"
DEBUG_DUMPS_DIR.mkdir(parents=True, exist_ok=True)

# ======================
# Towers / hotkeys
# ======================
TOWER_KEYS = {
    "DART": "q",
    # add more as needed
}

# ======================
# Static OBS pixel upgrade buttons (works when resolution/UI scale fixed)
# ======================
# OBS pixel space: 1280x720
UPGRADE_UI = {
    "PATH1": (1168, 280),
    "PATH2": (1168, 394),
    "PATH3": (1168, 512),
    "SELL":  (1146, 650),
}


# Desktop-space UI click targets (bypass OBS->desktop homography for sidebar/buttons).
# Saved per-profile in ui_desktop.json (and legacy ~/.config/btd6-ai/ui_desktop.json).
UI_DESKTOP = {}  # {"PATH1":[x,y], "PATH2":[x,y], "PATH3":[x,y], "SELL":[x,y]}

# UI capture mode: press U to toggle, then q/w/e/r to record PATH1/2/3/SELL at current mouse position.
ui_capture_mode = False

# ======================
# Globals / state
# ======================
H = None  # grid -> OBS
S = None  # OBS -> desktop

build_spots = []
spot_stats = {}
blocked_polygons = []

# Window IDs (strings as returned by xdotool)
BTD6_WIN = None
SHADOW_WIN = None

# Autonomous states
AUTO_OFF = 0
AUTO_ARMED = 1
AUTO_LIVE = 2
auto_state = AUTO_OFF

# Simple stage machine for LIVE
stage = "IDLE"
stage_deadline = 0.0
pending = None  # dict with fields for current action pipeline

# HUD values (best-effort)
last_hud = {"money": None, "round": None, "lives": None}
last_action = None
last_reason = "INIT"
last_act_ts = 0.0
last_heartbeat_ts = 0.0
last_drift_check_ts = 0.0

# Visual click marker
armed_marker = None  # (x_obs, y_obs, x_scr, y_scr, label, ts)

# Logs for debug dump
LOG_LINES = deque(maxlen=2000)

def log(msg: str):
    ts = time.strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line)
    LOG_LINES.append(line)

# Ring buffer of frames (timestamp, frame_bgr)
RING = deque(maxlen=RING_MAX)
_last_ring_push = 0.0

# ======================
# Math helpers
# ======================
def load_homography(path: Path, key: str):
    data = _safe_json_load(path, None)
    if not data:
        return None
    mat = data.get(key)
    if mat is None:
        return None
    try:
        return np.array(mat, dtype=np.float64)
    except Exception:
        return None

def save_homography(path: Path, key: str, mat):
    data = _safe_json_load(path, {})
    data[key] = np.asarray(mat).tolist()
    _safe_json_save(path, data)

    # Mirror into legacy paths so old tooling continues to work.
    try:
        if key == "H" and path == CALIB_FILE:
            _safe_json_save(LEGACY_CALIB_FILE, data)
        elif key == "S" and path == SCREEN_MAP_FILE:
            _safe_json_save(LEGACY_SCREEN_MAP_FILE, data)
    except Exception:
        pass

def apply_homography(M, x, y):
    pt = np.array([[[float(x), float(y)]]], dtype=np.float64)
    out = cv2.perspectiveTransform(pt, M)[0][0]
    return float(out[0]), float(out[1])

def grid_to_obs(Hm, gx, gy):
    # Grid coords are continuous; map to OBS pixels
    return apply_homography(Hm, gx, gy)

def obs_to_screen(Sm, x_obs, y_obs):
    return apply_homography(Sm, x_obs, y_obs)

def screen_geom():
    # returns (W,H) via xdotool getdisplaygeometry
    try:
        out = subprocess.check_output(["xdotool", "getdisplaygeometry"], text=True).strip().split()
        return int(out[0]), int(out[1])
    except Exception:
        return None

# ======================
# Window targeting
# ======================
def xdotool_out(args):
    return subprocess.check_output(["xdotool", *args], text=True).strip()

def get_active_window_id():
    try:
        return xdotool_out(["getactivewindow"])
    except Exception:
        return None

def get_window_name(win_id):
    try:
        return xdotool_out(["getwindowname", str(win_id)])
    except Exception:
        return ""

def get_window_class(win_id):
    """Best-effort window class without requiring xprop."""
    try:
        # xdotool provides a classname on X11 (no extra deps)
        out = subprocess.check_output(["xdotool", "getwindowclassname", str(win_id)], text=True, stderr=subprocess.DEVNULL)
        return out.strip()
    except Exception:
        try:
            out = subprocess.check_output(["xprop", "-id", str(win_id), "WM_CLASS"], text=True, stderr=subprocess.DEVNULL)
            return out.strip()
        except Exception:
            return ""

EXPECTED_BTD6_NAME_SUBSTR = "Bloons"
EXPECTED_BTD6_WMCLASS = None  # learned after lock; optional

def find_shadow_window():
    # Prefer exact shadow window name
    try:
        out = subprocess.check_output(
            ["xdotool", "search", "--onlyvisible", "--name", "BTD6 SHADOW"],
            text=True
        ).strip().splitlines()
        return out[0].strip() if out else None
    except Exception:
        return None


def refocus_shadow():
    """Bring the OpenCV SHADOW window back to focus (so keybinds keep working)."""
    global SHADOW_WIN
    try:
        if SHADOW_WIN is None:
            SHADOW_WIN = find_shadow_window()
        if SHADOW_WIN is None:
            return False
        subprocess.run(["xdotool", "windowactivate", "--sync", str(SHADOW_WIN)], check=False)
        time.sleep(FOCUS_DELAY)
        return True
    except Exception:
        return False

def find_btd6_window():
    for pat in ("Bloons TD 6", "BloonsTD6", "Bloons"):
        try:
            out = subprocess.check_output(
                ["xdotool", "search", "--onlyvisible", "--name", pat],
                text=True
            ).strip().splitlines()
            if out:
                return out[0].strip()
        except Exception:
            pass
    return None

def is_btd6_window(win_id):
    if not win_id:
        return False
    name = get_window_name(win_id)
    if EXPECTED_BTD6_NAME_SUBSTR and EXPECTED_BTD6_NAME_SUBSTR not in name:
        return False
    if EXPECTED_BTD6_WMCLASS:
        cls = get_window_class(win_id)
        if EXPECTED_BTD6_WMCLASS not in cls:
            return False
    return True

def require_windows(allow_reacquire=True):
    """Ensures SHADOW_WIN is found by name and BTD6_WIN is set + verified."""
    global SHADOW_WIN, BTD6_WIN, last_reason

    if SHADOW_WIN is None:
        SHADOW_WIN = find_shadow_window()

    if BTD6_WIN is None and allow_reacquire:
        BTD6_WIN = find_btd6_window()

    if BTD6_WIN and not is_btd6_window(BTD6_WIN):
        if allow_reacquire:
            log("WIN: BTD6_WIN failed verify; trying reacquire")
            BTD6_WIN = find_btd6_window()
        if BTD6_WIN and not is_btd6_window(BTD6_WIN):
            last_reason = "SKIP_NO_WIN"
            return False

    if not SHADOW_WIN:
        # shadow missing isn't fatal for clicks, but keys won't work; still return True
        SHADOW_WIN = None

    if not BTD6_WIN:
        last_reason = "SKIP_NO_WIN"
        return False

    return True

def focus_window(win_id):
    subprocess.run(["xdotool", "windowactivate", "--sync", str(win_id)], check=False)

# ======================
# Input + click helpers
# ======================


def do_click(x, y, tower_name=None):
    """Focus game, optionally send tower hotkey, click, then refocus shadow.
       Always sends Escape after placement attempts to avoid held-tower state.
       Never raises (returns True/False)."""
    global BTD6_WIN
    if BTD6_WIN is None:
        log("CLICK: BTD6_WIN not set")
        return False

    ok = True
    try:
        subprocess.run(["xdotool", "windowactivate", "--sync", str(BTD6_WIN)], check=False)
        time.sleep(FOCUS_DELAY)
        if tower_name is not None:
            subprocess.run(["xdotool", "key", "--window", str(BTD6_WIN), tower_name], check=False)
            time.sleep(KEY_DELAY)
        subprocess.run(["xdotool", "mousemove", "--window", str(BTD6_WIN), str(int(x)), str(int(y))], check=False)
        subprocess.run(["xdotool", "click", "--window", str(BTD6_WIN), "1"], check=False)
        time.sleep(CLICK_DELAY)
    except Exception as e:
        ok = False
        log(f"CLICK ERROR: {e}")
    finally:
        if tower_name is not None:
            try:
                subprocess.run(["xdotool", "key", "--window", str(BTD6_WIN), "Escape"], check=False)
            except Exception:
                pass
        refocus_shadow()
    return ok
def click_upgrade_path(path_id, retries=CLICK_RETRIES):
    """Click upgrade path button.
    Prefer desktop-calibrated UI coords if present (works even if S is playfield-only),
    otherwise fall back to OBS->desktop mapping.
    """
    global S, UI_DESKTOP
    key = f"PATH{int(path_id)}"
    if key not in UPGRADE_UI:
        log(f"Invalid upgrade path: {path_id}")
        return False

    if isinstance(UI_DESKTOP, dict) and key in UI_DESKTOP:
        x_scr, y_scr = UI_DESKTOP[key]
        return do_click(x_scr, y_scr, tower_name=None, retries=retries)

    if S is None:
        log("ERROR: missing screen map S; run F6 x4")
        return False
    x_obs, y_obs = UPGRADE_UI[key]
    x_scr, y_scr = obs_to_screen(S, x_obs, y_obs)
    return do_click(x_scr, y_scr, tower_name=None, retries=retries)

def save_blocked():
    _safe_json_save(BLOCKED_FILE, blocked_polygons)
    _safe_json_save(LEGACY_BLOCKED_FILE, blocked_polygons)
    log(f"Saved {len(blocked_polygons)} blocked polygons -> {BLOCKED_FILE}")

def save_spots():
    _safe_json_save(SPOTS_FILE, build_spots)
    _safe_json_save(LEGACY_SPOTS_FILE, build_spots)
    log(f"Saved {len(build_spots)} build spots -> {SPOTS_FILE}")

def save_stats():
    _safe_json_save(STATS_FILE, spot_stats)
    _safe_json_save(LEGACY_STATS_FILE, spot_stats)
    log(f"Saved spot stats -> {STATS_FILE}")

def save_ui_desktop():
    global UI_DESKTOP
    _safe_json_save(CFG / "ui_desktop.json", UI_DESKTOP)
    _safe_json_save(LEGACY_UI_DESKTOP_FILE, UI_DESKTOP)
    log(f"Saved UI_DESKTOP -> {CFG / 'ui_desktop.json'}")


# ======================
# Action policy (simple placeholder)
# ======================
def is_point_in_polygon(px, py, poly):
    # poly = [(x,y),...]
    inside = False
    n = len(poly)
    if n < 3:
        return False
    j = n - 1
    for i in range(n):
        xi, yi = poly[i]
        xj, yj = poly[j]
        intersect = ((yi > py) != (yj > py)) and (px < (xj - xi) * (py - yi) / (yj - yi + 1e-9) + xi)
        if intersect:
            inside = not inside
        j = i
    return inside

def is_obs_blocked(x_obs, y_obs):
    for poly in blocked_polygons:
        if is_point_in_polygon(x_obs, y_obs, poly):
            return True
    return False

def pick_action_simple():
    """Return an action dict or None.
    Action: {type:'PLACE', tower:'DART', gx,gy} or {type:'UPGRADE', path:int} etc.
    This is a placeholder policy: place at random saved spot, else center.
    """
    if build_spots:
        gx, gy = build_spots[int(time.time()) % len(build_spots)]
    else:
        gx, gy = GRID_W * 0.5, GRID_H * 0.6
    return {"type": "PLACE", "tower": "DART", "gx": float(gx), "gy": float(gy)}

# ======================
# Calibration utilities
# ======================
_clicks = []
_mouse_xy = (0, 0)
ui_calib_mode = False
blocked_edit_mode = False
_current_poly = []

def on_main_mouse(event, x, y, flags, param):
    """Mouse in SHADOW window is in OBS pixel coords."""
    global _mouse_xy, _current_poly, blocked_polygons, build_spots
    _mouse_xy = (x, y)

    if event != cv2.EVENT_LBUTTONDOWN:
        return

    # Blocked polygon edit mode: click points
    if blocked_edit_mode:
        _current_poly.append((int(x), int(y)))
        log(f"BLOCKED: add point {x},{y} (poly len={len(_current_poly)})")
        return

    # Normal: add a build spot based on current OBS->grid mapping inverse is not available; we store OBS-anchored grid using H inverse.
    if H is None:
        log("INFO: H not set (press c to calibrate grid<->OBS)")
        return

    # We need OBS->grid mapping: inverse of H
    try:
        Hinv = np.linalg.inv(H)
    except Exception:
        log("ERROR: cannot invert H")
        return

    gx, gy = apply_homography(Hinv, x, y)
    build_spots.append([gx, gy])
    log(f"SPOT: added grid ({gx:.2f},{gy:.2f}) from OBS ({x},{y})")

def calibrate_grid_to_obs(frame):
    """Interactive corner clicker: saves 4 OBS points; builds H mapping from grid rect to OBS quad."""
    global H
    clone = frame.copy()
    clicks = []

    def cb(ev, x, y, flags, param):
        nonlocal clicks
        if ev == cv2.EVENT_LBUTTONDOWN:
            clicks.append((int(x), int(y)))

    cv2.namedWindow("CALIBRATE", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("CALIBRATE", cb)

    while True:
        vis = clone.copy()
        for i, (x, y) in enumerate(clicks):
            cv2.circle(vis, (x, y), 6, (0, 255, 0), -1)
            cv2.putText(vis, str(i + 1), (x + 8, y - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.putText(
            vis, "Click playfield corners: TL,TR,BR,BL. S=save R=reset Q=quit",
            (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2
        )

        cv2.imshow("CALIBRATE", vis)
        key = cv2.waitKey(20) & 0xFF
        if key == ord('r'):
            clicks = []
        elif key == ord('q') or key == 27:
            break
        elif key == ord('s'):
            if len(clicks) != 4:
                log("Need 4 clicks.")
                continue
            # grid corners in (gx,gy): TL,TR,BR,BL
            src = np.array([
                [0, 0],
                [GRID_W, 0],
                [GRID_W, GRID_H],
                [0, GRID_H],
            ], dtype=np.float64)
            dst = np.array(clicks, dtype=np.float64)
            Hm, _ = cv2.findHomography(src, dst)
            if Hm is None:
                log("ERROR: could not compute H")
                continue
            H = Hm
            data = _safe_json_load(CALIB_FILE, {})
            data["clicks"] = clicks
            data["H"] = H.tolist()
            _safe_json_save(CALIB_FILE, data)
            _safe_json_save(LEGACY_CALIB_FILE, data)
            log(f"Saved H -> {CALIB_FILE}")
            break

    cv2.destroyWindow("CALIBRATE")

def record_screen_map_point():
    """Record a pair of (OBS point, desktop point) for S mapping.

User workflow: SHADOW focused, hover mouse over matching real-game corner, press F6.
"""
    # Use OBS corner clicks from calib if present
    clicks = _safe_json_load(CALIB_FILE, {}).get("clicks", None)
    if not clicks or len(clicks) != 4:
        log("ERROR: need calib.json with 4 clicks (press c and save) before screen mapping")
        return False

    # determine how many already captured
    data = _safe_json_load(SCREEN_MAP_FILE, {})
    pairs = data.get("pairs", [])
    idx = len(pairs)
    if idx >= 4:
        log("Screen map already has 4 points; press F7 to reset")
        return False

    obs_pt = clicks[idx]  # TL,TR,BR,BL order
    # capture desktop mouse position
    try:
        out = subprocess.check_output(["xdotool", "getmouselocation", "--shell"], text=True)
        d = dict(line.split("=") for line in out.strip().splitlines() if "=" in line)
        desk_x, desk_y = int(d["X"]), int(d["Y"])
    except Exception as e:
        log(f"ERROR: xdotool getmouselocation failed: {e}")
        return False

    pairs.append({"obs": obs_pt, "desk": [desk_x, desk_y]})
    data["pairs"] = pairs
    _safe_json_save(SCREEN_MAP_FILE, data)
    log(f"Screen map point {idx+1}/4: OBS {obs_pt} -> DESK {[desk_x, desk_y]}")
    if len(pairs) == 4:
        # compute S from OBS->desktop
        src = np.array([p["obs"] for p in pairs], dtype=np.float64)
        dst = np.array([p["desk"] for p in pairs], dtype=np.float64)
        Sm, _ = cv2.findHomography(src, dst)
        if Sm is None:
            log("ERROR: could not compute S")
            return False
        global S
        S = Sm
        save_homography(SCREEN_MAP_FILE, "S", S)
        log(f"Saved S -> {SCREEN_MAP_FILE}")
    return True

def reset_screen_map():
    global S
    S = None
    if SCREEN_MAP_FILE.exists():
        SCREEN_MAP_FILE.unlink()
    log("Reset screen map; press F6 x4")
    return True

# ======================
# Drift detection
# ======================
def mapping_drift_suspected():
    """Heuristic: mapped OBS corners should land within display geometry."""
    if S is None:
        return False
    geom = screen_geom()
    if not geom:
        return False
    W, Hh = geom
    # Sample 3 points around playfield-ish
    samples = [(120, 120), (520, 360), (920, 600)]  # keep inside playfield-ish region
    for x_obs, y_obs in samples:
        x_scr, y_scr = obs_to_screen(S, x_obs, y_obs)
        if not (-50 <= x_scr <= W + 50 and -50 <= y_scr <= Hh + 50):
            return True
    return False

# ======================
# Debug dump
# ======================
def ring_push(frame_bgr):
    global _last_ring_push
    now = time.time()
    if now - _last_ring_push < 1.0 / float(RING_FPS):
        return
    _last_ring_push = now
    RING.append((now, frame_bgr.copy()))

def dump_debug(tag: str, extra: dict | None = None):
    ts = time.strftime("%Y%m%d_%H%M%S")
    outdir = DEBUG_DUMPS_DIR / f"{ts}_{tag}"
    outdir.mkdir(parents=True, exist_ok=True)

    # Save logs
    (outdir / "log.txt").write_text("\n".join(LOG_LINES))

    # Save matrices + state
    state = {
        "profile": PROFILE,
        "BTD6_WIN": BTD6_WIN,
        "SHADOW_WIN": SHADOW_WIN,
        "auto_state": auto_state,
        "stage": stage,
        "pending": pending,
        "last_reason": last_reason,
        "last_action": last_action,
        "H": H.tolist() if H is not None else None,
        "S": S.tolist() if S is not None else None,
    }
    if extra:
        state.update(extra)
    _safe_json_save(outdir / "state.json", state)

    # Save last frames as mp4 (if any)
    if len(RING) >= 2:
        frames = [f for _, f in RING]
        h, w = frames[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        vid = cv2.VideoWriter(str(outdir / "ring.mp4"), fourcc, RING_FPS, (w, h))
        for f in frames:
            vid.write(f)
        vid.release()

    log(f"DEBUG DUMP -> {outdir}")
    return outdir

# ======================
# HUD + reward plumbing (best-effort; never gates actions)
# ======================
def update_hud(frame_bgr):
    global last_hud
    try:
        # must use raw frame (not annotated) — per project rule
        info = hud.read_hud_from_frame(frame_bgr)
        if isinstance(info, dict):
            for k in ("money", "round_cur", "lives"):
                if k in info:
                    if k == "round_cur":
                        last_hud["round"] = info[k]
                    elif k == "money":
                        last_hud["money"] = info[k]
                    elif k == "lives":
                        last_hud["lives"] = info[k]
    except Exception:
        pass

# ======================
# Autonomous stage machine
# ======================
def set_auto_state(new_state):
    global auto_state, stage, pending, last_reason
    auto_state = new_state
    if auto_state != AUTO_LIVE:
        stage = "IDLE"
        pending = None
    last_reason = "STATE"
    log(f"AUTO STATE -> {['OFF','ARMED','LIVE'][auto_state]}")
    return True

def plan_next_action():
    """Pick next action; returns action dict or None."""
    if H is None:
        return None
    act = pick_action_simple()

    # Convert to OBS and block check
    if act["type"] == "PLACE":
        x_obs, y_obs = grid_to_obs(H, act["gx"], act["gy"])
        if is_obs_blocked(x_obs, y_obs):
            return None
        act["x_obs"], act["y_obs"] = x_obs, y_obs
    return act

def begin_pipeline_for_action(act):
    """Initialize pending pipeline and stage."""
    global pending, stage, stage_deadline, last_action, armed_marker

    if act is None:
        return False

    last_action = act
    pending = {"act": act, "placed": False, "selected": False, "upgraded": False}

    if act["type"] == "PLACE":
        stage = "PLACE"
    else:
        stage = "IDLE"

    stage_deadline = time.time()
    # marker for ARMED visualization
    if S is not None and "x_obs" in act:
        x_scr, y_scr = obs_to_screen(S, act["x_obs"], act["y_obs"])
        armed_marker = (act["x_obs"], act["y_obs"], x_scr, y_scr, act["type"], time.time())
    return True

def step_pipeline():
    """Executes one stage step; returns True if did something."""
    global stage, stage_deadline, pending, last_reason

    if pending is None:
        return False
    act = pending["act"]

    now = time.time()
    if now < stage_deadline:
        last_reason = "SKIP_STAGE_CD"
        return False

    if act["type"] != "PLACE":
        stage = "IDLE"
        pending = None
        return False

    # Need mappings and windows
    if H is None:
        last_reason = "SKIP_NO_H"
        return False
    if S is None:
        last_reason = "SKIP_NO_S"
        return False
    if not require_windows(allow_reacquire=False):
        last_reason = "SKIP_NO_WIN"
        return False

    x_obs, y_obs = act["x_obs"], act["y_obs"]
    x_scr, y_scr = obs_to_screen(S, x_obs, y_obs)

    if stage == "PLACE":
        ok = do_click(x_scr, y_scr, tower_name=act.get("tower", "DART"), retries=CLICK_RETRIES)
        if not ok:
            last_reason = "ERR_CLICK_FAIL"
            dump_debug("click_fail", {"stage": stage})
            return False
        pending["placed"] = True
        stage = "SELECT"
        stage_deadline = time.time() + STAGE_COOLDOWN["SELECT"]
        return True

    if stage == "SELECT":
        ok = do_click(x_scr, y_scr, tower_name=None, retries=1)
        if not ok:
            last_reason = "ERR_CLICK_FAIL"
            dump_debug("select_fail", {"stage": stage})
            return False
        pending["selected"] = True
        stage = "UPGRADE"
        stage_deadline = time.time() + STAGE_COOLDOWN["UPGRADE"]
        return True

    if stage == "UPGRADE":
        # Example: buy one tier on path2 (your project default)
        ok = click_upgrade_path(2, retries=CLICK_RETRIES)
        if not ok:
            last_reason = "ERR_UPGRADE_FAIL"
            dump_debug("upgrade_fail", {"stage": stage})
            # Still end pipeline to avoid infinite loop
        pending["upgraded"] = True
        stage = "IDLE"
        pending = None
        return True

    return False

# ======================

# ==========================
# On-screen help overlay
# ==========================
HELP_LINES = [
    "c: calib corners (grid->OBS) | F6: add screen-map corner | F7: reset screen-map | s: save",
    "g: LOCK windows (focus game then press g in SHADOW) | F8: place once | F9: AUTO OFF/ARMED/LIVE",
    "U: UI capture (then 1/2/3/4 record PATH1/2/3/SELL at mouse) | 1/2/3: upgrade click",
    "Spots: j/k cycle | x delete | X clear | y toggle circles",
    "Blocked: b toggle edit | Enter commit | z undo point | Z delete last poly",
    "Profiles: p next | P prev | d toggle drift-check | q quit",
]

HELP_SCALE = 0.42  # ~70% of 0.6 used elsewhere
HELP_THICK = 1

def draw_help_overlay(img):
    """Readable keybind/help overlay at bottom with black text."""
    if not HELP_LINES:
        return
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.85  # readable
    thick = 2

    pad_x = 14
    pad_y = 12
    line_gap = 8

    sizes = [cv2.getTextSize(line, font, scale, thick)[0] for line in HELP_LINES]
    max_w = max((w for (w, h) in sizes), default=0)
    line_h = max((h for (w, h) in sizes), default=18)

    box_w = max_w + pad_x * 2
    box_h = (line_h + line_gap) * len(HELP_LINES) + pad_y * 2 - line_gap

    Himg, Wimg = img.shape[:2]
    x0 = 10
    y0 = max(0, Himg - box_h - 10)
    x1 = min(Wimg - 1, x0 + box_w)
    y1 = min(Himg - 1, y0 + box_h)

    overlay = img.copy()
    cv2.rectangle(overlay, (x0, y0), (x1, y1), (245, 245, 245), -1)
    img[:] = cv2.addWeighted(overlay, 0.85, img, 0.15, 0)

    y = y0 + pad_y + line_h
    for line in HELP_LINES:
        cv2.putText(img, line, (x0 + pad_x, y), font, scale, (0, 0, 0), thick, cv2.LINE_AA)
        y += line_h + line_gap


# Main loop
# ======================
def main():
    global SHADOW_WIN, BTD6_WIN, EXPECTED_BTD6_WMCLASS, H, S
    global auto_state, last_act_ts, last_heartbeat_ts, last_drift_check_ts, last_reason, armed_marker
    global blocked_edit_mode, _current_poly

    # Ensure default profile exists and load
    profs = list_profiles()
    set_profile(profs[0] if profs else DEFAULT_PROFILE)

    # Capture
    cap = cv2.VideoCapture(CAM_INDEX)
    if not cap.isOpened():
        raise SystemExit(f"ERROR: Cannot open camera index {CAM_INDEX}. Is OBS Virtual Camera running?")

    cv2.namedWindow("BTD6 SHADOW", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("BTD6 SHADOW", on_main_mouse)

    log("Keys: c=calib grid->OBS | F6 add screen-map point | F7 reset screen-map | F8 place once | F9 cycle OFF/ARMED/LIVE")
    log("      g=lock game window (focus game then press g from shadow) | 1/2/3 click upgrade path | b=toggle blocked-poly edit | Enter=commit poly")
    log("      p=next profile | P=prev profile | s=save spots/stats/blocked")

    # attempt to lock shadow window by name once it's created
    time.sleep(0.1)
    SHADOW_WIN = find_shadow_window()

    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            time.sleep(0.02)
            continue

        ring_push(frame)
        update_hud(frame)

        vis = frame.copy()

        # Draw overlay (does NOT affect OCR)
        cv2.putText(vis, f"PROFILE: {PROFILE}", (12, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

        auto_label = ["OFF", "ARMED", "LIVE"][auto_state]
        hud_txt = f"$={last_hud['money']} R={last_hud['round']} L={last_hud['lives']}"
        cv2.putText(vis, f"AUTO: {auto_label}  STAGE: {stage}  REASON: {last_reason}  {hud_txt}",
                    (12, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 2)

        # Expected / planned marker (yellow circle)
        if armed_marker is not None:
            x_obs, y_obs, x_scr, y_scr, label, ts = armed_marker
            # Keep it visible for a while so you can see where the AI "wants" to act.
            if time.time() - ts < 15.0:
                cv2.circle(vis, (int(x_obs), int(y_obs)), 12, (0, 255, 255), 2)
                cv2.putText(vis, f"TARGET {label} -> desk({int(x_scr)},{int(y_scr)})", (12, 42),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,255,255), 2)
            else:
                armed_marker = None

        # Blocked poly editing overlay
        if blocked_edit_mode:
            for pt in _current_poly:
                cv2.circle(vis, pt, 5, (0,0,255), -1)
            if len(_current_poly) >= 2:
                cv2.polylines(vis, [np.array(_current_poly, dtype=np.int32)], False, (0,0,255), 2)
            cv2.putText(vis, "BLOCKED EDIT: click points, Enter=commit, Esc=cancel", (12, 700),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,0,255), 2)

        # Help overlay
        draw_help_overlay(vis)

        cv2.imshow("BTD6 SHADOW", vis)

        # Heartbeat
        now = time.time()
        if auto_state != AUTO_OFF and (now - last_heartbeat_ts) >= HEARTBEAT_INTERVAL:
            last_heartbeat_ts = now
            ok_win = require_windows(allow_reacquire=True)
            log(f"HB: AUTO={auto_label} stage={stage} reason={last_reason} win={'OK' if ok_win else 'BAD'} hud={hud_txt}")

        # Drift detector (auto pause to ARMED)
        if auto_state == AUTO_LIVE and (now - last_drift_check_ts) >= DRIFT_CHECK_INTERVAL:
            last_drift_check_ts = now
            if mapping_drift_suspected():
                log("DRIFT SUSPECTED: pausing to ARMED — redo F6 mapping")
                dump_debug("drift_suspected")
                set_auto_state(AUTO_ARMED)

        # Autonomous planning/execution
        if auto_state in (AUTO_ARMED, AUTO_LIVE) and (now - last_act_ts) >= ACTION_COOLDOWN:
            # If no pending pipeline, plan one
            if pending is None:
                act = plan_next_action()
                if act is None:
                    last_reason = "SKIP_BLOCKED"
                else:
                    begin_pipeline_for_action(act)
                    last_reason = "PLANNED"

            if auto_state == AUTO_LIVE:
                did = step_pipeline()
                if did:
                    last_act_ts = time.time()
            else:
                # ARMED: do not click, just update marker
                last_act_ts = time.time()

        k = cv2.waitKeyEx(1)
        key = k & 0xFF

        if key == 27 or key == ord('q'):
            break

        # Calibration
        if key == ord('c'):
            calibrate_grid_to_obs(frame)
            # reload
            H = load_homography(CALIB_FILE, key="H")
            continue

        # Screen mapping
        if k == 0:  # no key
            pass
        elif k == 0xFF:
            pass

        # OpenCV doesn't always pass function keys consistently across setups; we support both.
        # F6/F7/F8/F9 often appear as 0 with extended codes in waitKeyEx; but we keep compatibility
        # by also listening to ord equivalents: '6','7','8','9' as fallback.
        if key == ord('6'):
            record_screen_map_point()
        elif key == ord('7'):
            reset_screen_map()
        elif key == ord('8'):
            # manual place once
            if H is None or S is None:
                log("Manual place needs H + S (c then F6 x4)")
            else:
                act = pick_action_simple()
                act = plan_next_action()
                if act and act.get("type") == "PLACE":
                    x_obs, y_obs = act["x_obs"], act["y_obs"]
                    x_scr, y_scr = obs_to_screen(S, x_obs, y_obs)
                    do_click(x_scr, y_scr, tower_name=act.get("tower","DART"))
        elif key == ord('9'):
            # cycle OFF -> ARMED -> LIVE -> OFF
            if auto_state == AUTO_OFF:
                set_auto_state(AUTO_ARMED)
            elif auto_state == AUTO_ARMED:
                set_auto_state(AUTO_LIVE)
            else:
                set_auto_state(AUTO_OFF)

        # Function-key support (X11 keysyms via waitKeyEx)
        elif k == 65475:  # F6
            record_screen_map_point()
        elif k == 65476:  # F7
            reset_screen_map()
        elif k == 65477:  # F8
            if H is None or S is None:
                log("Manual place needs H + S (c then F6 x4)")
            else:
                act = plan_next_action()
                if act and act.get("type") == "PLACE":
                    x_obs, y_obs = act["x_obs"], act["y_obs"]
                    x_scr, y_scr = obs_to_screen(S, x_obs, y_obs)
                    do_click(x_scr, y_scr, tower_name=act.get("tower","DART"))
        elif k == 65478:  # F9
            if auto_state == AUTO_OFF:
                set_auto_state(AUTO_ARMED)
            elif auto_state == AUTO_ARMED:
                set_auto_state(AUTO_LIVE)
            else:
                set_auto_state(AUTO_OFF)


        
        # UI capture (desktop coords for sidebar buttons)
        elif key == ord('U') or key == ord('u'):
            global ui_capture_mode
            ui_capture_mode = not ui_capture_mode
            log(f"UI CAPTURE MODE -> {'ON' if ui_capture_mode else 'OFF'} (1/2/3/4 record PATH1/2/3/SELL at current mouse)")
        elif ui_capture_mode and key in (ord('1'), ord('2'), ord('3'), ord('4')):
            mx, my = get_mouse_desktop_pos()
            if mx is None:
                log("UI CAPTURE: cannot read mouse position (xdotool?)")
            else:
                keymap = {ord('1'): "PATH1", ord('2'): "PATH2", ord('3'): "PATH3", ord('4'): "SELL"}
                kk = keymap[key]
                UI_DESKTOP[kk] = [int(mx), int(my)]
                log(f"UI CAPTURE: {kk} -> desktop({int(mx)},{int(my)})")
                save_ui_desktop()

# Upgrades (manual)
        elif key == ord('1'):
            if not ui_capture_mode:
                click_upgrade_path(1)
        elif key == ord('2'):
            if not ui_capture_mode:
                click_upgrade_path(2)
        elif key == ord('3'):
            if not ui_capture_mode:
                click_upgrade_path(3)

        # Lock game window
        elif key == ord('g'):
            # Lock BTD6 + SHADOW window IDs robustly (prevents Chrome mis-targeting).
            # If active window is the shadow window, fall back to search-by-name for the real game.
            SHADOW_WIN = find_shadow_window()

            active = get_active_window_id()
            active_name = get_window_name(active) if active else ""
            if active and ("SHADOW" not in active_name.upper()) and (active != SHADOW_WIN):
                BTD6_WIN = active
            else:
                BTD6_WIN = find_btd6_window()

            EXPECTED = get_window_class(BTD6_WIN) if BTD6_WIN else ""
            global EXPECTED_BTD6_WMCLASS
            EXPECTED_BTD6_WMCLASS = EXPECTED if EXPECTED else None

            log(f"LOCK: BTD6_WIN={BTD6_WIN} name='{get_window_name(BTD6_WIN)}'")
            if EXPECTED_BTD6_WMCLASS:
                log(f"LOCK: WM_CLASS={EXPECTED_BTD6_WMCLASS}")


        # Profiles
        elif key == ord('p'):
            profs = list_profiles()
            if PROFILE in profs and profs:
                idx = profs.index(PROFILE)
                set_profile(profs[(idx + 1) % len(profs)])
        elif key == ord('P'):
            profs = list_profiles()
            if PROFILE in profs and profs:
                idx = profs.index(PROFILE)
                set_profile(profs[(idx - 1) % len(profs)])

        # Blocked polygon editing toggle
        elif key == ord('b'):
            blocked_edit_mode = not blocked_edit_mode
            if not blocked_edit_mode:
                _current_poly = []
            log(f"BLOCKED EDIT -> {blocked_edit_mode}")

        elif k == 13:  # Enter commits polygon
            if blocked_edit_mode and len(_current_poly) >= 3:
                blocked_polygons.append(_current_poly[:])
                log(f"BLOCKED: committed polygon with {len(_current_poly)} points (total {len(blocked_polygons)})")
                _current_poly = []
                save_blocked()

        elif k == 8:  # Backspace undo last point
            if blocked_edit_mode and _current_poly:
                _current_poly.pop()

        elif key == ord('s'):
            save_spots()
            save_stats()
            save_blocked()
            save_ui_desktop()

        elif key == ord('d'):
            dump_debug("manual")

    cap.release()
    cv2.destroyAllWindows()
    log("Exit")


def get_mouse_desktop_pos():
    """Return (x,y) desktop mouse position using xdotool, or (None,None) on failure."""
    try:
        out = subprocess.check_output(["xdotool", "getmouselocation", "--shell"], text=True)
        vals = {}
        for line in out.strip().splitlines():
            if "=" in line:
                k, v = line.split("=", 1)
                vals[k.strip()] = v.strip()
        x = int(vals.get("X", ""))
        y = int(vals.get("Y", ""))
        return x, y
    except Exception:
        return None, None

if __name__ == "__main__":
    main()
