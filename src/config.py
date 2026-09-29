"""Vis settings (camera + audio), edited in the "Edit Settings" panel (H) and persisted to
rsv_settings.json on every change, so they survive closing the vis (or the GUI killing it).

Camera settings mirror Rocket League's own (same names, ranges and meaning): FOV is HORIZONTAL like
in RL, Distance/Height are the yaw-only offset behind the car, Angle pitches the view, Stiffness 0 lets
the camera pull back at speed, Transition speed scales ball-cam <-> car-cam switching.
"""
import json
import os

SETTINGS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "rsv_settings.json")


class ConfigVal:
    def __init__(self, default, min, max, decimals=None):
        self.val = float(default)
        self.min = float(min)
        self.max = float(max)
        self.decimals = decimals

    def __float__(self):
        return self.val


class Config:
    # name -> (default, min, max, decimals). Defaults = a typical pro RL camera.
    CAMERA = {
        "camera_fov": (110, 60, 110, 0),
        "camera_distance": (270, 100, 400, 0),
        "camera_height": (100, 40, 200, 0),
        "camera_angle": (-4, -15, 0, 0),
        "camera_stiffness": (0.45, 0, 1, 2),
        "camera_transition_speed": (1.2, 1, 2, 2),
        "camera_bird_fov": (60, 20, 120, 0),
    }
    # per-category sound volumes (percent, on top of the master volume)
    SOUND_MIX = {
        "vol_ball": "Ball touches",
        "vol_demo": "Demos",
        "vol_impact": "Body impacts / bumps",
        "vol_post": "Goal posts / crossbar",
        "vol_boost": "Boost",
        "vol_engine": "Car engine",
        "vol_flips": "Flips / jumps",
        "vol_reset": "Flip reset indicator",
    }

    # Graphics (Edit Settings > Graphics): name -> (label, [(value, shown text), ...], default)
    GRAPHICS = {
        "gfx_preset": ("Quality preset", [("custom", "Custom"), ("low", "Low"), ("medium", "Medium"), ("high", "High")],
                       "custom"),
        "gfx_aa": ("Anti-aliasing (MSAA)", [(0, "Off"), (2, "2x"), (4, "4x"), (8, "8x")],
                   int(os.environ.get("RSV_MSAA", "4"))),
        "gfx_resolution": ("Resolution", [("auto", "Balanced (max 2.1 MP, upscaled)"), ("native", "Native"),
                                          ("150", "Supersampled 1.5x"), ("200", "Supersampled 2x")], "auto"),
        "gfx_shadows": ("Shadows", [(1, "On"), (0, "Off")], 1),
        "gfx_ball_trail": ("Ball trail", [(1, "On"), (0, "Off")], 1),
        "gfx_ball_marker": ("Ball circles", [(1, "On"), (0, "Off")], 1),
        "gfx_detail": ("Distant detail", [("smooth", "Smooth"), ("sharp", "Sharp (may shimmer)")], "smooth"),
        "gfx_vsync": ("VSync", [(1, "On"), (0, "Off")], int(os.environ.get("RSV_VSYNC", "1"))),
        "gfx_fps": ("Frame rate cap", [(0, "Monitor refresh"), (60, "60"), (120, "120"), (144, "144"),
                                       (240, "240"), (-1, "Unlimited")], 0),
    }

    # Visual quality sliders (Edit Settings > Graphics): name -> (label, level names, default level index)
    QUALITY = {
        "q_shadow": ("Shadow quality", ["Low", "Medium", "High"], 2),
        "q_crowd": ("Crowd", ["Low", "Medium", "High"], 2),
        "q_map": ("Map detail", ["Low", "Medium", "High"], 2),
        "gfx_grass": ("Grass", ["Off", "Low", "Medium", "High"], 2),
        "q_particles": ("Particles", ["Low", "Medium", "High"], 2),
        "q_res": ("Render resolution", ["70%", "85%", "100%"], 2),
    }
    # Low = 165 fps on the laptop's integrated GPU (Radeon 780M), High = everything maxed
    PRESETS = {
        "low": {"gfx_aa": 2, "q_shadow": 0, "q_crowd": 0, "q_map": 0, "gfx_grass": 0, "q_particles": 0, "q_res": 0},
        "medium": {"gfx_aa": 4, "q_shadow": 1, "q_crowd": 1, "q_map": 1, "gfx_grass": 1, "q_particles": 1, "q_res": 1},
        "high": {"gfx_aa": 8, "q_shadow": 2, "q_crowd": 2, "q_map": 2, "gfx_grass": 3, "q_particles": 2, "q_res": 2},
    }

    def apply_preset(self, name):
        for k, v in self.PRESETS.get(name, {}).items():
            setattr(self, k, v)
        self.gfx_preset = name if name in self.PRESETS else "custom"

    def matching_preset(self):
        for name, vals in self.PRESETS.items():
            if all(getattr(self, k) == v for k, v in vals.items()):
                return name
        return "custom"

    def __init__(self):
        gfx = self._load().get("graphics", {})
        for name, (_label, choices, d) in self.GRAPHICS.items():
            v = gfx.get(name, d)
            setattr(self, name, v if v in [c[0] for c in choices] else d)
        for name, (_label, levels, d) in self.QUALITY.items():
            v = gfx.get(name, d)
            setattr(self, name, v if isinstance(v, int) and 0 <= v < len(levels) else d)
        saved = self._load().get("camera", {})
        for name, (d, lo, hi, dec) in self.CAMERA.items():
            v = saved.get(name, d)
            try:
                v = min(hi, max(lo, float(v)))
            except (TypeError, ValueError):
                v = d
            setattr(self, name, ConfigVal(v, lo, hi, dec))
        vol = self._load().get("volume", 0.7)
        self.volume = ConfigVal(round(float(vol) * 100), 0, 100, 0)
        mix = self._load().get("sound_mix", {})
        for name in self.SOUND_MIX:
            try:
                v = min(100.0, max(0.0, float(mix.get(name, 100))))
            except (TypeError, ValueError):
                v = 100.0
            setattr(self, name, ConfigVal(v, 0, 100, 0))

    @staticmethod
    def _load():
        try:
            with open(SETTINGS_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except (OSError, ValueError):
            return {}

    def save(self):
        cur = self._load()
        cur["camera"] = {name: getattr(self, name).val for name in self.CAMERA}
        cur["volume"] = round(self.volume.val / 100.0, 3)
        cur["sound_mix"] = {name: round(getattr(self, name).val) for name in self.SOUND_MIX}
        cur["graphics"] = {name: getattr(self, name) for name in list(self.GRAPHICS) + list(self.QUALITY)}
        try:
            tmp = SETTINGS_PATH + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(cur, f, indent=1)
            os.replace(tmp, SETTINGS_PATH)
        except OSError:
            pass
