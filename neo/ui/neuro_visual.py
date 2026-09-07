from __future__ import annotations

import math
import random

import numpy as np
from PySide6.QtCore import QTimer
from PySide6.QtGui import QImage, QPainter
from PySide6.QtWidgets import QWidget

from ..core.state import AgentState

# ── State parameters ──────────────────────────────────────────────────────
_STATE_HUE: dict[AgentState, float] = {
    AgentState.IDLE:       0.40,   # green
    AgentState.LISTENING:  0.58,   # blue
    AgentState.PROCESSING: 0.12,   # amber
    AgentState.SPEAKING:   0.40,   # green
    AgentState.ERROR:      0.00,   # red
}
_STATE_SAT: dict[AgentState, float] = {
    AgentState.IDLE:       0.75,
    AgentState.LISTENING:  0.70,
    AgentState.PROCESSING: 0.85,
    AgentState.SPEAKING:   0.70,
    AgentState.ERROR:      0.90,
}
_STATE_SPIN: dict[AgentState, float] = {
    AgentState.IDLE:       0.00022,
    AgentState.LISTENING:  0.00062,
    AgentState.PROCESSING: 0.00090,
    AgentState.SPEAKING:   0.00075,
    AgentState.ERROR:      0.00015,
}
_STATE_BREATHE: dict[AgentState, float] = {
    AgentState.IDLE:       0.0010,
    AgentState.LISTENING:  0.0028,
    AgentState.PROCESSING: 0.0045,
    AgentState.SPEAKING:   0.0035,
    AgentState.ERROR:      0.0010,
}
_STATE_TICK_MS: dict[AgentState, int] = {
    AgentState.IDLE:       33,
    AgentState.LISTENING:  25,
    AgentState.PROCESSING: 20,
    AgentState.SPEAKING:   20,
    AgentState.ERROR:      33,
}

# Camera/projection constants
_CAM_DIST    = 3.2   # perspective camera distance (sphere radius = 1)
_CLIP_SCALE  = 0.80  # sphere fills 80 % of widget height in clip space

_N_SPHERE    = 5000  # main particle cloud
_N_STARS     = 300   # background star field (fixed, no rotation)

# Background colour (matches theme)
_BG = np.array([0.024, 0.039, 0.031], dtype=np.float32)


def _rand_unit_sphere(n: int, rng: random.Random) -> np.ndarray:
    """Uniformly distributed unit-sphere directions, shape (n, 3)."""
    pts = []
    while len(pts) < n:
        x, y, z = (rng.gauss(0, 1), rng.gauss(0, 1), rng.gauss(0, 1))
        r = math.sqrt(x*x + y*y + z*z)
        if r > 1e-6:
            pts.append((x/r, y/r, z/r))
    return np.array(pts, dtype=np.float32)


def _hsv_to_rgb(h: float, s: float, v: np.ndarray) -> np.ndarray:
    """Vectorised HSV → RGB. h,s scalar; v (N,) array. Returns (N, 3) float32."""
    h6   = (h * 6.0) % 6.0
    seg  = int(h6)
    f    = h6 - seg
    c    = v * s
    x    = c * (1.0 - abs(h6 % 2.0 - 1.0))
    m    = v - c
    z    = np.zeros_like(v)
    mapping = [(c, x, z), (x, c, z), (z, c, x),
               (z, x, c), (x, z, c), (c, z, x)]
    r0, g0, b0 = mapping[seg]
    return np.stack([r0 + m, g0 + m, b0 + m], axis=1)


def _box_blur1(img: np.ndarray, r: int) -> np.ndarray:
    """Single-pass separable box blur. Output shape == input shape."""
    H, W, C = img.shape
    # Horizontal
    p = np.pad(img, ((0, 0), (r, r), (0, 0)), mode='edge')
    z = np.zeros((H, 1, C), dtype=img.dtype)
    cs = np.cumsum(np.concatenate([z, p], axis=1), axis=1)
    h = (cs[:, 2*r+1:, :] - cs[:, :W, :]) / (2*r + 1)
    # Vertical
    p = np.pad(h, ((r, r), (0, 0), (0, 0)), mode='edge')
    z = np.zeros((1, W, C), dtype=img.dtype)
    cs = np.cumsum(np.concatenate([z, p], axis=0), axis=0)
    return (cs[2*r+1:, :, :] - cs[:H, :, :]) / (2*r + 1)


def _gauss_blur(img: np.ndarray, r: int) -> np.ndarray:
    """3 × box blur ≈ Gaussian — eliminates square grid artefacts."""
    out = _box_blur1(img, r)
    out = _box_blur1(out, r)
    return _box_blur1(out, r)


def _true_gauss_blur(img: np.ndarray, sigma: float) -> np.ndarray:
    """True separable Gaussian — produces perfectly round halos."""
    r = max(1, int(3.0 * sigma))
    x = np.arange(-r, r + 1, dtype=np.float32)
    k = np.exp(-x ** 2 / (2.0 * sigma ** 2))
    k /= k.sum()

    def _conv(arr: np.ndarray, axis: int) -> np.ndarray:
        pad = [(0, 0)] * 3
        pad[axis] = (r, r)
        p = np.pad(arr, pad, mode='edge')
        out = np.zeros_like(arr)
        for i, ki in enumerate(k):
            sl: list = [slice(None)] * 3
            sl[axis] = slice(i, i + arr.shape[axis])
            out += ki * p[tuple(sl)]
        return out

    return _conv(_conv(img, 1), 0)   # horizontal then vertical


class _ParticleData:
    """Pre-computed, immutable particle geometry (regenerated on resize)."""

    def __init__(self, n_sphere: int, n_stars: int, seed: int = 42) -> None:
        rng = random.Random(seed)

        # ── Sphere particles ──────────────────────────────────────────────
        dirs  = _rand_unit_sphere(n_sphere, rng)
        # Gaussian around r=0.88: dense shell + some interior fill
        radii = np.clip(
            np.array([rng.gauss(0.88, 0.07) for _ in range(n_sphere)], np.float32),
            0.55, 1.02,
        )
        # Hero particles (5 %) are 3-5× larger for bright nodes
        sizes = np.array(
            [rng.uniform(3.0, 5.0) if rng.random() < 0.05
             else rng.uniform(0.8, 2.5)
             for _ in range(n_sphere)],
            dtype=np.float32,
        )
        phases = np.array(
            [rng.uniform(0.0, 2.0 * math.pi) for _ in range(n_sphere)],
            dtype=np.float32,
        )
        self.sphere_dirs   = dirs            # (N, 3) unit vectors
        self.sphere_radii  = radii           # (N,)
        self.sphere_sizes  = sizes           # (N,)
        self.sphere_phases = phases          # (N,)

        # ── Background stars (fixed, never rotate) ────────────────────────
        star_dirs  = _rand_unit_sphere(n_stars, rng)
        star_radii = np.array(
            [rng.uniform(8.0, 22.0) for _ in range(n_stars)], dtype=np.float32
        )
        star_phases = np.array(
            [rng.uniform(0.0, 2.0 * math.pi) for _ in range(n_stars)],
            dtype=np.float32,
        )
        # Pre-project stars (they never move)
        p   = star_dirs * star_radii[:, None]
        psc = _CAM_DIST / (_CAM_DIST - p[:, 2])
        self.star_sx     = p[:, 0] * psc * 0.055   # clip-space x (small scale)
        self.star_sy     = p[:, 1] * psc * 0.055   # clip-space y
        self.star_depth  = np.clip((p[:, 2] + 1.0) * 0.5, 0.0, 1.0)
        self.star_phases = star_phases


class NeuroVisual(QWidget):
    """NEO's central animation: a dense, glowing particle-cloud sphere that
    rotates, breathes, and colour-shifts with AgentState — JARVIS-style
    volumetric look achieved entirely via numpy (fast separable box-blur glow
    + additive scatter) and QPainter image blit.  No OpenGL required."""

    _COMPACT_SIZE = 280
    _FOCUS_SIZE   = 560

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._focus = False
        self.setFixedSize(self._COMPACT_SIZE, self._COMPACT_SIZE)

        self._state         = AgentState.IDLE
        self._hue           = _STATE_HUE[AgentState.IDLE]
        self._sat           = _STATE_SAT[AgentState.IDLE]
        self._yaw           = 0.0
        self._breathe_phase = 0.0
        self._breathe       = 0.5
        self._time          = 0.0
        self._audio_level        = 0.0   # raw input 0-1, set externally
        self._audio_level_smooth = 0.0   # smoothed for rendering

        self._pdata = _ParticleData(_N_SPHERE, _N_STARS)

        self._timer = QTimer(self)
        self._timer.timeout.connect(self._tick)
        self._timer.setInterval(_STATE_TICK_MS[AgentState.IDLE])
        self._timer.start()

    # ── Public API ────────────────────────────────────────────────────────
    def set_focus(self, focus: bool) -> None:
        if focus == self._focus:
            return
        self._focus = focus
        size = self._FOCUS_SIZE if focus else self._COMPACT_SIZE
        self.setFixedSize(size, size)

    def set_state(self, state: AgentState) -> None:
        self._state = state
        self._timer.setInterval(_STATE_TICK_MS[state])
        if state != AgentState.SPEAKING:
            self._audio_level = 0.0

    def set_audio_level(self, level: float) -> None:
        """Drive the sphere's voice-reactivity (0.0 = silence, 1.0 = loud).
        Call repeatedly while SPEAKING; resets automatically on state change."""
        self._audio_level = max(0.0, min(1.0, float(level)))

    # ── Animation ─────────────────────────────────────────────────────────
    def _tick(self) -> None:
        dt = self._timer.interval()
        self._yaw           += dt * _STATE_SPIN[self._state]
        self._breathe_phase += dt * _STATE_BREATHE[self._state]
        self._breathe        = 0.5 + 0.5 * math.sin(self._breathe_phase)
        self._time          += dt / 1000.0

        target_hue = _STATE_HUE[self._state]
        target_sat = _STATE_SAT[self._state]
        self._hue  += (target_hue - self._hue) * 0.06
        self._sat  += (target_sat - self._sat) * 0.06

        # Smooth audio level — fast attack (0.4), slow release (0.12)
        target = self._audio_level
        alpha  = 0.40 if target > self._audio_level_smooth else 0.12
        self._audio_level_smooth += (target - self._audio_level_smooth) * alpha

        self.update()

    def showEvent(self, event) -> None:  # noqa: N802
        super().showEvent(event)
        if not self._timer.isActive():
            self._timer.start()

    def hideEvent(self, event) -> None:  # noqa: N802
        self._timer.stop()
        super().hideEvent(event)

    # ── Rendering ─────────────────────────────────────────────────────────
    def _rotation_matrix(self) -> np.ndarray:
        """3×3 rotation: yaw (Y-axis) followed by oscillating pitch (X-axis)."""
        yaw   = self._yaw
        pitch = math.sin(self._yaw * 0.38) * 0.32
        cy, sy = math.cos(yaw),   math.sin(yaw)
        cp, sp = math.cos(pitch), math.sin(pitch)
        # Ry @ Rx
        return np.array([
            [cy,  sy*sp,  sy*cp],
            [0,   cp,    -sp   ],
            [-sy, cy*sp,  cy*cp],
        ], dtype=np.float32)

    def _render_frame(self, W: int, H: int) -> QImage:
        pd    = self._pdata
        av    = self._audio_level_smooth            # 0-1 voice reactivity
        # breath: base oscillation + voice-reactive swell (SPEAKING only)
        voice_swell = av * 0.18 if self._state == AgentState.SPEAKING else 0.0
        breath = 0.97 + 0.06 * self._breathe + voice_swell

        # ── Project sphere particles ──────────────────────────────────────
        R    = self._rotation_matrix()
        pos  = (pd.sphere_dirs * (pd.sphere_radii * breath)[:, None]) @ R.T  # (N, 3)
        psc  = _CAM_DIST / np.maximum(_CAM_DIST - pos[:, 2], 0.01)           # (N,)
        depth = np.clip((pos[:, 2] + 1.0) * 0.5, 0.0, 1.0)                   # (N,)

        cx, cy = W * 0.5, H * 0.5
        px = (pos[:, 0] * psc * _CLIP_SCALE * cx + cx).astype(np.int32)
        py = ((-pos[:, 1]) * psc * _CLIP_SCALE * cy + cy).astype(np.int32)

        # Per-particle alpha: depth + twinkle + size + voice boost
        tw    = 0.5 + 0.5 * np.sin(self._time * 2.4 + pd.sphere_phases)
        alpha = (0.45 + 0.55 * depth) * (0.55 + 0.45 * tw) * (pd.sphere_sizes / 3.5)
        alpha *= (1.0 + 0.7 * av)   # voice level brightens all particles

        # Colours (HSV depth-modulated value)
        val  = np.clip(0.65 + 0.35 * depth + 0.15 * av, 0.0, 1.0).astype(np.float32)
        rgb  = _hsv_to_rgb(self._hue % 1.0, self._sat, val)  # (N, 3)
        weighted = rgb * alpha[:, None]                        # (N, 3)

        # Scatter into float buffer via bincount (fast vectorised scatter)
        buf = np.zeros((H, W, 3), dtype=np.float32)
        mask = (px >= 0) & (px < W) & (py >= 0) & (py < H)
        if mask.any():
            idx = py[mask] * W + px[mask]
            for ch in range(3):
                buf[:, :, ch].flat += np.bincount(
                    idx, weights=weighted[mask, ch], minlength=H * W
                )

        # ── Background stars (no rotation, tiny) ─────────────────────────
        stw    = 0.35 + 0.65 * np.sin(self._time * 1.5 + pd.star_phases)
        s_alph = stw * 0.40
        spx = (pd.star_sx * cx + cx).astype(np.int32)
        spy = (pd.star_sy * cy + cy).astype(np.int32)
        s_mask = (spx >= 0) & (spx < W) & (spy >= 0) & (spy < H)
        if s_mask.any():
            s_idx = spy[s_mask] * W + spx[s_mask]
            star_rgb = np.tile([0.78, 0.90, 1.00], (s_mask.sum(), 1))
            s_w = star_rgb * s_alph[s_mask, None]
            for ch in range(3):
                buf[:, :, ch].flat += np.bincount(
                    s_idx, weights=s_w[:, ch], minlength=H * W
                )

        # ── Two-scale glow: true-Gaussian halo + wide bloom ─────────────
        W_w = self.width()
        g1 = _true_gauss_blur(buf, 1.5)           # sigma=1.5px → tight round dots
        r_bloom = max(14, int(W_w * 0.07))         # ~32px wide bloom
        g2 = _gauss_blur(buf, r_bloom)

        glow_mult = 22.0 + 16.0 * av
        result = np.clip(_BG + g1 * glow_mult + g2 * (2.2 + 2.5 * av), 0.0, 1.0)

        # ── Convert to QImage ─────────────────────────────────────────────
        rgb8  = (result * 255.0).astype(np.uint8)
        alpha = np.full((H, W, 1), 255, dtype=np.uint8)
        rgba  = np.ascontiguousarray(np.concatenate([rgb8, alpha], axis=2))
        return QImage(rgba.data, W, H, W * 4, QImage.Format.Format_RGBA8888)

    def paintEvent(self, event) -> None:  # noqa: N802
        W = self.width()
        H = self.height()
        img = self._render_frame(W, H)
        painter = QPainter(self)
        painter.drawImage(0, 0, img)
        painter.end()
