from __future__ import annotations

import math
import random

import numpy as np
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QImage, QPainter, QPixmap
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
    AgentState.IDLE:       50,   # ~20 fps idle — saves GPU/CPU
    AgentState.LISTENING:  33,
    AgentState.PROCESSING: 25,
    AgentState.SPEAKING:   25,
    AgentState.ERROR:      50,
}

# Camera/projection constants
_CAM_DIST    = 3.2   # perspective camera distance (sphere radius = 1)
_CLIP_SCALE  = 0.80  # sphere fills 80 % of widget height in clip space

_N_SPHERE    = 4000  # main particle cloud
_N_STARS     = 250   # background star field (fixed, no rotation)

# Internal render resolution — blur cost is O(N²), so keep this fixed
# regardless of widget/screen size; Qt scales up with SmoothTransformation.
_RENDER_SIZE = 400

# Background colour — pure black for maximum contrast/glow pop
_BG = np.array([0.0, 0.0, 0.0], dtype=np.float32)


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
    k = 2 * r + 1

    def _blur_axis(a: np.ndarray, ax: int) -> np.ndarray:
        N = a.shape[ax]
        # Pre-allocate padded + zero-prepended buffer: size N+2r+1 along ax
        shape = list(a.shape)
        shape[ax] = N + 2 * r + 1
        buf = np.empty(shape, dtype=np.float32)
        sl0 = [slice(None)] * a.ndim
        sl0[ax] = slice(0, 1)
        buf[tuple(sl0)] = 0.0           # prepended zero
        sl1 = [slice(None)] * a.ndim
        sl1[ax] = slice(1, 1 + r)
        src1 = [slice(None)] * a.ndim
        src1[ax] = slice(0, 1)
        buf[tuple(sl1)] = a[tuple(src1)]  # left/top edge pad
        sl2 = [slice(None)] * a.ndim
        sl2[ax] = slice(1 + r, 1 + r + N)
        buf[tuple(sl2)] = a               # data
        sl3 = [slice(None)] * a.ndim
        sl3[ax] = slice(1 + r + N, None)
        src3 = [slice(None)] * a.ndim
        src3[ax] = slice(-1, None)
        buf[tuple(sl3)] = a[tuple(src3)]  # right/bottom edge pad
        cs = buf.cumsum(axis=ax)
        hi = [slice(None)] * a.ndim
        hi[ax] = slice(k, k + N)
        lo = [slice(None)] * a.ndim
        lo[ax] = slice(0, N)
        return (cs[tuple(hi)] - cs[tuple(lo)]) / k

    return _blur_axis(_blur_axis(img, 1), 0)


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
        # Stack shifted views and dot with kernel — no Python loop
        views = np.stack(
            [p.take(range(i, i + arr.shape[axis]), axis=axis) for i in range(len(k))],
            axis=0,
        )
        return np.tensordot(k, views, axes=([0], [0]))

    return _conv(_conv(img, 1), 0)


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
        # Hero particles (8 %) are larger for bright accent nodes
        sizes = np.array(
            [rng.uniform(4.0, 7.0) if rng.random() < 0.08
             else rng.uniform(1.2, 3.5)
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

    def _render_frame(self) -> QPixmap:
        """Render at fixed _RENDER_SIZE×_RENDER_SIZE; caller scales to widget."""
        S  = _RENDER_SIZE
        pd = self._pdata
        av = self._audio_level_smooth

        voice_swell = av * 0.18 if self._state == AgentState.SPEAKING else 0.0
        breath = 0.97 + 0.06 * self._breathe + voice_swell

        # ── Project sphere particles ──────────────────────────────────────
        R     = self._rotation_matrix()
        pos   = (pd.sphere_dirs * (pd.sphere_radii * breath)[:, None]) @ R.T
        psc   = _CAM_DIST / np.maximum(_CAM_DIST - pos[:, 2], 0.01)
        depth = np.clip((pos[:, 2] + 1.0) * 0.5, 0.0, 1.0)

        cx = cy = S * 0.5
        px = (pos[:, 0] * psc * _CLIP_SCALE * cx + cx).astype(np.int32)
        py = ((-pos[:, 1]) * psc * _CLIP_SCALE * cy + cy).astype(np.int32)

        tw    = 0.5 + 0.5 * np.sin(self._time * 2.4 + pd.sphere_phases)
        alpha = (0.55 + 0.45 * depth) * (0.60 + 0.40 * tw) * (pd.sphere_sizes / 3.0)
        alpha *= (1.0 + 0.9 * av)

        val     = np.clip(0.65 + 0.35 * depth + 0.15 * av, 0.0, 1.0).astype(np.float32)
        rgb     = _hsv_to_rgb(self._hue % 1.0, self._sat, val)
        weighted = rgb * alpha[:, None]

        buf  = np.zeros((S, S, 3), dtype=np.float32)
        mask = (px >= 0) & (px < S) & (py >= 0) & (py < S)
        if mask.any():
            idx = py[mask] * S + px[mask]
            for ch in range(3):
                buf[:, :, ch].flat += np.bincount(idx, weights=weighted[mask, ch], minlength=S * S)

        # ── Background stars ──────────────────────────────────────────────
        stw    = 0.35 + 0.65 * np.sin(self._time * 1.5 + pd.star_phases)
        s_alph = stw * 0.40
        spx = (pd.star_sx * cx + cx).astype(np.int32)
        spy = (pd.star_sy * cy + cy).astype(np.int32)
        s_mask = (spx >= 0) & (spx < S) & (spy >= 0) & (spy < S)
        if s_mask.any():
            s_idx  = spy[s_mask] * S + spx[s_mask]
            s_rgb  = np.tile([0.78, 0.90, 1.00], (s_mask.sum(), 1))
            s_w    = s_rgb * s_alph[s_mask, None]
            for ch in range(3):
                buf[:, :, ch].flat += np.bincount(s_idx, weights=s_w[:, ch], minlength=S * S)

        # ── Glow: 2-pass box blur for halo, 1-pass for bloom ────────────────
        g1_tmp = _box_blur1(buf, 2)
        g1     = _box_blur1(g1_tmp, 2)         # 2×box → smoother halo
        g2     = _box_blur1(buf, 24)            # wide bloom (single pass, fast)

        glow_mult = 26.0 + 18.0 * av
        result = np.clip(_BG + g1 * glow_mult + g2 * (2.8 + 3.0 * av), 0.0, 1.0)

        rgb8 = (result * 255.0).astype(np.uint8)
        a8   = np.full((S, S, 1), 255, dtype=np.uint8)
        rgba = np.ascontiguousarray(np.concatenate([rgb8, a8], axis=2))
        img  = QImage(rgba.data, S, S, S * 4, QImage.Format.Format_RGBA8888)
        return QPixmap.fromImage(img)

    def paintEvent(self, event) -> None:  # noqa: N802
        W, H   = self.width(), self.height()
        pixmap = self._render_frame()
        scaled = pixmap.scaled(
            W, H,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        x = (W - scaled.width()) // 2
        y = (H - scaled.height()) // 2
        painter.drawPixmap(x, y, scaled)
        painter.end()
