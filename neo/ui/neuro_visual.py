from __future__ import annotations

import math
import random

import numpy as np
import threading

from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtGui import QImage, QPainter, QPixmap
from PySide6.QtWidgets import QWidget

from ..core.state import AgentState

# ── State parameters ──────────────────────────────────────────────────────
# NEO's logo is green (#35e08a -> hue 0.416), and the sphere is the logo at
# this size, so every state stays in that family. States separate by shade,
# spin and brightness instead of by swapping to an unrelated colour -- the
# old blue LISTENING made NEO look like a different application mid-sentence.
# Error keeps red: it has to read as wrong at a glance, and that convention
# outranks matching the palette.
_LOGO_HUE = 0.416
_STATE_HUE: dict[AgentState, float] = {
    AgentState.IDLE:       _LOGO_HUE,
    AgentState.LISTENING:  0.448,  # cooler green, still unmistakably green
    AgentState.PROCESSING: 0.360,  # warmer lime
    AgentState.SPEAKING:   _LOGO_HUE,
    AgentState.ERROR:      0.000,  # red
}
_STATE_SAT: dict[AgentState, float] = {
    AgentState.IDLE:       0.76,   # the logo's own saturation
    AgentState.LISTENING:  0.68,
    AgentState.PROCESSING: 0.82,
    AgentState.SPEAKING:   0.72,
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
# A frame costs ~33 ms to render, so ticking faster than that only spins the
# timer against a render that is already busy. 30 fps is the ceiling; idle
# deliberately sits well under it to keep the machine quiet.
_STATE_TICK_MS: dict[AgentState, int] = {
    # Idle is where NEO spends nearly all of its life, and the idle sphere
    # turns at 0.0002 rad/ms -- 10 fps is visually indistinguishable there
    # and costs a third of a core instead of half of one.
    AgentState.IDLE:       100,
    AgentState.LISTENING:  40,
    AgentState.PROCESSING: 33,
    AgentState.SPEAKING:   33,
    AgentState.ERROR:      100,
}

# Camera/projection constants
_CAM_DIST    = 3.2   # perspective camera distance (sphere radius = 1)
_CLIP_SCALE  = 0.80  # sphere fills 80 % of widget height in clip space

# Measured at 680 px: scattering 4 000 particles costs 2.8 ms and 60 000
# costs 7.9 ms, while a single blur pass costs 7-14 ms no matter how many
# particles produced the image. Density is therefore nearly free and
# resolution is not -- and density is what the sphere was short of. At
# 4 000 the shell read as scattered specks ("144p"); this many fills it.
_N_SPHERE    = 60000
_N_STARS     = 900   # background star field (fixed, no rotation)

# Rendered 1:1 with the widget so nothing is ever upscaled -- the old fixed
# 600 px buffer was stretched to the widget's 680 px and that soft, slightly
# smeared result was most of what looked low-resolution. Capped because blur
# cost grows with the square of this.
_MIN_RENDER_SIZE = 360
_MAX_RENDER_SIZE = 900

# Upper end of the glow range the colour table covers. Above 1.0 so the
# brightest cores still clip channel by channel and burn toward white.
_GLOW_RANGE = 2.0

# Background colour — exact match for app theme #060a08
_BG = np.array([6 / 255, 10 / 255, 8 / 255], dtype=np.float32)


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


def _splat(flat: np.ndarray, S: int, fx: np.ndarray, fy: np.ndarray,
           weight: np.ndarray) -> None:
    """Add particles to a flat (S*S,) intensity buffer with bilinear
    sub-pixel weights.

    Rounding each particle to its nearest pixel is what made the sphere look
    pixelated: every dot became a hard-edged square that jittered by a whole
    pixel as the sphere turned. Spreading the energy over the four
    surrounding pixels gives sub-pixel positions, so dots are round and
    motion is smooth.
    """
    x0 = np.floor(fx).astype(np.int32)
    y0 = np.floor(fy).astype(np.int32)
    m = (x0 >= 0) & (x0 < S - 1) & (y0 >= 0) & (y0 < S - 1)
    if not m.any():
        return
    xi, yi = x0[m], y0[m]
    wx = (fx[m] - xi).astype(np.float32)
    wy = (fy[m] - yi).astype(np.float32)
    a = weight[m].astype(np.float32)

    base = yi.astype(np.int64) * S + xi
    idx = np.concatenate([base, base + 1, base + S, base + S + 1])
    w = np.concatenate([
        a * (1.0 - wx) * (1.0 - wy),
        a * wx * (1.0 - wy),
        a * (1.0 - wx) * wy,
        a * wx * wy,
    ])
    flat += np.bincount(idx, weights=w, minlength=S * S)


def _box_blur1(img: np.ndarray, r: int) -> np.ndarray:
    """Single-pass separable box blur. Output shape == input shape.

    Works on a 2-D intensity map, which is how it is used: blurring one
    channel instead of three is three times less cumsum, and the sphere is a
    single hue anyway, so colour is applied once after the blur.
    """
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

    return _blur_axis(_blur_axis(img, img.ndim - 1), 0)


def _wide_bloom(img: np.ndarray, S: int, factor: int = 4) -> np.ndarray:
    """The broad atmospheric glow, computed at 1/factor resolution.

    A bloom this wide carries no detail finer than the blur radius, so
    downsampling first costs nothing visually and the blur then runs over
    sixteen times fewer pixels. Measured at 680 px: 7.6 ms full resolution
    against 3.8 ms this way, and that saving is what pays for rendering the
    rest of the frame at the widget's native size.
    """
    small = S // factor
    trimmed = img[: small * factor, : small * factor]
    low = trimmed.reshape(small, factor, small, factor).mean(axis=(1, 3))
    low = _box_blur1(low, max(1, 20 // factor))
    up = np.repeat(np.repeat(low, factor, axis=0), factor, axis=1)
    if up.shape[0] == S:
        return up
    # Integer division left a few pixels uncovered; pad with the edge row
    # and column rather than returning a smaller array.
    pad = S - up.shape[0]
    return np.pad(up, ((0, pad), (0, pad)), mode="edge")


def _gauss_blur(img: np.ndarray, r: int) -> np.ndarray:
    """2 × box blur ≈ triangle filter — round halos without the third pass,
    which cost more than it added visually."""
    return _box_blur1(_box_blur1(img, r), r)


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
        # A thin shell reads as a sphere; a thick one projects to a filled
        # disc with no silhouette. Most particles sit near r=1 so the limb
        # crowds and draws the outline, with a light interior haze behind it.
        radii = np.array(
            [rng.gauss(0.97, 0.03) if rng.random() < 0.88
             else rng.uniform(0.35, 0.92)
             for _ in range(n_sphere)],
            dtype=np.float32,
        ).clip(0.30, 1.03)
        # A few bright nodes over a dim field, rather than one uniform
        # brightness: that contrast is most of what makes it read as depth.
        # The bright fraction is small because at 60 000 particles even 1 %
        # is 600 highlights, which is already plenty of sparkle.
        sizes = np.array(
            [rng.uniform(6.0, 11.0) if rng.random() < 0.012
             else rng.uniform(0.35, 1.5)
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
    _FOCUS_SIZE   = 680

    _frame_ready = Signal(QPixmap)  # emitted from render thread → main thread

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
        self._audio_level        = 0.0
        self._audio_level_smooth = 0.0

        self._pdata = _ParticleData(_N_SPHERE, _N_STARS)

        # Widget must be visually transparent so no Qt background colour
        # paints on top of (or slightly differs from) our rendered _BG.
        self.setStyleSheet("background: transparent; border: none;")

        self._cached_pixmap: QPixmap | None = None
        self._render_lock = threading.Lock()   # one render thread at a time
        self._rendering   = False

        self._frame_ready.connect(self._on_frame_ready)

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

    def _on_frame_ready(self, pixmap: QPixmap) -> None:
        self._cached_pixmap = pixmap
        self._rendering = False
        self.update()

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

        target = self._audio_level
        alpha  = 0.40 if target > self._audio_level_smooth else 0.12
        self._audio_level_smooth += (target - self._audio_level_smooth) * alpha

        # Kick off a render only if the previous one finished
        if not self._rendering:
            self._rendering = True
            # Snapshot mutable state for the thread
            snap = (
                self._state, self._hue, self._sat, self._yaw,
                self._breathe, self._time, self._audio_level_smooth,
                self._render_size(),
            )
            t = threading.Thread(target=self._render_thread, args=(snap,), daemon=True)
            t.start()

    def showEvent(self, event) -> None:  # noqa: N802
        super().showEvent(event)
        if not self._timer.isActive():
            self._timer.start()

    def hideEvent(self, event) -> None:  # noqa: N802
        self._timer.stop()
        super().hideEvent(event)

    def _render_thread(self, snap: tuple) -> None:
        try:
            pixmap = self._render_frame(snap)
            self._frame_ready.emit(pixmap)
        except RuntimeError:
            # Widget was destroyed while this frame was still rendering --
            # normal on shutdown, and the frame has nowhere to go anyway.
            pass

    # ── Rendering ─────────────────────────────────────────────────────────
    def _rotation_matrix(self, yaw: float) -> np.ndarray:
        pitch = math.sin(yaw * 0.38) * 0.32
        cy, sy = math.cos(yaw),   math.sin(yaw)
        cp, sp = math.cos(pitch), math.sin(pitch)
        return np.array([
            [cy,  sy*sp,  sy*cp],
            [0,   cp,    -sp   ],
            [-sy, cy*sp,  cy*cp],
        ], dtype=np.float32)

    def _render_size(self) -> int:
        """Pixel size of the frame to render: the widget's own size, so the
        result is blitted 1:1 instead of being stretched."""
        side = max(self.width(), self.height()) * self.devicePixelRatioF()
        return int(max(_MIN_RENDER_SIZE, min(_MAX_RENDER_SIZE, side)))

    def _render_frame(self, snap: tuple) -> QPixmap:
        """Pure function — runs on a background thread, no self mutation."""
        state, hue, sat, yaw, breathe, time_, av, S = snap
        pd = self._pdata

        voice_swell = av * 0.18 if state == AgentState.SPEAKING else 0.0
        breath = 0.97 + 0.06 * breathe + voice_swell

        R     = self._rotation_matrix(yaw)
        # Rotate the unit directions, then scale: keeping the unit normal
        # around is what makes the rim lighting below possible.
        ndir  = pd.sphere_dirs @ R.T
        pos   = ndir * (pd.sphere_radii * breath)[:, None]
        psc   = _CAM_DIST / np.maximum(_CAM_DIST - pos[:, 2], 0.01)
        depth = np.clip((pos[:, 2] + 1.0) * 0.5, 0.0, 1.0)

        cx = cy = S * 0.5
        fx = pos[:, 0] * psc * _CLIP_SCALE * cx + cx
        fy = (-pos[:, 1]) * psc * _CLIP_SCALE * cy + cy

        tw = 0.5 + 0.5 * np.sin(time_ * 2.4 + pd.sphere_phases)

        # Rim (Fresnel) lighting: a particle is brightest where the shell
        # turns away from the camera, i.e. at the silhouette. That bright
        # outline is what reads as "hollow sphere" -- shading by depth alone
        # lit the middle and left the thing looking like a flat disc of
        # confetti no matter how the particles were distributed.
        rim   = (1.0 - np.abs(ndir[:, 2])) ** 1.7
        front = 0.32 + 0.68 * depth       # back hemisphere sits behind
        shade = (0.05 + 0.95 * rim) * front

        # Shading rides in the intensity, so colour stays uniform and is
        # applied once after the blur.
        alpha = shade * (0.60 + 0.40 * tw) * (pd.sphere_sizes / 3.0)
        alpha *= (1.0 + 0.9 * av)

        # One single-channel intensity map for sphere + stars: blurring one
        # channel instead of three is the difference between ~90 ms and
        # ~30 ms a frame at 600 px.
        flat = np.zeros(S * S, dtype=np.float32)
        _splat(flat, S, fx, fy, alpha)

        stw    = 0.35 + 0.65 * np.sin(time_ * 1.5 + pd.star_phases)
        _splat(flat, S, pd.star_sx * cx + cx, pd.star_sy * cy + cy, stw * 0.40)

        buf = flat.reshape(S, S)

        # Two passes, not one: a single box blur is a square kernel, so
        # every particle came out as a little hard-edged square. Two passes
        # convolve to a triangle kernel, which is round enough that the
        # dots stop having corners -- the bilinear splat softens where the
        # dot sits, but only the kernel decides what shape it is.
        g1 = _gauss_blur(buf, 1)
        # The atmosphere is low-frequency by definition, so computing it at
        # a quarter of the resolution and scaling back up is visually
        # identical and half the cost.
        g2 = _wide_bloom(buf, S)

        glow = g1 * (11.0 + 8.0 * av) + g2 * (5.0 + 5.0 * av)
        colour = _hsv_to_rgb(hue % 1.0, sat, np.array([1.0], np.float32))[0]

        # Colourize through a 256-entry lookup table rather than a full
        # (S, S, 3) float broadcast. The glow is one channel, so the whole
        # colour mapping fits in a table and the frame becomes a single
        # gather -- 18 ms of broadcast, clip and float-to-byte conversion
        # down to about 5 ms, which is what pays for the extra particles.
        lut = np.empty((256, 4), dtype=np.uint8)
        levels = np.linspace(0.0, _GLOW_RANGE, 256, dtype=np.float32)
        lut[:, :3] = np.clip(
            _BG[None, :] * 255.0 + levels[:, None] * (colour * 255.0)[None, :],
            0.0, 255.0,
        ).astype(np.uint8)
        lut[:, 3] = 255

        # Quantized over 0.._GLOW_RANGE rather than 0..1 so the brightest
        # cores still clip per channel and go white-hot, the way they did
        # when this was computed in floating point.
        index = np.clip(glow * (255.0 / _GLOW_RANGE), 0.0, 255.0).astype(np.uint8)
        rgba = np.ascontiguousarray(lut[index])
        img = QImage(rgba.data, S, S, S * 4, QImage.Format.Format_RGBA8888)
        return QPixmap.fromImage(img.copy())

    def paintEvent(self, event) -> None:  # noqa: N802
        W, H = self.width(), self.height()
        pixmap = self._cached_pixmap
        if pixmap is None:
            return  # first frame not ready; parent background shows through
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
