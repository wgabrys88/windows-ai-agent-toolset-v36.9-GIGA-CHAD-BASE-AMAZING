"""FRANZ capture — GDI screen capture with VLM-optimised action annotations.

Standalone subprocess for the FRANZ visual-agent system.  Captures the
primary monitor (or a focused sub-region) via Win32 GDI, applies annotations
onto the native-resolution buffer, optionally resizes to target dimensions,
encodes as PNG, and emits base64.

Annotation Design
~~~~~~~~~~~~~~~~~
Annotations are optimised for a **2B-parameter vision-language model**,
not for human readability:

- **Large red circles** (≥56px diameter) at action points — guaranteed to
  span multiple vision-encoder patches.
- **Sequential white numerals** at each action — the model reads "1", "2",
  "3" which map directly to the ordered action list in the text prompt.
- **Thick white outlines** on all marks — high contrast on any desktop
  background.
- **Solid (not dashed) lines** for trails and drag arrows — continuous
  features are more salient to patch-based encoders.
- **Pure red (255, 0, 0)** as the annotation colour — the single most
  common "overlay" colour in pretraining object-detection datasets.

Focus Mode
~~~~~~~~~~
When the action list contains a ``focus(x1, y1, x2, y2)`` entry (0–1000
virtual coordinates), only the specified screen region is captured at its
native pixel dimensions.  No resize is performed — the model receives
exactly the cropped pixels, minimising inference cost.

``focus`` is extracted *before* the pipeline runs and modifies the GDI
capture origin and extent.  The rest of the pipeline sees a smaller
"screen" and operates identically.  When ``focus`` is present, no other
action annotations are expected (only calibration runs).

Protocol
--------
Read one JSON object from **stdin**, write one raw base64 PNG string to
**stdout**, then exit.

Input JSON fields::

    actions : list[str]   — e.g. ``["left_click(500, 300)", "type('hello')"]``
    width   : int         — target image width  (0 = use native/cropped size)
    height  : int         — target image height (0 = use native/cropped size)
    marks   : bool        — whether to draw annotations (default True)

Output::

    ASCII base64 string of a 32-bit RGBA PNG.

Pipeline::

    (optional) extract focus region from actions
    → GDI BitBlt (native res, full screen or focused region)
    → BGRA→RGBA (force α=0xFF)
    → Canvas mutations (calibration fill + numbered action annotations)
    → RGBA→BGRA (force α=0xFF — no annotation alpha leaks into GDI resize)
    → GDI StretchBlt (downsample to target, skipped if already correct size)
    → BGRA→RGBA (force α=0xFF)
    → PNG encode
    → base64

Alpha Integrity
~~~~~~~~~~~~~~~
Semi-transparent annotation colours are alpha-blended onto the opaque
screenshot buffer by ``Canvas.put()``.  The resulting RGB values correctly
reflect the blend, but alpha residuals (α=253, α=254) are artefacts of
the compositing math.  To prevent GDI's HALFTONE ``StretchBlt`` from
using these non-uniform alpha values during interpolation (which would
subtly corrupt RGB), ``_rgba_to_bgra()`` forces all alpha back to 0xFF
before handing the buffer to GDI.  This guarantees single source of truth:
final RGB is determined solely by the native-resolution Canvas content.

Environment: Windows 11, Python 3.13+, no third-party dependencies.
"""

from __future__ import annotations

import ast
import base64
import ctypes
import ctypes.wintypes
import json
import math
import struct
import sys
import zlib
from pathlib import Path
from typing import Final

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

_CALIBRATION: Final[bool] = True  # Set False for production
_CAL_PERSIST_ACTIONS: Final[bool] = True  # If True, accumulate marks across turns (calibration only)
_CAL_ACTIONS_FILE: Final[Path] = Path(__file__).with_name("cal_actions.json")

# Simple calibration reference circle (0–1000 coords)
_CAL_CIRCLE_X: Final[int] = 500
_CAL_CIRCLE_Y: Final[int] = 500
_CAL_CIRCLE_R: Final[int] = 60

type Color = tuple[int, int, int, int]
type Point = tuple[int, int]

# VLM-optimised annotation colours
MARK_FILL:    Final[Color] = (255, 0, 0, 180)      # semi-transparent red
MARK_OUTLINE: Final[Color] = (255, 255, 255, 230)   # near-opaque white outline
MARK_TEXT:    Final[Color] = (255, 255, 255, 255)    # pure white for numerals
TRAIL_COLOR:  Final[Color] = (255, 0, 0, 120)        # lighter red for trails
BLACK_OPAQUE: Final[Color] = (0, 0, 0, 255)

# ---------------------------------------------------------------------------
# Win32 constants & structures
# ---------------------------------------------------------------------------

_BI_RGB:     Final[int] = 0
_DIB_RGB:    Final[int] = 0
_SRCCOPY:    Final[int] = 0x00CC0020
_CAPTUREBLT: Final[int] = 0x40000000
_HALFTONE:   Final[int] = 4


class _BITMAPINFOHEADER(ctypes.Structure):
    _fields_ = [
        ("biSize",          ctypes.wintypes.DWORD),
        ("biWidth",         ctypes.wintypes.LONG),
        ("biHeight",        ctypes.wintypes.LONG),
        ("biPlanes",        ctypes.wintypes.WORD),
        ("biBitCount",      ctypes.wintypes.WORD),
        ("biCompression",   ctypes.wintypes.DWORD),
        ("biSizeImage",     ctypes.wintypes.DWORD),
        ("biXPelsPerMeter", ctypes.wintypes.LONG),
        ("biYPelsPerMeter", ctypes.wintypes.LONG),
        ("biClrUsed",       ctypes.wintypes.DWORD),
        ("biClrImportant",  ctypes.wintypes.DWORD),
    ]


class _BITMAPINFO(ctypes.Structure):
    _fields_ = [
        ("bmiHeader", _BITMAPINFOHEADER),
        ("bmiColors", ctypes.wintypes.DWORD * 3),
    ]


# ---------------------------------------------------------------------------
# Win32 library init & DPI
# ---------------------------------------------------------------------------

_shcore: Final = ctypes.WinDLL("shcore", use_last_error=True)
_shcore.SetProcessDpiAwareness(2)

_user32: Final = ctypes.WinDLL("user32", use_last_error=True)
_gdi32:  Final = ctypes.WinDLL("gdi32", use_last_error=True)

_screen_w: Final[int] = _user32.GetSystemMetrics(0)
_screen_h: Final[int] = _user32.GetSystemMetrics(1)

# ---------------------------------------------------------------------------
# GDI helpers
# ---------------------------------------------------------------------------


def _make_bmi(w: int, h: int) -> _BITMAPINFO:
    """Build a top-down 32-bpp BITMAPINFO for *w*×*h*."""
    bmi = _BITMAPINFO()
    bmi.bmiHeader.biSize = ctypes.sizeof(_BITMAPINFOHEADER)
    bmi.bmiHeader.biWidth = w
    bmi.bmiHeader.biHeight = -h
    bmi.bmiHeader.biPlanes = 1
    bmi.bmiHeader.biBitCount = 32
    bmi.bmiHeader.biCompression = _BI_RGB
    return bmi


def _capture_bgra(
    cw: int, ch: int, src_x: int = 0, src_y: int = 0,
) -> bytes:
    """Grab a *cw*×*ch* region of the screen starting at (*src_x*, *src_y*).

    When *src_x* and *src_y* are both 0 and *cw*×*ch* equals the screen
    dimensions, this captures the full primary monitor (original behaviour).
    When a sub-region is specified, only those pixels are captured — the
    resulting buffer is *cw*×*ch*, not the full screen.
    """
    sdc = _user32.GetDC(0)
    memdc = _gdi32.CreateCompatibleDC(sdc)
    bits = ctypes.c_void_p()
    hbmp = _gdi32.CreateDIBSection(
        sdc, ctypes.byref(_make_bmi(cw, ch)),
        _DIB_RGB, ctypes.byref(bits), None, 0,
    )
    old = _gdi32.SelectObject(memdc, hbmp)
    _gdi32.BitBlt(memdc, 0, 0, cw, ch, sdc, src_x, src_y, _SRCCOPY | _CAPTUREBLT)
    raw = bytes((ctypes.c_ubyte * (cw * ch * 4)).from_address(bits.value))
    _gdi32.SelectObject(memdc, old)
    _gdi32.DeleteObject(hbmp)
    _gdi32.DeleteDC(memdc)
    _user32.ReleaseDC(0, sdc)
    return raw


def _resize_bgra(
    src: bytes, sw: int, sh: int, dw: int, dh: int,
) -> bytes:
    """Resize a BGRA buffer from *sw*×*sh* to *dw*×*dh* using GDI HALFTONE.

    Handles both downsampling and upsampling.
    """
    sdc = _user32.GetDC(0)
    src_dc = _gdi32.CreateCompatibleDC(sdc)
    dst_dc = _gdi32.CreateCompatibleDC(sdc)
    src_bmp = _gdi32.CreateCompatibleBitmap(sdc, sw, sh)
    old_src = _gdi32.SelectObject(src_dc, src_bmp)
    _gdi32.SetDIBits(
        sdc, src_bmp, 0, sh, src,
        ctypes.byref(_make_bmi(sw, sh)), _DIB_RGB,
    )
    dst_bits = ctypes.c_void_p()
    dst_bmp = _gdi32.CreateDIBSection(
        sdc, ctypes.byref(_make_bmi(dw, dh)),
        _DIB_RGB, ctypes.byref(dst_bits), None, 0,
    )
    old_dst = _gdi32.SelectObject(dst_dc, dst_bmp)
    _gdi32.SetStretchBltMode(dst_dc, _HALFTONE)
    _gdi32.SetBrushOrgEx(dst_dc, 0, 0, None)
    _gdi32.StretchBlt(dst_dc, 0, 0, dw, dh, src_dc, 0, 0, sw, sh, _SRCCOPY)
    raw = bytearray(
        (ctypes.c_ubyte * (dw * dh * 4)).from_address(dst_bits.value),
    )
    # Safety net: force opaque (input should already be α=0xFF from _rgba_to_bgra)
    raw[3::4] = b"\xff" * (dw * dh)
    out = bytes(raw)
    _gdi32.SelectObject(dst_dc, old_dst)
    _gdi32.SelectObject(src_dc, old_src)
    _gdi32.DeleteObject(dst_bmp)
    _gdi32.DeleteObject(src_bmp)
    _gdi32.DeleteDC(dst_dc)
    _gdi32.DeleteDC(src_dc)
    _user32.ReleaseDC(0, sdc)
    return out


# ---------------------------------------------------------------------------
# Pixel-format conversions (lossless channel swaps, alpha normalised)
# ---------------------------------------------------------------------------


def _bgra_to_rgba(bgra: bytes) -> bytes:
    """Swap B↔R channels and force alpha to 0xFF.

    GDI does not guarantee meaningful alpha values in captured bitmaps.
    Forcing 0xFF produces a clean opaque buffer for Canvas blending.
    """
    n = len(bgra)
    out = bytearray(n)
    out[0::4] = bgra[2::4]
    out[1::4] = bgra[1::4]
    out[2::4] = bgra[0::4]
    out[3::4] = b"\xff" * (n // 4)
    return bytes(out)


def _rgba_to_bgra(rgba: bytes) -> bytes:
    """Swap R↔B channels and force alpha to 0xFF for GDI consumption.

    After Canvas annotation, some pixels may have alpha residuals
    (α=253, α=254) from semi-transparent blending.  GDI's HALFTONE
    StretchBlt interpolates all four channels including alpha — if alpha
    varies across adjacent pixels, it subtly corrupts RGB interpolation.

    Forcing α=0xFF here guarantees single source of truth: the resize
    operates on colour channels only, producing results determined
    solely by the native-resolution Canvas RGB content.
    """
    n = len(rgba)
    out = bytearray(n)
    out[0::4] = rgba[2::4]   # B ← R
    out[1::4] = rgba[1::4]   # G ← G
    out[2::4] = rgba[0::4]   # R ← B
    out[3::4] = b"\xff" * (n // 4)  # Force opaque — no annotation alpha leaks to GDI
    return bytes(out)


# ---------------------------------------------------------------------------
# PNG encoding
# ---------------------------------------------------------------------------


def _encode_png(rgba: bytes, w: int, h: int) -> bytes:
    """Minimal valid PNG (RGBA, filter=None, zlib level 6)."""
    stride = w * 4
    raw = bytearray()
    for y in range(h):
        raw.append(0)
        raw.extend(rgba[y * stride : (y + 1) * stride])
    ihdr = struct.pack(">IIBBBBB", w, h, 8, 6, 0, 0, 0)
    idat = zlib.compress(bytes(raw), 6)

    def chunk(tag: bytes, body: bytes) -> bytes:
        crc = zlib.crc32(tag + body) & 0xFFFFFFFF
        return struct.pack(">I", len(body)) + tag + body + struct.pack(">I", crc)

    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", ihdr)
        + chunk(b"IDAT", idat)
        + chunk(b"IEND", b"")
    )


# ---------------------------------------------------------------------------
# Canvas — 2-D drawing surface over a flat RGBA bytearray
# ---------------------------------------------------------------------------


class Canvas:
    """Immediate-mode 2-D drawing surface backed by a flat RGBA buffer.

    All public draw methods mutate the buffer in-place and return ``self``
    for optional chaining.

    Alpha blending in ``put()`` assumes the background alpha is always
    0xFF, which is guaranteed by ``_bgra_to_rgba()`` and ``fill_solid()``.
    The blended RGB values are correct; alpha residuals are artefacts
    that get normalised to 0xFF by ``_rgba_to_bgra()`` before resize.
    """

    __slots__ = ("buf", "w", "h")

    def __init__(self, rgba: bytes | bytearray, w: int, h: int) -> None:
        self.buf = bytearray(rgba) if isinstance(rgba, bytes) else rgba
        self.w = w
        self.h = h

    # -- whole-buffer operations --------------------------------------------

    def fill_solid(self, c: Color) -> Canvas:
        """Replace every pixel with *c*.  Calibration = just another annotation."""
        self.buf[:] = bytes(c) * (self.w * self.h)
        return self

    # -- pixel ops ----------------------------------------------------------

    def put(self, x: int, y: int, c: Color) -> None:
        """Set a single pixel with alpha blending (bounds-checked).

        Uses straight-alpha "over" compositing.  Correct only when the
        existing background alpha is 0xFF (always true in this pipeline).
        """
        if 0 <= x < self.w and 0 <= y < self.h:
            i = (y * self.w + x) << 2
            sa = c[3] / 255.0
            da = 1.0 - sa
            self.buf[i]     = min(255, int(c[0] * sa + self.buf[i]     * da))
            self.buf[i + 1] = min(255, int(c[1] * sa + self.buf[i + 1] * da))
            self.buf[i + 2] = min(255, int(c[2] * sa + self.buf[i + 2] * da))
            self.buf[i + 3] = min(255, int(c[3]      + self.buf[i + 3] * da))

    def put_opaque(self, x: int, y: int, c: Color) -> None:
        """Set a single pixel without blending (bounds-checked)."""
        if 0 <= x < self.w and 0 <= y < self.h:
            i = (y * self.w + x) << 2
            self.buf[i]     = c[0]
            self.buf[i + 1] = c[1]
            self.buf[i + 2] = c[2]
            self.buf[i + 3] = c[3]

    def put_thick(self, x: int, y: int, c: Color, t: int = 1) -> None:
        """Set a *t*×*t* square of pixels centred on (*x*, *y*)."""
        half = t >> 1
        for dy in range(-half, half + 1):
            for dx in range(-half, half + 1):
                self.put(x + dx, y + dy, c)

    # -- primitives ---------------------------------------------------------

    def line(
        self, x1: int, y1: int, x2: int, y2: int, c: Color, t: int = 3,
    ) -> Canvas:
        """Bresenham line with thickness *t*."""
        dx, dy = abs(x2 - x1), abs(y2 - y1)
        sx = 1 if x1 < x2 else -1
        sy = 1 if y1 < y2 else -1
        err = dx - dy
        x, y = x1, y1
        while True:
            self.put_thick(x, y, c, t)
            if x == x2 and y == y2:
                break
            e2 = err << 1
            if e2 > -dy:
                err -= dy
                x += sx
            if e2 < dx:
                err += dx
                y += sy
        return self

    def circle(
        self, cx: int, cy: int, r: int, c: Color,
        filled: bool = False, thickness: int = 3,
    ) -> Canvas:
        """Circle — filled disc or outline with configurable *thickness*."""
        r2_outer = r * r
        r2_inner = max(0, (r - thickness)) ** 2
        for oy in range(-r, r + 1):
            for ox in range(-r, r + 1):
                d2 = ox * ox + oy * oy
                if filled:
                    if d2 <= r2_outer:
                        self.put(cx + ox, cy + oy, c)
                elif r2_inner <= d2 <= r2_outer:
                    self.put(cx + ox, cy + oy, c)
        return self

    def fill_polygon(self, pts: list[Point], c: Color) -> Canvas:
        """Scanline-fill an arbitrary polygon (even-odd rule)."""
        if len(pts) < 3:
            return self
        ys = [p[1] for p in pts]
        lo_y = max(0, min(ys))
        hi_y = min(self.h - 1, max(ys))
        n = len(pts)
        for y in range(lo_y, hi_y + 1):
            nodes: list[int] = []
            j = n - 1
            for i in range(n):
                yi, yj = pts[i][1], pts[j][1]
                if (yi < y <= yj) or (yj < y <= yi):
                    nodes.append(
                        int(pts[i][0] + (y - yi) / (yj - yi) * (pts[j][0] - pts[i][0]))
                    )
                j = i
            nodes.sort()
            for k in range(0, len(nodes) - 1, 2):
                for x in range(max(0, nodes[k]), min(self.w, nodes[k + 1] + 1)):
                    self.put(x, y, c)
        return self

    def rect(
        self, x: int, y: int, rw: int, rh: int, c: Color, t: int = 3,
    ) -> Canvas:
        """Axis-aligned rectangle outline."""
        self.line(x, y, x + rw, y, c, t)
        self.line(x + rw, y, x + rw, y + rh, c, t)
        self.line(x + rw, y + rh, x, y + rh, c, t)
        self.line(x, y + rh, x, y, c, t)
        return self

    # -- composites ---------------------------------------------------------

    def arrowhead(
        self, x1: int, y1: int, x2: int, y2: int, c: Color,
        length: int = 20, angle_deg: float = 28.0,
    ) -> Canvas:
        """Filled triangular arrowhead at (*x2*, *y2*)."""
        angle = math.atan2(y2 - y1, x2 - x1)
        ha = math.radians(angle_deg)
        lx = int(x2 - length * math.cos(angle - ha))
        ly = int(y2 - length * math.sin(angle - ha))
        rx = int(x2 - length * math.cos(angle + ha))
        ry = int(y2 - length * math.sin(angle + ha))
        return self.fill_polygon([(x2, y2), (lx, ly), (rx, ry)], c)

    def arrow(
        self, x1: int, y1: int, x2: int, y2: int, c: Color,
        t: int = 6, head_len: int = 24, head_deg: float = 28.0,
    ) -> Canvas:
        """Solid thick line + filled arrowhead."""
        self.line(x1, y1, x2, y2, c, t)
        return self.arrowhead(x1, y1, x2, y2, c, head_len, head_deg)

    def numeral(
        self, cx: int, cy: int, n: int,
        fill: Color, outline: Color, scale: int = 4,
    ) -> Canvas:
        """Render an integer (one or more digits) centred on (*cx*, *cy*).

        Uses a 5×7 bitmap font scaled by *scale*.  At scale=4, each
        digit is 20×28 pixels — spans ~2×2 vision-encoder patches.
        Multi-digit numbers are supported (e.g. 10, 11, 12...).
        """
        digits = str(n)
        glyph_w = 5 * scale
        gap = 1 * scale  # 1-pixel gap between digits, scaled
        total_w = len(digits) * glyph_w + (len(digits) - 1) * gap
        start_x = cx - total_w // 2 + glyph_w // 2
        for i, ch in enumerate(digits):
            digit_cx = start_x + i * (glyph_w + gap)
            self._render_single_digit(digit_cx, cy, int(ch), fill, outline, scale)
        return self

    def _render_single_digit(
        self, cx: int, cy: int, d: int,
        fill: Color, outline: Color, scale: int,
    ) -> None:
        """Render one digit 0–9 centred on (*cx*, *cy*)."""
        gw = 5 * scale
        gh = 7 * scale
        ox = cx - gw // 2
        oy = cy - gh // 2
        digit = _DIGITS[d]
        # Outline pass — offset in 8 directions, using put_opaque
        for ddy in (-1, 0, 1):
            for ddx in (-1, 0, 1):
                if ddx == 0 and ddy == 0:
                    continue
                for ri, row in enumerate(digit):
                    for ci in range(5):
                        if row & (1 << (4 - ci)):
                            for sy in range(scale):
                                for sx in range(scale):
                                    self.put_opaque(
                                        ox + ci * scale + sx + ddx * 2,
                                        oy + ri * scale + sy + ddy * 2,
                                        outline,
                                    )
        # Fill pass — always opaque overwrite on top of outline
        for ri, row in enumerate(digit):
            for ci in range(5):
                if row & (1 << (4 - ci)):
                    for sy in range(scale):
                        for sx in range(scale):
                            self.put_opaque(
                                ox + ci * scale + sx,
                                oy + ri * scale + sy,
                                fill,
                            )

    def to_bytes(self) -> bytes:
        """Return the buffer as immutable ``bytes``."""
        return bytes(self.buf)


# ---------------------------------------------------------------------------
# 5×7 bitmap font for digits 0–9 (each row is a 5-bit bitmask)
# ---------------------------------------------------------------------------

_DIGITS: Final[list[list[int]]] = [
    # 0
    [0b01110,
     0b10001,
     0b10011,
     0b10101,
     0b11001,
     0b10001,
     0b01110],
    # 1
    [0b00100,
     0b01100,
     0b00100,
     0b00100,
     0b00100,
     0b00100,
     0b01110],
    # 2
    [0b01110,
     0b10001,
     0b00001,
     0b00110,
     0b01000,
     0b10000,
     0b11111],
    # 3
    [0b01110,
     0b10001,
     0b00001,
     0b00110,
     0b00001,
     0b10001,
     0b01110],
    # 4
    [0b00010,
     0b00110,
     0b01010,
     0b10010,
     0b11111,
     0b00010,
     0b00010],
    # 5
    [0b11111,
     0b10000,
     0b11110,
     0b00001,
     0b00001,
     0b10001,
     0b01110],
    # 6
    [0b00110,
     0b01000,
     0b10000,
     0b11110,
     0b10001,
     0b10001,
     0b01110],
    # 7
    [0b11111,
     0b00001,
     0b00010,
     0b00100,
     0b01000,
     0b01000,
     0b01000],
    # 8
    [0b01110,
     0b10001,
     0b10001,
     0b01110,
     0b10001,
     0b10001,
     0b01110],
    # 9
    [0b01110,
     0b10001,
     0b10001,
     0b01111,
     0b00001,
     0b00010,
     0b01100],
]


# ---------------------------------------------------------------------------
# Coordinate helper
# ---------------------------------------------------------------------------


def _norm(coord: int, extent: int) -> int:
    """Map a 0–1000 virtual coordinate to pixel space."""
    return int((coord / 1000.0) * extent)


# ---------------------------------------------------------------------------
# Focus extraction — pre-pipeline scan
# ---------------------------------------------------------------------------


def _extract_focus(
    actions: list[str],
) -> tuple[int, int, int, int] | None:
    """Scan actions for a ``focus(x1, y1, x2, y2)`` entry.

    Returns pixel-space ``(src_x, src_y, crop_w, crop_h)`` or ``None``.
    Coordinates are in 0–1000 virtual space, mapped to native screen pixels.
    The ``focus`` entry is consumed (removed from the list) so it does not
    reach the annotation dispatcher.
    """
    for i, line in enumerate(actions):
        parsed = _parse_action_args(line)
        if parsed is None:
            continue
        name, args = parsed
        if name == "focus" and len(args) >= 4:
            x1 = _norm(int(args[0]), _screen_w)
            y1 = _norm(int(args[1]), _screen_h)
            x2 = _norm(int(args[2]), _screen_w)
            y2 = _norm(int(args[3]), _screen_h)
            # Normalise: ensure x1<x2, y1<y2, clamp to screen
            x1, x2 = max(0, min(x1, x2)), min(_screen_w, max(x1, x2))
            y1, y2 = max(0, min(y1, y2)), min(_screen_h, max(y1, y2))
            crop_w = x2 - x1
            crop_h = y2 - y1
            if crop_w > 0 and crop_h > 0:
                actions.pop(i)  # consume the focus action
                return (x1, y1, crop_w, crop_h)
    return None


# ---------------------------------------------------------------------------
# Annotation composers (all operate on Canvas at native resolution)
# ---------------------------------------------------------------------------


def _mark_point(
    cv: Canvas, x: int, y: int, n: int,
    px: int | None, py: int | None,
) -> None:
    """Standard point marker: outline circle + fill circle + numeral.

    Used as the base for all click-type annotations.
    """
    # Movement trail from previous action
    if px is not None and py is not None and math.hypot(x - px, y - py) > 30:
        cv.line(px, py, x, y, TRAIL_COLOR, t=4)

    # White outline circle (larger)
    cv.circle(x, y, 32, MARK_OUTLINE, filled=True)
    # Red fill circle
    cv.circle(x, y, 28, MARK_FILL, filled=True)
    # White numeral
    cv.numeral(x, y, n, MARK_TEXT, BLACK_OPAQUE, scale=4)


def _annotate_left_click(
    cv: Canvas, x: int, y: int, n: int,
    px: int | None, py: int | None,
) -> None:
    """Single click: one numbered circle."""
    _mark_point(cv, x, y, n, px, py)


def _annotate_right_click(
    cv: Canvas, x: int, y: int, n: int,
    px: int | None, py: int | None,
) -> None:
    """Right click: numbered circle + small white square modifier."""
    _mark_point(cv, x, y, n, px, py)
    # Small square in top-right to distinguish from left click
    cv.rect(x + 20, y - 36, 16, 16, MARK_TEXT, t=3)


def _annotate_double_click(
    cv: Canvas, x: int, y: int, n: int,
    px: int | None, py: int | None,
) -> None:
    """Double click: numbered circle + outer ring."""
    _mark_point(cv, x, y, n, px, py)
    # Outer ring signals "double"
    cv.circle(x, y, 42, MARK_OUTLINE, thickness=3)


def _annotate_drag(
    cv: Canvas,
    x1: int, y1: int, x2: int, y2: int, n: int,
    px: int | None, py: int | None,
) -> None:
    """Drag: numbered circle at start + thick arrow to end + circle at end."""
    # Trail from previous action to drag start
    if px is not None and py is not None and math.hypot(x1 - px, y1 - py) > 30:
        cv.line(px, py, x1, y1, TRAIL_COLOR, t=4)

    # Start marker
    cv.circle(x1, y1, 20, MARK_OUTLINE, filled=True)
    cv.circle(x1, y1, 16, MARK_FILL, filled=True)
    cv.numeral(x1, y1, n, MARK_TEXT, BLACK_OPAQUE, scale=3)

    # Thick arrow from start to end
    cv.arrow(x1, y1, x2, y2, MARK_FILL, t=6, head_len=28, head_deg=25.0)

    # End marker (outline only — no numeral)
    cv.circle(x2, y2, 20, MARK_OUTLINE, thickness=4)
    cv.circle(x2, y2, 16, MARK_FILL, thickness=3)


def _annotate_type(
    cv: Canvas, x: int, y: int, n: int,
) -> None:
    """Type action: red rectangle + numeral inside.

    Positioned at the last known cursor location (from previous click/drag).
    If ``type`` is the first action in the list, it is silently skipped
    because there is no known cursor position to anchor the marker.
    """
    pad = 30
    cv.rect(x - pad, y - pad // 2, pad * 2, pad, MARK_FILL, t=4)
    cv.rect(x - pad - 2, y - pad // 2 - 2, pad * 2 + 4, pad + 4, MARK_OUTLINE, t=2)
    cv.numeral(x, y, n, MARK_TEXT, BLACK_OPAQUE, scale=3)


# ---------------------------------------------------------------------------
# Argument parser (safe — no eval)
# ---------------------------------------------------------------------------


def _parse_action_args(line: str) -> tuple[str, list[object]] | None:
    """Extract function name and args from ``name(a, b, ...)``.

    Uses ``ast.literal_eval`` — only parses literal values (ints, strings,
    tuples).  No code execution is possible.
    """
    paren = line.find("(")
    if paren == -1:
        return None
    name = line[:paren].strip()
    try:
        args = list(
            ast.literal_eval(f"({line[paren + 1 : line.rfind(')')]},)")
        )
    except (ValueError, SyntaxError):
        return None
    return name, args


# ---------------------------------------------------------------------------
# Unified annotation pipeline
# ---------------------------------------------------------------------------


def _apply_all_annotations(
    cv: Canvas, actions: list[str], marks: bool,
) -> None:
    """Apply calibration and numbered action annotations to *cv*.

    Execution order:

    1. **Calibration fill** (if enabled) — wipes buffer to solid black.
    2. **Action annotations** — in list order, each numbered sequentially
       starting from 1.

    All mutations happen on the native-resolution buffer before any resize.
    The ``focus`` action has already been extracted and consumed before this
    function is called — it will not appear in *actions*.
    """
    # Step 1: Calibration override
    if _CALIBRATION:
        cv.fill_solid(BLACK_OPAQUE)
        cx, cy = _norm(_CAL_CIRCLE_X, cv.w), _norm(_CAL_CIRCLE_Y, cv.h)
        cv.circle(cx, cy, _CAL_CIRCLE_R, MARK_OUTLINE, filled=False, thickness=6)

    # Step 2: Action annotations
    if not (marks and actions):
        return

    px: int | None = None
    py: int | None = None
    action_num = 1  # sequential numbering for the VLM

    for line in actions:
        parsed = _parse_action_args(line)
        if parsed is None:
            continue
        name, args = parsed

        match name:
            case "left_click" if len(args) >= 2:
                x, y = _norm(int(args[0]), cv.w), _norm(int(args[1]), cv.h)
                _annotate_left_click(cv, x, y, action_num, px, py)
                px, py = x, y
                action_num += 1

            case "right_click" if len(args) >= 2:
                x, y = _norm(int(args[0]), cv.w), _norm(int(args[1]), cv.h)
                _annotate_right_click(cv, x, y, action_num, px, py)
                px, py = x, y
                action_num += 1

            case "double_left_click" if len(args) >= 2:
                x, y = _norm(int(args[0]), cv.w), _norm(int(args[1]), cv.h)
                _annotate_double_click(cv, x, y, action_num, px, py)
                px, py = x, y
                action_num += 1

            case "drag" if len(args) >= 4:
                x1 = _norm(int(args[0]), cv.w)
                y1 = _norm(int(args[1]), cv.h)
                x2 = _norm(int(args[2]), cv.w)
                y2 = _norm(int(args[3]), cv.h)
                _annotate_drag(cv, x1, y1, x2, y2, action_num, px, py)
                px, py = x2, y2
                action_num += 1

            case "type":
                if px is not None and py is not None:
                    _annotate_type(cv, px, py, action_num)
                    action_num += 1


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


def capture(
    actions: list[str], width: int, height: int, marks: bool,
) -> str:
    """Capture screen (or focused region), annotate, resize, return base64 PNG.

    If *actions* contains a ``focus(x1, y1, x2, y2)`` entry, only that
    screen region is captured at native pixel dimensions.  The *width* and
    *height* parameters are overridden by the crop size — no scaling is
    performed.  Pass ``width=0, height=0`` to signal "use crop dimensions".
    """
    # 0. Pre-scan: extract focus region (if any) before pipeline starts
    actions = list(actions)  # defensive copy — _extract_focus mutates
    focus = _extract_focus(actions)

    if _CALIBRATION and _CAL_PERSIST_ACTIONS and focus is None:
        try:
            hist = json.loads(_CAL_ACTIONS_FILE.read_text(encoding="utf-8"))
            if not isinstance(hist, list):
                hist = []
        except Exception:
            hist = []
        # Keep only strings (defensive)
        hist = [x for x in hist if isinstance(x, str)]
        hist.extend(actions)
        actions = hist
        try:
            _CAL_ACTIONS_FILE.write_text(json.dumps(hist), encoding="utf-8")
        except Exception:
            pass


    if focus is not None:
        src_x, src_y, sw, sh = focus
        # Override target to match crop — no resize, model gets native pixels
        width, height = sw, sh
    else:
        src_x, src_y = 0, 0
        sw, sh = _screen_w, _screen_h

    # 1. Capture at native resolution (full screen or focused region)
    bgra = _capture_bgra(sw, sh, src_x, src_y)

    # 2. Convert to RGBA for annotation (forces α=0xFF)
    rgba = _bgra_to_rgba(bgra)

    # 3. All mutations on native-res buffer (calibration + annotations)
    cv = Canvas(rgba, sw, sh)
    _apply_all_annotations(cv, actions, marks)
    rgba = cv.to_bytes()

    # 4. Resize the fully-annotated image (if needed)
    if (sw, sh) != (width, height):
        bgra = _rgba_to_bgra(rgba)  # Forces α=0xFF — no alpha leak into GDI resize
        bgra = _resize_bgra(bgra, sw, sh, width, height)
        rgba = _bgra_to_rgba(bgra)
        sw, sh = width, height

    # 5. Encode & emit
    png = _encode_png(rgba, sw, sh)
    return base64.b64encode(png).decode("ascii")


def main() -> None:
    """Entry point: read JSON from stdin, write base64 PNG to stdout."""
    request = json.loads(sys.stdin.read())
    b64 = capture(
        actions=request.get("actions", []),
        width=request.get("width", 0),
        height=request.get("height", 0),
        marks=request.get("marks", True),
    )
    sys.stdout.write(b64)
    sys.stdout.flush()


if __name__ == "__main__":
    main()

