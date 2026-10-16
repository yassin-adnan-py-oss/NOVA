# NOVA - AI Vision Assistant  |  "Bixby-style" portrait UI
# ----------------------------------------------------------------------
# pip install opencv-python ultralytics pyttsx3 pillow numpy torch torchvision timm ollama
#
# Same engine as nova_ui.py (YOLO tracking, colours, memory, agent, OCR, voice)
# but a completely new look: camera-first phone layout, floating pins,
# teal viewfinder, radar, glass cards, and a round mode bar at the bottom.
#
# Shortcuts: Space start/stop | S describe | V voice | O read text | D depth | F fit/fill
# ----------------------------------------------------------------------

import os
import sys
import math
import time
import queue
import textwrap
import threading
import traceback
import collections
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import cv2
import numpy as np

# ------------------------------------------------------------------
# Crash diagnostics
# ------------------------------------------------------------------
def _write_crash(tag, msg):
    try:
        with open("nova_crash.log", "a", encoding="utf-8") as f:
            f.write(f"\n===== {tag} {time.strftime('%Y-%m-%d %H:%M:%S')} =====\n{msg}\n")
    except Exception:
        pass


def _log_crash(t, v, tb):
    _write_crash("MAIN", "".join(traceback.format_exception(t, v, tb)))
    sys.__excepthook__(t, v, tb)


def _thread_hook(args):
    msg = "".join(traceback.format_exception(args.exc_type, args.exc_value, args.exc_traceback))
    _write_crash("THREAD", msg)
    print(f"[thread error] {msg}", file=sys.stderr)


sys.excepthook = _log_crash
threading.excepthook = _thread_hook

# ------------------------------------------------------------------
# Optional NOVA modules
# ------------------------------------------------------------------
try:
    from ultralytics import YOLO
    YOLO_OK = True
except ImportError:
    YOLO_OK = False
try:
    from nova_color_organizer import ColorOrganizer
    COLOR_OK = True
except ImportError:
    COLOR_OK = False
try:
    from nova_memory import Memory
    MEMORY_OK = True
except ImportError:
    MEMORY_OK = False
try:
    from nova_agent_llm import Agent
    AGENT_OK = True
except ImportError:
    AGENT_OK = False
try:
    from nova_ocr import OCRReader
    OCR_OK = True
except ImportError:
    OCR_OK = False
try:
    import ollama  # noqa: F401
    OLLAMA_PKG_OK = True
except ImportError:
    OLLAMA_PKG_OK = False
try:
    import pyttsx3  # noqa: F401
    TTS_OK = True
except ImportError:
    TTS_OK = False
try:
    from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageTk
    PIL_OK = True
except ImportError:
    PIL_OK = False

# ------------------------------------------------------------------
# Design tokens
# ------------------------------------------------------------------
C = dict(bg="#05060a", sheet="#0d1020", card="#161a2c", card2="#1e2338", border="#2a3047",
         text="#eef1fa", muted="#8a92aa", teal="#2dd4bf", accent="#7c9cff",
         ok="#4ade80", warn="#fbbf24", danger="#ff5470")
F = "Segoe UI"
FONT = (F, 10)
FONT_B = (F, 10, "bold")
FONT_S = (F, 9)

POS_PHRASE = {"left": "on your left", "center": "straight ahead", "right": "on your right"}
POS_SHORT = {"left": "Left", "center": "Ahead", "right": "Right"}
STATUS_COL = {"approaching": C["danger"], "moving away": C["warn"], "stable": C["teal"]}
STATUS_TXT = {"approaching": "APPROACHING", "moving away": "MOVING AWAY", "stable": "STEADY"}

CATS = [
    ({"person"}, "#4f8cff"),
    ({"car", "bus", "truck", "bicycle", "motorcycle", "train", "boat", "airplane"}, "#ff9f43"),
    ({"dog", "cat", "bird", "horse", "cow", "sheep", "bear", "elephant"}, "#f472b6"),
    ({"chair", "couch", "bed", "dining table", "bench", "toilet", "potted plant"}, "#c084fc"),
    ({"tv", "laptop", "cell phone", "keyboard", "mouse", "remote", "clock"}, "#2dd4bf"),
    ({"bottle", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich",
      "orange", "wine glass", "pizza", "cake"}, "#4ade80"),
]


def cat_color(label):
    for names, col in CATS:
        if label in names:
            return col
    return "#94a3b8"


# ==================================================================
# Rendering (Pillow, 2x supersampled).  Runs in the worker thread.
# ==================================================================
SS = 2
_FONT_CACHE = {}


def get_font(px, bold=False):
    key = (px, bold)
    if key in _FONT_CACHE:
        return _FONT_CACHE[key]
    names = (["segoeuib.ttf", "arialbd.ttf", "DejaVuSans-Bold.ttf", "Arial Bold.ttf"] if bold
             else ["segoeui.ttf", "arial.ttf", "DejaVuSans.ttf", "Arial.ttf"])
    font = None
    for n in names:
        try:
            font = ImageFont.truetype(n, px)
            break
        except Exception:
            continue
    if font is None:
        try:
            font = ImageFont.load_default(px)
        except Exception:
            font = ImageFont.load_default()
    _FONT_CACHE[key] = font
    return font


def rgba(c, a=255):
    if isinstance(c, str):
        h = c.lstrip("#")
        return (int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16), a)
    return (c[0], c[1], c[2], a)


class Gfx:
    """RGBA overlay with a logical-pixel API (drawn at 2x, averaged down)."""

    def __init__(self, W, H):
        self.W, self.H = W, H
        self.img = Image.new("RGBA", (W * SS, H * SS), (255, 255, 255, 0))
        self.d = ImageDraw.Draw(self.img)

    def rrect(self, box, r, fill=None, outline=None, width=1, a=255, oa=255):
        x0, y0, x1, y1 = [v * SS for v in box]
        self.d.rounded_rectangle([x0, y0, x1, y1], radius=r * SS,
                                 fill=rgba(fill, a) if fill else None,
                                 outline=rgba(outline, oa) if outline else None,
                                 width=max(int(width * SS), 1))

    def circle(self, cx, cy, r, fill=None, outline=None, width=1, a=255, oa=255):
        self.d.ellipse([(cx - r) * SS, (cy - r) * SS, (cx + r) * SS, (cy + r) * SS],
                       fill=rgba(fill, a) if fill else None,
                       outline=rgba(outline, oa) if outline else None,
                       width=max(int(width * SS), 1))

    def line(self, pts, color, width=2, a=255):
        self.d.line([(x * SS, y * SS) for x, y in pts], fill=rgba(color, a),
                    width=max(int(width * SS), 1), joint="curve")

    def poly(self, pts, fill, a=255):
        self.d.polygon([(x * SS, y * SS) for x, y in pts], fill=rgba(fill, a))

    def arc(self, cx, cy, r, start, end, color, width=2, a=255):
        self.d.arc([(cx - r) * SS, (cy - r) * SS, (cx + r) * SS, (cy + r) * SS],
                   start, end, fill=rgba(color, a), width=max(int(width * SS), 1))

    def tw(self, s, size, bold=False):
        return get_font(int(size * SS), bold).getlength(s) / SS

    def text(self, xy, s, size, bold=False, color="#ffffff", anchor="lm", a=255, shadow=False):
        f = get_font(int(size * SS), bold)
        x, y = xy[0] * SS, xy[1] * SS
        try:
            if shadow:
                self.d.text((x + SS, y + SS), s, font=f, fill=(0, 0, 0, int(a * 0.65)), anchor=anchor)
            self.d.text((x, y), s, font=f, fill=rgba(color, a), anchor=anchor)
        except Exception:
            self.d.text((x, y), s, font=f, fill=rgba(color, a))

    def result(self):
        return self.img.resize((self.W, self.H), Image.BOX)


def glass(base, box, r=16, tint=(8, 10, 20), ta=0.5, circle=False, blur=8):
    """Blurred translucent panel pasted straight into the base image."""
    x0, y0, x1, y1 = [int(v) for v in box]
    x0, y0 = max(x0, 0), max(y0, 0)
    x1, y1 = min(x1, base.width), min(y1, base.height)
    if x1 - x0 < 4 or y1 - y0 < 4:
        return
    region = base.crop((x0, y0, x1, y1)).filter(ImageFilter.GaussianBlur(blur))
    region = Image.blend(region, Image.new("RGB", region.size, tint), ta)
    m = Image.new("L", (region.width * 3, region.height * 3), 0)
    md = ImageDraw.Draw(m)
    if circle:
        md.ellipse([0, 0, m.width - 1, m.height - 1], fill=255)
    else:
        md.rounded_rectangle([0, 0, m.width - 1, m.height - 1], radius=r * 3, fill=255)
    base.paste(region, (x0, y0), m.resize(region.size, Image.LANCZOS))


def cover(rgb, W, H):
    h, w = rgb.shape[:2]
    s = max(W / w, H / h)
    nw, nh = max(int(math.ceil(w * s)), W), max(int(math.ceil(h * s)), H)
    big = cv2.resize(rgb, (nw, nh), interpolation=cv2.INTER_LINEAR)
    x0, y0 = (nw - W) // 2, (nh - H) // 2
    return big[y0:y0 + H, x0:x0 + W]


class Renderer:
    TEAL = C["teal"]

    def __init__(self):
        self._masks = {}
        self._bg = {}

    # ---------------- helpers ----------------
    def _round_mask(self, w, h, r):
        key = (w, h, r)
        if key not in self._masks:
            m = Image.new("L", (w * 3, h * 3), 0)
            ImageDraw.Draw(m).rounded_rectangle([0, 0, w * 3 - 1, h * 3 - 1], radius=r * 3, fill=255)
            self._masks[key] = m.resize((w, h), Image.LANCZOS)
        return self._masks[key]

    @staticmethod
    def _pin(g, x, y, color, letter, pulse=None):
        cy = y - 32
        if pulse is not None:                       # approaching -> red pulse ring
            g.circle(x, cy, 21 + 6 * pulse, outline=C["danger"], width=3, oa=int(200 - 120 * pulse))
        g.poly([(x - 9, cy + 13), (x + 9, cy + 13), (x, y - 5)], color)
        g.circle(x, cy, 19, fill=color, outline="#ffffff", width=2)
        g.text((x, cy + 1), letter, 18, True, "#ffffff", anchor="mm")

    @staticmethod
    def _icon_voice(g, cx, cy, mode):
        g.poly([(cx - 10, cy - 4), (cx - 5, cy - 4), (cx + 2, cy - 10), (cx + 2, cy + 10),
                (cx - 5, cy + 4), (cx - 10, cy + 4)], "#ffffff")
        if mode in ("alerts", "all"):
            g.arc(cx + 2, cy, 7, -45, 45, "#ffffff", 2)
        if mode == "all":
            g.arc(cx + 2, cy, 13, -45, 45, "#ffffff", 2)
        if mode == "off":
            g.line([(cx - 11, cy + 11), (cx + 11, cy - 11)], C["danger"], 3)

    def _top_bar(self, g, ctx, hits, running):
        W, cy = ctx["W"], 40

        def back(cx):
            g.circle(cx, cy, 21, fill=(0, 0, 0), a=120)
        back(34)
        g.line([(39, cy - 10), (29, cy), (39, cy + 10)], "#ffffff", 3)
        hits.append((8, cy - 26, 62, cy + 26, "back"))

        back(86)
        self._icon_voice(g, 86, cy, ctx["voice"])
        g.text((86, cy + 31), ctx["voice"].title(), 10, True, "#ffffff", "mm", shadow=True)
        hits.append((62, cy - 26, 112, cy + 36, "voice"))

        if running:
            g.circle(W / 2 - 40, cy, 4, fill=C["danger"])
        g.text((W / 2 + (4 if running else 0), cy), "NOVA", 19, True, "#ffffff", "mm", shadow=True)

        fx = W - 96
        back(fx)
        if ctx["fill"]:
            g.rrect((fx - 8, cy - 8, fx + 8, cy + 8), 3, fill="#ffffff")
        else:
            for sx, sy in ((-1, -1), (1, -1), (-1, 1), (1, 1)):
                px, py = fx + sx * 10, cy + sy * 10
                g.line([(px - sx * 6, py), (px, py), (px, py - sy * 6)], "#ffffff", 2.5)
        hits.append((fx - 24, cy - 26, fx + 24, cy + 26, "fit"))

        dx = W - 44
        if ctx["depth"]:
            g.circle(dx, cy, 21, fill=self.TEAL)
            g.text((dx, cy + 1), "3D", 12, True, "#06201d", "mm")
        else:
            back(dx)
            g.text((dx, cy + 1), "3D", 12, True, "#ffffff", "mm")
        hits.append((dx - 24, cy - 26, dx + 24, cy + 26, "depth"))

    def _viewfinder(self, g, vf, danger, ctx):
        x0, y0, x1, y1 = vf
        t = ctx["t"]
        col = C["danger"] if danger else self.TEAL
        pulse = 0.5 + 0.5 * math.sin(t * 8) if danger else 0
        g.rrect((x0, y0, x1, y1), 18, outline=col, width=2 + 2 * pulse, oa=int(170 + 70 * pulse))
        L = 30
        for sx, px in ((1, x0), (-1, x1)):
            for sy, py in ((1, y0), (-1, y1)):
                g.line([(px + sx * L, py), (px, py), (px, py + sy * L)], col, 4)
        if not danger:                               # scan line
            yy = y0 + ((t * 0.35) % 1) * (y1 - y0)
            for k in range(4):
                g.line([(x0 + 10, yy - k * 2), (x1 - 10, yy - k * 2)], self.TEAL, 2, a=70 - 15 * k)

    def _chips(self, g, vf, ctx, hits):
        x = vf[0] + 14
        y = vf[1] + 14
        for key, label in (("all", "All"), ("moving", "Moving"), ("close", "Close")):
            w = g.tw(label, 13, True) + 28
            active = ctx["filter"] == key
            if active:
                g.rrect((x, y, x + w, y + 30), 15, fill=self.TEAL, a=235)
                g.text((x + w / 2, y + 15), label, 13, True, "#06201d", "mm")
            else:
                g.rrect((x, y, x + w, y + 30), 15, fill=(0, 0, 0), a=130, outline="#ffffff", oa=70)
                g.text((x + w / 2, y + 15), label, 13, True, "#ffffff", "mm")
            hits.append((x, y, x + w, y + 30, f"filter:{key}"))
            x += w + 8

    def _radar(self, g, base, vf, dets):
        r = 34
        cx, cy = vf[2] - 52, vf[3] - 52
        glass(base, (cx - r, cy - r, cx + r, cy + r), circle=True, tint=(6, 12, 20), ta=0.55)
        g.circle(cx, cy, r, outline="#ffffff", width=1.5, oa=110)
        g.circle(cx, cy, r * 0.62, outline="#ffffff", width=1, oa=50)
        g.circle(cx, cy, r * 0.28, outline="#ffffff", width=1, oa=50)
        for ang in (-19, 19):                        # thirds boundaries
            a = math.radians(ang)
            g.line([(cx, cy), (cx + r * math.sin(a), cy - r * math.cos(a))], "#ffffff", 1, a=60)
        g.poly([(cx, cy - 6), (cx - 4, cy + 3), (cx + 4, cy + 3)], "#ffffff")
        spread = {"left": -38, "center": 0, "right": 38}
        rad = {"Very close": 0.33, "Close": 0.6, "Far": 0.86}
        for d in dets[:12]:
            a = math.radians(spread[d["direction"]])
            rr = r * rad[d["prox"]]
            g.circle(cx + rr * math.sin(a), cy - rr * math.cos(a), 3.6, fill=STATUS_COL[d["status"]])

    def _danger_card(self, g, base, x0, y0, bw, d, t):
        pulse = 0.5 + 0.5 * math.sin(t * 8)
        glass(base, (x0, y0, x0 + bw, y0 + 76), r=18, tint=(70, 8, 28), ta=0.62)
        g.rrect((x0, y0, x0 + bw, y0 + 76), 18, outline=C["danger"], width=2, oa=int(160 + 90 * pulse))
        tx, ty = x0 + 38, y0 + 38
        g.poly([(tx, ty - 20), (tx + 22, ty + 16), (tx - 22, ty + 16)], C["danger"])
        g.text((tx, ty + 6), "!", 20, True, "#ffffff", "mm")
        name = f"{d['color']} {d['label']}".strip().title()
        g.text((x0 + 74, y0 + 25), "WARNING", 11, True, C["danger"])
        g.text((x0 + 74, y0 + 46), f"{name} {d['status']}", 17, True, "#ffffff")
        g.text((x0 + 74, y0 + 64), POS_PHRASE[d["direction"]].capitalize(), 12, False, "#ffd0d8")
        return 76

    @staticmethod
    def _toast_lines(text, bw):
        chars = max(int((bw - 36) / 7.6), 12)
        wrapped = textwrap.wrap(text, chars)
        lines = wrapped[:5]
        if len(wrapped) > 5:
            lines[-1] = lines[-1][:-1] + "\u2026"
        return lines

    def _toast(self, g, base, x0, y0, bw, text):
        lines = self._toast_lines(text, bw)
        bh = 36 + 20 * len(lines)
        glass(base, (x0, y0, x0 + bw, y0 + bh), r=18, tint=(6, 14, 24), ta=0.6)
        g.rrect((x0, y0, x0 + bw, y0 + bh), 18, outline=self.TEAL, width=1.5, oa=150)
        g.text((x0 + 18, y0 + 18), "SCENE", 10, True, self.TEAL)
        for i, ln in enumerate(lines):
            g.text((x0 + 18, y0 + 40 + i * 20), ln, 14, False, "#ffffff")
        return bh

    def _list(self, g, base, W, top, bottom, dets, ctx):
        y = top
        if not dets:
            g.text((W / 2, (top + bottom) / 2 - 10),
                   "Nothing here" if ctx["filter"] != "all" else "Scanning your surroundings\u2026",
                   15, True, "#ffffff", "mm", a=190)
            return
        rowh, gap = 60, 8
        for d in dets:
            if y + rowh > bottom:
                break
            glass(base, (14, y, W - 14, y + rowh), r=18, tint=(10, 14, 28), ta=0.5)
            g.rrect((14, y, W - 14, y + rowh), 18, outline="#ffffff", oa=28)
            col = cat_color(d["label"])
            g.circle(44, y + rowh / 2, 18, fill=col, outline="#ffffff", width=1.5)
            g.text((44, y + rowh / 2 + 1), d["label"][:1].upper(), 17, True, "#ffffff", "mm")
            name = f"{d['color']} {d['label']}".strip().title()
            g.text((74, y + 22), name, 15, True, "#ffffff")
            g.text((74, y + 42), f"{d['prox']}  \u00B7  {POS_PHRASE[d['direction']]}", 12, False, "#aab2c8")
            st = STATUS_TXT[d["status"]]
            sw = g.tw(st, 10.5, True) + 18
            g.rrect((W - 28 - sw, y + 18, W - 28, y + 42), 12, fill=STATUS_COL[d["status"]], a=60,
                    outline=STATUS_COL[d["status"]], oa=160)
            g.text((W - 28 - sw / 2, y + 30), st, 10.5, True, STATUS_COL[d["status"]], "mm")
            y += rowh + gap

    # ---------------- live frame ----------------
    def compose_live(self, frame_bgr, dets, danger, ctx):
        W, H = ctx["W"], ctx["H"]
        h, w = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        hits = []

        if ctx["fill"]:
            base = Image.fromarray(np.ascontiguousarray(cover(rgb, W, H)))
            vf = (14, 90, W - 14, H - 14)
            s = max(W / w, H / h)
            ox, oy = (W - w * s) / 2, (H - h * s) / 2
            # soft vignette so icons stay readable
            top = Image.new("RGB", (W, 110), (0, 0, 0))
            fade = np.tile(np.linspace(150, 0, 110, dtype=np.uint8)[:, None], (1, W))
            base.paste(top, (0, 0), Image.fromarray(fade))
        else:
            small = cover(rgb, max(W // 4, 8), max(H // 4, 8))
            small = cv2.GaussianBlur(small, (0, 0), 4)
            bg = cv2.resize(small, (W, H), interpolation=cv2.INTER_LINEAR)
            base = Image.fromarray(cv2.convertScaleAbs(bg, alpha=0.45))
            vw = W - 24
            vh = int(vw * h / w)
            if vh > H * 0.55:
                vh = int(H * 0.55)
                vw = int(vh * w / h)
            vx, vy = (W - vw) // 2, 90
            vid = Image.fromarray(cv2.resize(rgb, (vw, vh), interpolation=cv2.INTER_LINEAR))
            base.paste(vid, (vx, vy), self._round_mask(vw, vh, 18))
            vf = (vx, vy, vx + vw, vy + vh)
            s, ox, oy = vw / w, vx, vy

        def mp(x, y):
            return ox + x * s, oy + y * s

        # filter
        f = ctx["filter"]
        vis = [d for d in dets if f == "all"
               or (f == "moving" and d["status"] != "stable")
               or (f == "close" and d["prox"] in ("Close", "Very close"))]

        g = Gfx(W, H)
        self._top_bar(g, ctx, hits, True)
        self._viewfinder(g, vf, bool(danger), ctx)
        self._chips(g, vf, ctx, hits)

        # pins
        placed = []
        t = ctx["t"]
        top_reserved = 0
        if ctx["fill"]:
            if danger:
                top_reserved += 88
            if ctx.get("toast"):
                top_reserved += 36 + 20 * len(self._toast_lines(ctx["toast"], vf[2] - vf[0] - 36)) + 12
        for d in sorted(vis, key=lambda d: -d["area"])[:6]:
            x1, y1, x2, y2 = d["box"]
            ax, ay = mp((x1 + x2) / 2, y1)
            bx1, by1 = mp(x1, y1)
            bx2, by2 = mp(x2, y2)
            if not (vf[0] < ax < vf[2] and vf[1] < ay + 40 < vf[3] + 40):
                continue
            col = cat_color(d["label"])
            if d["status"] == "approaching":
                g.rrect((bx1, by1, bx2, by2), 10, outline=C["danger"], width=2, oa=210)
            elif d["status"] == "moving away":
                g.rrect((bx1, by1, bx2, by2), 10, outline=C["warn"], width=1.5, oa=170)
            else:
                g.rrect((bx1, by1, bx2, by2), 10, outline="#ffffff", width=1, oa=70)
            px = min(max(ax, vf[0] + 50), vf[2] - 50)
            py = min(max(ay + 8, vf[1] + 92 + top_reserved), vf[3] - 62)
            for qx, qy in placed:
                if abs(qx - px) < 70 and abs(qy - py) < 70:
                    py = qy + 74
            py = min(py, vf[3] - 62)
            placed.append((px, py))
            pulse = 0.5 + 0.5 * math.sin(t * 7) if d["status"] == "approaching" else None
            self._pin(g, px, py, col, d["label"][:1].upper(), pulse)
            name = f"{d['color']} {d['label']}".strip().title()
            g.text((px, py + 14), name, 15, True, "#ffffff", "mm", shadow=True)
            g.text((px, py + 31), f"{d['prox']} \u00B7 {POS_SHORT[d['direction']]}", 12, False,
                   "#e8ecf8", "mm", shadow=True)

        self._radar(g, base, vf, dets)
        if ctx["fill"]:                      # cards float at the top of the camera
            cx0, cw, cy0 = vf[0] + 18, vf[2] - vf[0] - 36, vf[1] + 58
        else:                                # cards sit in the list area under the video
            cx0, cw, cy0 = 14, W - 28, vf[3] + 14
        if danger:
            cy0 += self._danger_card(g, base, cx0, cy0, cw, danger[0], t) + 12
        if ctx.get("toast"):
            cy0 += self._toast(g, base, cx0, cy0, cw, ctx["toast"]) + 12
        if not ctx["fill"]:
            self._list(g, base, W, cy0 if (danger or ctx.get("toast")) else vf[3] + 14, H - 6, vis, ctx)

        g.text((W - 18, H - 12), f"{ctx['fps']:.0f} FPS", 10, False, "#ffffff", "rm", a=110)

        out = base.convert("RGBA")
        out.alpha_composite(g.result())
        return out.convert("RGB"), hits

    # ---------------- idle / welcome ----------------
    def compose_idle(self, W, H, t, message=None):
        key = (W, H)
        if key not in self._bg:
            yy = np.linspace(0, 1, H)[:, None, None]
            top = np.array([14, 18, 46], dtype=np.float32)
            bot = np.array([4, 5, 10], dtype=np.float32)
            grad = top * (1 - yy) + bot * yy
            gx = np.linspace(-1, 1, W)[None, :]
            gy = np.linspace(-1, 1, H)[:, None]
            glow = np.exp(-((gx / 0.8) ** 2 + ((gy + 0.45) / 0.35) ** 2))[:, :, None]
            grad = grad + glow * np.array([10, 70, 66], dtype=np.float32)
            self._bg = {key: Image.fromarray(np.clip(grad, 0, 255).astype(np.uint8))}
        base = self._bg[key].copy()
        g = Gfx(W, H)
        hits = []
        cx, cy = W / 2, H * 0.30
        for i, (rr, a) in enumerate(((52, 150), (74, 95), (98, 55))):
            wob = 5 * math.sin(t * 2 - i * 0.9)
            g.circle(cx, cy, rr + wob, outline=self.TEAL, width=2, oa=a)
        g.circle(cx, cy, 36, fill=self.TEAL, a=40)
        g.circle(cx, cy, 27, outline="#ffffff", width=3.5)
        g.circle(cx, cy, 10, fill=self.TEAL)
        g.text((cx, cy + 142), "NOVA", 46, True, "#ffffff", "mm")
        g.text((cx, cy + 184), "A I   V I S I O N   A S S I S T A N T", 11, True, self.TEAL, "mm")

        by = H * 0.62
        if message:
            dots = "." * (int(t * 2.5) % 4)
            g.text((cx, by + 20), message + dots, 16, True, "#ffffff", "mm", a=220)
        else:
            bw = min(W - 80, 260)
            g.rrect((cx - bw / 2, by, cx + bw / 2, by + 56), 28, fill="#ffffff")
            g.text((cx, by + 28), "Start Camera", 17, True, "#06201d", "mm")
            hits.append((cx - bw / 2, by, cx + bw / 2, by + 56, "start"))
            g.rrect((cx - bw / 2, by + 70, cx + bw / 2, by + 122), 26, fill=(255, 255, 255), a=22,
                    outline="#ffffff", oa=80)
            g.text((cx, by + 96), "Open video file", 15, True, "#ffffff", "mm")
            hits.append((cx - bw / 2, by + 70, cx + bw / 2, by + 122, "open"))
        g.text((cx, H - 24), "Space start/stop  \u00B7  S describe  \u00B7  V voice  \u00B7  F fit/fill",
               11, False, "#ffffff", "mm", a=110)
        out = base.convert("RGBA")
        out.alpha_composite(g.result())
        return out.convert("RGB"), hits


# ==================================================================
# Small Tk widgets for the sheets
# ==================================================================
class Btn(tk.Label):
    KINDS = {"primary": (C["teal"], "#7be8db", "#06201d"),
             "ghost": (C["card2"], C["border"], C["text"]),
             "danger": ("#3a1822", "#55202f", C["danger"])}

    def __init__(self, master, text, command, kind="ghost", font=FONT_B, padx=14, pady=7):
        self._bg, self._hover, fg = self.KINDS[kind]
        super().__init__(master, text=text, bg=self._bg, fg=fg, font=font,
                         padx=padx, pady=pady, cursor="hand2")
        self._cmd = command
        self.bind("<Enter>", lambda e: self.configure(bg=self._hover))
        self.bind("<Leave>", lambda e: self.configure(bg=self._bg))
        self.bind("<Button-1>", lambda e: self._cmd())


def make_entry(master):
    return tk.Entry(master, bg=C["card2"], fg=C["text"], font=FONT, insertbackground=C["text"],
                    relief="flat", highlightthickness=1, highlightbackground=C["border"],
                    highlightcolor=C["teal"])


# ==================================================================
# Speech: one worker thread, one queue
# ==================================================================
class Speaker:
    def __init__(self, log):
        self.q = queue.Queue()
        self.log = log
        self.busy = False
        self.mode = "alerts"          # off | alerts | all
        threading.Thread(target=self._run, daemon=True).start()

    @property
    def enabled(self):
        return self.mode != "off" and TTS_OK

    def idle(self):
        return (not self.busy) and self.q.empty()

    def say(self, text, urgent=False, force=False):
        if not TTS_OK or (self.mode == "off" and not force):
            return
        if urgent:
            try:
                while True:
                    self.q.get_nowait()
            except queue.Empty:
                pass
        self.q.put(text)

    def _run(self):
        while True:
            text = self.q.get()
            if text is None:
                break
            try:
                self.busy = True
                import pyttsx3 as tts
                eng = tts.init()          # created INSIDE this thread (Windows-safe)
                eng.setProperty("rate", 180)
                eng.say(text)
                eng.runAndWait()
                eng.stop()
            except Exception as e:
                self.log(f"TTS error: {e}")
            finally:
                self.busy = False


# ==================================================================
# Application
# ==================================================================
MODES = [("memory", "\u2315", "Memory"), ("text", "T", "Text"), ("objects", "\u25C9", "Objects"),
         ("scene", "\u25A3", "Scene"), ("chat", "\u2726", "Assistant")]
VOICE_MODES = ["alerts", "all", "off"]


class NovaApp:
    def __init__(self, root):
        self.root = root
        root.title("NOVA")
        root.configure(bg=C["bg"])
        sh, sw = root.winfo_screenheight(), root.winfo_screenwidth()
        H = min(int(sh * 0.88), 940)
        Wd = max(int(H * 0.5), 400)
        root.geometry(f"{Wd}x{H}+{(sw - Wd) // 2}+{max((sh - H) // 2 - 20, 0)}")
        root.minsize(360, 640)

        self.ui_q = queue.Queue()
        self._lock = threading.Lock()
        self._frame_out = None          # (PIL image, hits)
        self._frame_new = False
        self.hits = []
        self.view_size = (Wd, H - 100)
        self.renderer = Renderer()

        self.stop_evt = threading.Event()
        self.stop_evt.set()
        self.thread = None
        self.latest_frame = None
        self.current_objects = []
        self.fps = 0.0
        self.fill = False
        self.filter = "all"
        self.mode = "objects"
        self.toast = None
        self.log_lines = collections.deque(maxlen=300)

        self.model = None
        self.depth_model = self.depth_transform = self._depth_device = None
        self.depth_on = False
        self.ocr_reader = None
        self.color_organizer = ColorOrganizer() if COLOR_OK else None
        self.memory = Memory() if MEMORY_OK else None
        self.speaker = Speaker(self.log)
        self.agent = None
        if AGENT_OK and self.memory and OLLAMA_PKG_OK:
            try:
                self.agent = Agent(self.memory, self.get_current_objects)
            except Exception as e:
                self.log(f"Agent init failed: {e}")

        self.history = {}
        self.color_cache = {}
        self.mem_last = {}
        self.last_announced = {}
        self.last_speech = 0.0
        self._last_idle = 0.0
        self._last_mem_refresh = 0.0

        if not PIL_OK:
            messagebox.showerror("NOVA", "Pillow is required:  pip install pillow")
        self._build_style()
        self._build_ui()
        self._bind_keys()
        root.protocol("WM_DELETE_WINDOW", self.on_close)
        self._tick()

    # ---------------- thread-safe helpers ----------------
    def ui(self, fn):
        self.ui_q.put(fn)

    def log(self, msg):
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        self.log_lines.append(line)
        print(line)

    def get_current_objects(self):
        return list(self.current_objects)

    # ==================================================================
    # UI build
    # ==================================================================
    def _build_style(self):
        s = ttk.Style(self.root)
        s.theme_use("clam")
        s.configure("Nova.Treeview", background=C["card"], fieldbackground=C["card"],
                    foreground=C["text"], rowheight=28, borderwidth=0, font=FONT)
        s.configure("Nova.Treeview.Heading", background=C["card2"], foreground=C["muted"],
                    font=FONT_B, borderwidth=0, relief="flat", padding=5)
        s.map("Nova.Treeview", background=[("selected", C["card2"])],
              foreground=[("selected", C["text"])])
        s.configure("Vertical.TScrollbar", background=C["card2"], troughcolor=C["card"],
                    bordercolor=C["card"], arrowcolor=C["muted"], relief="flat")

    def _build_ui(self):
        # mode bar (bottom)
        self.modebar = tk.Canvas(self.root, height=100, bg=C["bg"], highlightthickness=0)
        self.modebar.pack(side="bottom", fill="x")
        self.modebar.bind("<Configure>", lambda e: self._draw_modebar())
        self.modebar.bind("<Button-1>", self._modebar_click)

        # camera area
        self.stage = tk.Frame(self.root, bg="#000000")
        self.stage.pack(side="top", fill="both", expand=True)
        self.canvas = tk.Canvas(self.stage, bg="#000000", highlightthickness=0, cursor="hand2")
        self.canvas.pack(fill="both", expand=True)
        self.img_item = self.canvas.create_image(0, 0, anchor="nw")
        self.canvas.bind("<Configure>", self._on_canvas_resize)
        self.canvas.bind("<Button-1>", self._on_click)

        # bottom sheets
        self.sheets = {}
        self._build_memory_sheet()
        self._build_chat_sheet()
        self._build_text_sheet()

    def _new_sheet(self, key, title):
        sheet = tk.Frame(self.stage, bg=C["sheet"])
        tk.Frame(sheet, bg=C["teal"], height=2).pack(fill="x")
        head = tk.Frame(sheet, bg=C["sheet"])
        head.pack(fill="x", padx=14, pady=(10, 4))
        tk.Label(head, text=title, bg=C["sheet"], fg=C["text"], font=(F, 13, "bold")).pack(side="left")
        close = tk.Label(head, text="\u2715", bg=C["sheet"], fg=C["muted"], font=(F, 14), cursor="hand2")
        close.pack(side="right")
        close.bind("<Button-1>", lambda e: self.set_mode("objects"))
        self.sheets[key] = sheet
        return sheet

    def _build_memory_sheet(self):
        p = self._new_sheet("memory", "Where did I leave\u2026?")
        row = tk.Frame(p, bg=C["sheet"])
        row.pack(fill="x", padx=14)
        self.mem_entry = make_entry(row)
        self.mem_entry.pack(side="left", fill="x", expand=True, ipady=6)
        self.mem_entry.bind("<Return>", lambda e: self.search_memory())
        Btn(row, "Ask", self.search_memory, "primary").pack(side="left", padx=(8, 0))
        chips = tk.Frame(p, bg=C["sheet"])
        chips.pack(fill="x", padx=14, pady=8)
        for q in ("phone", "bottle", "cup", "laptop"):
            Btn(chips, q, lambda q=q: self._quick_memory(q), font=FONT_S, padx=10, pady=3).pack(
                side="left", padx=(0, 6))
        self.mem_result = tk.StringVar(value='Ask something like: "where is my phone?"')
        tk.Label(p, textvariable=self.mem_result, bg=C["card"], fg=C["warn"], font=FONT,
                 wraplength=360, justify="left", anchor="w", padx=12, pady=10).pack(fill="x", padx=14)
        head = tk.Frame(p, bg=C["sheet"])
        head.pack(fill="x", padx=14, pady=(10, 4))
        tk.Label(head, text="Saved records", bg=C["sheet"], fg=C["muted"], font=FONT_S).pack(side="left")
        Btn(head, "Clear", self.clear_memory, "danger", font=FONT_S, padx=9, pady=2).pack(side="right")
        wrap = tk.Frame(p, bg=C["sheet"])
        wrap.pack(fill="both", expand=True, padx=14, pady=(0, 12))
        cols = ("Object", "Color", "Where", "When")
        self.memory_tree = ttk.Treeview(wrap, columns=cols, show="headings",
                                        style="Nova.Treeview", selectmode="none", height=4)
        for c, w in zip(cols, (90, 60, 70, 110)):
            self.memory_tree.heading(c, text=c)
            self.memory_tree.column(c, width=w, anchor="center")
        sb = ttk.Scrollbar(wrap, orient="vertical", command=self.memory_tree.yview)
        self.memory_tree.configure(yscrollcommand=sb.set)
        self.memory_tree.pack(side="left", fill="both", expand=True)
        sb.pack(side="right", fill="y")

    def _build_chat_sheet(self):
        p = self._new_sheet("chat", "Assistant")
        self.agent_lbl = tk.Label(p, bg=C["sheet"], font=(F, 9, "bold"), anchor="w")
        self.agent_lbl.pack(fill="x", padx=14)
        self._update_agent_status()

        # input row + quick chips are packed FIRST (side=bottom) so they can never be clipped
        row = tk.Frame(p, bg=C["sheet"])
        row.pack(side="bottom", fill="x", padx=14, pady=(0, 12))
        self.chat_entry = make_entry(row)
        self.chat_entry.pack(side="left", fill="x", expand=True, ipady=8)
        self.chat_entry.bind("<Return>", lambda e: self.send_chat())
        Btn(row, "Send", self.send_chat, "primary").pack(side="left", padx=(8, 0))

        chips = tk.Frame(p, bg=C["sheet"])
        chips.pack(side="bottom", fill="x", padx=14, pady=8)
        for q in ("What's around me?", "Where is my phone?", "Is anything close?"):
            Btn(chips, q, lambda q=q: self.send_chat(q), font=FONT_S, padx=8, pady=3).pack(
                side="left", padx=(0, 5))

        wrap = tk.Frame(p, bg=C["card"])
        wrap.pack(side="top", fill="both", expand=True, padx=14, pady=(6, 0))
        self.chat_box = tk.Text(wrap, bg=C["card"], fg=C["text"], font=FONT, wrap="word", height=3,
                                relief="flat", state="disabled", padx=10, pady=8,
                                highlightthickness=0, spacing3=6)
        sb = ttk.Scrollbar(wrap, orient="vertical", command=self.chat_box.yview)
        self.chat_box.configure(yscrollcommand=sb.set)
        sb.pack(side="right", fill="y")
        self.chat_box.pack(side="left", fill="both", expand=True)
        self.chat_box.tag_configure("you", foreground=C["teal"], font=FONT_B)
        self.chat_box.tag_configure("nova", foreground=C["accent"], font=FONT_B)
        self._chat_append("NOVA", "Hi! Ask me what's around you or where you left something.")

    def _build_text_sheet(self):
        p = self._new_sheet("text", "Read text")
        row = tk.Frame(p, bg=C["sheet"])
        row.pack(fill="x", padx=14)
        Btn(row, "\U0001F4F7  Capture & Read", self.run_ocr, "primary").pack(side="left")
        Btn(row, "\U0001F50A  Read Aloud", self.ocr_speak).pack(side="left", padx=8)
        self.ocr_text = tk.Text(p, bg=C["card"], fg=C["text"], font=(F, 11), wrap="word",
                                relief="flat", padx=12, pady=10, highlightthickness=0, height=4)
        self.ocr_text.pack(fill="both", expand=True, padx=14, pady=12)

    def _update_agent_status(self):
        if self.agent:
            try:
                from nova_agent_llm import MODEL_NAME
            except Exception:
                MODEL_NAME = "?"
            msg, col = f"\u25CF Agent ONLINE ({MODEL_NAME})", C["ok"]
        else:
            msg, col = "\u25CF Agent offline - built-in answers active", C["warn"]
        self.agent_lbl.configure(text=msg, fg=col)

    # ---------------- mode bar ----------------
    def _draw_modebar(self):
        c = self.modebar
        c.delete("all")
        w = c.winfo_width() or 400
        n = len(MODES)
        for i, (key, icon, label) in enumerate(MODES):
            cx, cy = (i + 0.5) * w / n, 38
            active = (key == self.mode)
            if active:
                c.create_oval(cx - 29, cy - 29, cx + 29, cy + 29, fill="#ffffff", outline="")
                c.create_text(cx, cy - 1, text=icon, fill="#06201d", font=("Segoe UI Symbol", 20, "bold"))
                c.create_text(cx, cy + 46, text=label, fill="#ffffff", font=(F, 10, "bold"))
            else:
                c.create_oval(cx - 22, cy - 22, cx + 22, cy + 22, outline="#4a5070", width=2)
                c.create_text(cx, cy - 1, text=icon, fill="#c9cfe4", font=("Segoe UI Symbol", 15))

    def _modebar_click(self, e):
        w = self.modebar.winfo_width() or 400
        idx = min(int(e.x / (w / len(MODES))), len(MODES) - 1)
        self.set_mode(MODES[idx][0])

    def set_mode(self, key):
        if key == "scene":
            self.describe_scene_action()
            return
        if key == self.mode and key != "objects":
            key = "objects"
        self.mode = key
        for k, sheet in self.sheets.items():
            sheet.place_forget()
        if key in self.sheets:
            self.sheets[key].place(relx=0, rely=1.0, relwidth=1, relheight=0.62, anchor="sw")
            self.sheets[key].lift()
            if key == "memory":
                self.refresh_memory_list()
                self.mem_entry.focus_set()
            elif key == "chat":
                self.chat_entry.focus_set()
        self._draw_modebar()

    # ---------------- canvas events ----------------
    def _on_canvas_resize(self, e):
        self.view_size = (max(e.width, 240), max(e.height, 360))

    def _on_click(self, e):
        for x0, y0, x1, y1, act in reversed(self.hits):
            if x0 <= e.x <= x1 and y0 <= e.y <= y1:
                self._do(act)
                return

    def _do(self, act):
        if act == "start":
            self.start_camera()
        elif act == "open":
            self.open_video()
        elif act == "back":
            self.stop_stream()
        elif act == "voice":
            self.cycle_voice()
        elif act == "fit":
            self.fill = not self.fill
        elif act == "depth":
            self.toggle_depth()
        elif act.startswith("filter:"):
            self.filter = act.split(":")[1]

    def _bind_keys(self):
        def handler(e):
            if isinstance(self.root.focus_get(), (tk.Entry, tk.Text)):
                return
            k = e.keysym.lower()
            if k == "space":
                self.stop_stream() if not self.stop_evt.is_set() else self.start_camera()
            elif k == "s":
                self.describe_scene_action()
            elif k == "v":
                self.cycle_voice()
            elif k == "o":
                self.set_mode("text")
                self.run_ocr()
            elif k == "d":
                self.toggle_depth()
            elif k == "f":
                self.fill = not self.fill
        self.root.bind("<Key>", handler)

    # ==================================================================
    # Voice / depth
    # ==================================================================
    def cycle_voice(self):
        i = VOICE_MODES.index(self.speaker.mode)
        self.speaker.mode = VOICE_MODES[(i + 1) % 3]
        self.log(f"Voice mode: {self.speaker.mode}")

    def toggle_depth(self):
        self.depth_on = not self.depth_on
        if self.depth_on and self.depth_model is None:
            threading.Thread(target=self._load_depth, daemon=True).start()

    def _load_depth(self):
        try:
            import torch
            self.log("Loading MiDaS...")
            m = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")
            self._depth_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            m.to(self._depth_device).eval()
            self.depth_transform = torch.hub.load("intel-isl/MiDaS", "transforms").small_transform
            self.depth_model = m
            self.log("MiDaS ready.")
        except Exception as e:
            self.log(f"Failed to load MiDaS: {e}")
            self.depth_on = False

    # ==================================================================
    # Start / stop
    # ==================================================================
    def _load_yolo(self):
        if not YOLO_OK:
            self.log("Ultralytics not installed: pip install ultralytics")
            return False
        if self.model is None:
            try:
                self.log("Loading YOLO...")
                self.model = YOLO("yolov8s.pt")
                self.log("YOLO ready.")
            except Exception as e:
                self.log(f"YOLO failed to load: {e}")
                return False
        return True

    def start_camera(self):
        self._start_stream(0)

    def open_video(self):
        path = filedialog.askopenfilename(filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv")])
        if path:
            self._start_stream(path)

    def _open_capture(self, source):
        if source == 0:
            backend = cv2.CAP_DSHOW if sys.platform.startswith("win") else cv2.CAP_ANY
            cap = cv2.VideoCapture(0, backend)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
            return cap
        return cv2.VideoCapture(source)

    def _start_stream(self, source):
        self.stop_stream()
        cap = self._open_capture(source)
        if not cap.isOpened():
            messagebox.showerror("NOVA", "Could not open the video source.")
            return
        self.history.clear()
        self.color_cache.clear()
        self.last_announced.clear()
        with self._lock:
            self._frame_out, self._frame_new = None, False
        self.stop_evt = threading.Event()
        self.thread = threading.Thread(target=self._video_loop, args=(cap, source, self.stop_evt),
                                       daemon=True)
        self.thread.start()
        self.loading = True
        self.log(f"Stream started: {source}")

    def stop_stream(self):
        self.stop_evt.set()
        t = self.thread
        if t is not None and t.is_alive() and t is not threading.current_thread():
            t.join(timeout=2.0)
        self.thread = None
        self.loading = False
        with self._lock:
            self._frame_out, self._frame_new = None, False
        self.current_objects = []
        self.toast = None

    def on_close(self):
        self.stop_stream()
        self.speaker.q.put(None)
        self.root.destroy()

    # ==================================================================
    # Analysis (same logic as nova_ui.py)
    # ==================================================================
    def _status_for(self, tid, area):
        h = self.history.setdefault(tid, collections.deque(maxlen=8))
        h.append(area)
        if len(h) < 4:
            return "stable"
        v = list(h)
        old, new = sum(v[:2]) / 2, sum(v[-2:]) / 2
        ratio = new / old if old else 1.0
        if ratio > 1.10:
            return "approaching"
        if ratio < 0.90:
            return "moving away"
        return "stable"

    @staticmethod
    def _proximity(frac):
        return "Very close" if frac > 0.30 else "Close" if frac > 0.10 else "Far"

    def _colour_for(self, tid, frame, bbox, n):
        c = self.color_cache.get(tid)
        if c and n - c[0] < 20:
            return c[1]
        name = ""
        if self.color_organizer:
            try:
                cols = self.color_organizer.get_dominant_colors(frame, bbox=bbox, top_n=2)
                name = cols[0]["color"] if cols else ""
            except Exception:
                pass
        self.color_cache[tid] = (n, name)
        return name

    def _analyze(self, frame, n):
        h, w = frame.shape[:2]
        fa = float(h * w)
        dets = []
        res = self.model.track(frame, persist=True, verbose=False, conf=0.35)
        boxes = res[0].boxes
        if boxes is None or len(boxes) == 0:
            return dets
        ids = boxes.id.int().tolist() if boxes.id is not None else [-(i + 1) for i in range(len(boxes))]
        for box, tid, cls, conf in zip(boxes.xyxy.tolist(), ids, boxes.cls.tolist(), boxes.conf.tolist()):
            x1, y1, x2, y2 = box
            area = max((x2 - x1) * (y2 - y1), 1.0)
            cx = (x1 + x2) / 2
            direction = "left" if cx < w / 3 else "right" if cx > 2 * w / 3 else "center"
            status = self._status_for(tid, area) if tid >= 0 else "stable"
            prox = self._proximity(area / fa)
            color = self._colour_for(tid, frame, (x1, y1, x2, y2), n)
            label = self.model.names[int(cls)]
            if self.memory and time.time() - self.mem_last.get(tid, 0) > 1.0:
                self.mem_last[tid] = time.time()
                try:
                    self.memory.remember(label, direction, color)
                except Exception as e:
                    self.log(f"Memory write error: {e}")
            dets.append(dict(id=tid, label=label, color=color, direction=direction, status=status,
                             prox=prox, conf=conf, box=(int(x1), int(y1), int(x2), int(y2)), area=area))
        if len(self.history) > 400:
            self.history.clear()
        return dets

    @staticmethod
    def _danger_list(dets):
        out = [d for d in dets if
               (d["status"] == "approaching" and d["prox"] in ("Close", "Very close")) or
               (d["prox"] == "Very close" and d["direction"] == "center")]
        return sorted(out, key=lambda d: -d["area"])

    @staticmethod
    def _name(d):
        return f"{d['color']} {d['label']}".strip()

    def _speech_policy(self, dets, danger):
        sp = self.speaker
        if not sp.enabled:
            return
        now = time.time()
        if danger:
            d = danger[0]
            key = ("danger", d["label"], d["direction"])
            if now - self.last_announced.get(key, 0) > 3.0:
                self.last_announced[key] = now
                sp.say(f"Warning! {self._name(d)} {d['status']}, {POS_PHRASE[d['direction']]}", urgent=True)
                self.last_speech = now
            return
        if sp.mode != "all" or not sp.idle() or now - self.last_speech < 2.5:
            return
        for d in sorted(dets, key=lambda d: (d["status"] != "approaching", -d["area"])):
            key = (d["status"], d["label"], d["direction"])
            if now - self.last_announced.get(key, 0) > 8.0:
                self.last_announced[key] = now
                self.last_speech = now
                sp.say(f"{self._name(d)} {POS_PHRASE[d['direction']]}, {d['status']}")
                break

    def _depth_overlay(self, frame, cache):
        try:
            import torch
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            inp = self.depth_transform(rgb).to(self._depth_device)
            with torch.no_grad():
                pred = self.depth_model(inp)
                pred = torch.nn.functional.interpolate(
                    pred.unsqueeze(1), size=rgb.shape[:2], mode="bicubic", align_corners=False).squeeze()
            dn = cv2.normalize(pred.cpu().numpy(), None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
            cache["map"] = cv2.applyColorMap(dn, cv2.COLORMAP_INFERNO)
        except Exception as e:
            self.log(f"Depth error: {e}")
            self.depth_on = False

    # ==================================================================
    # Video loop (worker thread - never touches Tk widgets)
    # ==================================================================
    def _video_loop(self, cap, source, stop_evt):
        crashed = False
        try:
            if not self._load_yolo():
                return
            self.loading = False
            n, fails, t0 = 0, 0, time.time()
            depth_cache = {}
            render_err = 0
            while not stop_evt.is_set():
                ok, frame = cap.read()
                if not ok:
                    if isinstance(source, str) and fails < 3:
                        cap.release()
                        cap = cv2.VideoCapture(source)
                        fails += 1
                        continue
                    break
                fails = 0
                n += 1
                if frame.shape[1] > 960:
                    s = 960 / frame.shape[1]
                    frame = cv2.resize(frame, (960, int(frame.shape[0] * s)))
                self.latest_frame = frame.copy()

                dets, danger = [], []
                try:
                    dets = self._analyze(frame, n)
                    danger = self._danger_list(dets)
                    self._speech_policy(dets, danger)
                except Exception as e:
                    self.log(f"Tracking error: {e}")

                shown = frame
                if self.depth_on and self.depth_model is not None:
                    if n % 3 == 1 or "map" not in depth_cache:
                        self._depth_overlay(frame, depth_cache)
                    if "map" in depth_cache and depth_cache["map"].shape[:2] == frame.shape[:2]:
                        shown = cv2.addWeighted(frame, 0.6, depth_cache["map"], 0.4, 0)

                if n % 10 == 0:
                    now = time.time()
                    self.fps = 10 / max(now - t0, 1e-6)
                    t0 = now
                self.current_objects = [dict(label=d["label"], color=d["color"], direction=d["direction"],
                                             status=d["status"], distance=d["prox"]) for d in dets]

                W, H = self.view_size
                toast = self.toast[0] if self.toast and time.time() < self.toast[1] else None
                ctx = dict(W=W, H=H, fill=self.fill, voice=self.speaker.mode, depth=self.depth_on,
                           filter=self.filter, toast=toast, fps=self.fps, t=time.time())
                try:
                    out = self.renderer.compose_live(shown, dets, danger, ctx)
                except Exception:
                    render_err += 1
                    if render_err <= 3:
                        self.log("Render error:\n" + traceback.format_exc())
                    out = (Image.fromarray(cv2.cvtColor(cv2.resize(shown, (W, H)), cv2.COLOR_BGR2RGB)), [])
                with self._lock:
                    self._frame_out, self._frame_new = out, True
        except Exception:
            crashed = True
            err = traceback.format_exc()
            self.log(f"Video loop crashed:\n{err}")
            _write_crash("VIDEO LOOP", err)
        finally:
            try:
                cap.release()
            except Exception:
                pass
            if not stop_evt.is_set():
                self.loading = False
                self.ui(lambda: self.log("Stream ended (error)" if crashed else "Stream ended"))
                stop_evt.set()
                with self._lock:
                    self._frame_out, self._frame_new = None, False

    # ==================================================================
    # Main-thread tick
    # ==================================================================
    loading = False

    def _tick(self):
        try:
            for _ in range(60):
                try:
                    self.ui_q.get_nowait()()
                except queue.Empty:
                    break
                except Exception:
                    pass
            with self._lock:
                out, new = self._frame_out, self._frame_new
                self._frame_new = False
            now = time.time()
            W, H = self.view_size
            if PIL_OK:
                if new and out is not None:
                    self._show(out[0], out[1])
                elif (self.stop_evt.is_set() or out is None) and now - self._last_idle > 0.07:
                    self._last_idle = now
                    msg = "Loading AI models" if (not self.stop_evt.is_set()) else None
                    img, hits = self.renderer.compose_idle(W, H, now, msg)
                    self._show(img, hits)
            if self.mode == "memory" and now - self._last_mem_refresh > 3:
                self._last_mem_refresh = now
                self.refresh_memory_list()
        except Exception:
            _write_crash("TICK", traceback.format_exc())
        self.root.after(25, self._tick)

    def _show(self, img, hits):
        imgtk = ImageTk.PhotoImage(image=img)
        self.canvas.itemconfig(self.img_item, image=imgtk)
        self.imgtk = imgtk
        self.hits = hits

    # ==================================================================
    # Scene / memory / chat / OCR  (same behaviour as nova_ui.py)
    # ==================================================================
    def describe_scene(self):
        objs = self.get_current_objects()
        if not objs:
            return "I don't see anything right now."
        names = {"left": "On your left", "center": "Straight ahead", "right": "On your right"}
        parts = []
        for d in ("left", "center", "right"):
            items = [o for o in objs if o["direction"] == d]
            if items:
                cnt = collections.Counter(f"{o['color']} {o['label']}".strip() for o in items)
                parts.append(f"{names[d]}: " + ", ".join(k if n == 1 else f"{n} {k}s" for k, n in cnt.items()) + ".")
        close = [o for o in objs if o.get("distance") in ("Close", "Very close")]
        if close:
            parts.append(f"{len(close)} object{'s' if len(close) > 1 else ''} very near you.")
        return " ".join(parts)

    def describe_scene_action(self):
        text = self.describe_scene()
        self.toast = (text, time.time() + 7)
        self.speaker.say(text, urgent=True, force=True)
        self.log(f"Scene: {text}")

    def _quick_memory(self, word):
        self.mem_entry.delete(0, "end")
        self.mem_entry.insert(0, f"where are my {word}?")
        self.search_memory()

    def search_memory(self):
        q = self.mem_entry.get().strip()
        if not q:
            return
        if not self.memory:
            self.mem_result.set("Memory module is not available.")
            return
        try:
            ans = self.memory.recall(q)
        except Exception as e:
            ans = f"Error: {e}"
        self.mem_result.set(ans)
        self.speaker.say(str(ans))

    def refresh_memory_list(self):
        try:
            self.memory_tree.delete(*self.memory_tree.get_children())
            if not self.memory:
                return
            for label, info in list(self.memory.records.items()):
                self.memory_tree.insert("", "end", values=(
                    label, info.get("color", "") or "-", info.get("direction", ""),
                    info.get("readable_time", "")))
        except Exception:
            pass

    def clear_memory(self):
        if self.memory and messagebox.askyesno("NOVA", "Clear all memory records?"):
            try:
                self.memory.forget_all()
                self.refresh_memory_list()
            except Exception as e:
                self.log(f"Memory clear error: {e}")

    def _chat_append(self, who, text):
        try:
            self.chat_box.configure(state="normal")
            self.chat_box.insert("end", f"{who}\n", "you" if who == "You" else "nova")
            self.chat_box.insert("end", f"{text}\n\n")
            self.chat_box.see("end")
            self.chat_box.configure(state="disabled")
        except Exception:
            pass

    def _offline_answer(self, q):
        ql = q.lower()
        if any(w in ql for w in ("around", "describe", "see", "scene")):
            return self.describe_scene()
        if any(w in ql for w in ("close", "near", "danger", "careful")):
            close = [o for o in self.get_current_objects() if o.get("distance") in ("Close", "Very close")]
            if not close:
                return "Nothing is close to you right now."
            return "Close: " + ", ".join(f"{o['label']} {POS_PHRASE[o['direction']]}" for o in close)
        if self.memory:
            return self.memory.recall(q)
        return "I can't answer that right now."

    def send_chat(self, preset=None):
        q = (preset or self.chat_entry.get()).strip()
        if not q:
            return
        self.chat_entry.delete(0, "end")
        self._chat_append("You", q)

        def work():
            try:
                reply = self.agent.handle(q) if self.agent else self._offline_answer(q)
            except Exception as e:
                try:
                    reply = self._offline_answer(q)
                except Exception:
                    reply = f"Error: {e}"
            self.ui(lambda r=reply: (self._chat_append("NOVA", r), self.refresh_memory_list()))
            self.speaker.say(str(reply))
        threading.Thread(target=work, daemon=True).start()

    def run_ocr(self):
        if self.stop_evt.is_set() or self.latest_frame is None:
            messagebox.showinfo("NOVA", "Start the camera first, then try again.")
            return
        if not OCR_OK:
            messagebox.showinfo("NOVA", "nova_ocr module not available. Install easyocr.")
            return
        frame = self.latest_frame.copy()
        self.ocr_text.delete("1.0", "end")
        self.ocr_text.insert("end", "Reading...")

        def work():
            try:
                if self.ocr_reader is None:
                    self.log("Loading OCR reader (first run can take a while)...")
                    self.ocr_reader = OCRReader()
                tmp = "_nova_ocr_tmp.png"
                cv2.imwrite(tmp, frame)
                text, info = self.ocr_reader.read_and_analyze(tmp)
                if os.path.exists(tmp):
                    os.remove(tmp)
                out = f"{text}\n\n--- Extracted info ---\n"
                out += "\n".join(f"{k}: {v}" for k, v in info.items()) if info else "Nothing found."

                def fill():
                    self.ocr_text.delete("1.0", "end")
                    self.ocr_text.insert("end", out)
                self.ui(fill)
            except Exception as e:
                self.log(f"OCR error: {e}")
                self.ui(lambda: (self.ocr_text.delete("1.0", "end"),
                                 self.ocr_text.insert("end", f"OCR error: {e}")))
        threading.Thread(target=work, daemon=True).start()

    def ocr_speak(self):
        txt = self.ocr_text.get("1.0", "end").split("--- Extracted info ---")[0].strip()
        if txt and txt != "Reading...":
            self.speaker.say(txt, urgent=True, force=True)


if __name__ == "__main__":
    root = tk.Tk()
    NovaApp(root)
    root.mainloop()
