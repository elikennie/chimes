"""
Editorial-style interface for the motion tracker: white page, centred
wordmark, small uppercase type, thin rules. Everything is drawn onto one
OpenCV canvas, with text rendered through Pillow so real fonts can be used.

The panels slide away and the camera grows to fill the window when the mouse
is idle. Full screen mode shows only the camera and the tracked objects.
"""

import time

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from chimes import TUBE_NAMES, depth_for, tube_for

# Page layout (pixels).
W, H = 1280, 760
MARGIN = 32
SIDE_W = 180
GAP = 36
CAM_X = MARGIN + SIDE_W + GAP
CAM_W = W - CAM_X - MARGIN
CAM_H = CAM_W * 9 // 16
CAM_Y = 104
NAV_BASELINE = 58

PROC_W = 800           # width the motion detector works at (slider px are in this scale)
FULL_W = 1920          # full screen image width; the window scales it to the screen
FULL_BAR_H = 76        # height of the white bar shown in full screen on mouse move
STRIKE_FLASH = 0.35    # seconds a box glows after ringing a chime
STRIP_FLASH = 0.5      # seconds a grid strip glows after its chime rings
IDLE_SECONDS = 5   # how long without mouse movement before panels slide away

# Colours are BGR (OpenCV order); the greys are the same either way.
INK = (17, 17, 17)
GREY = (125, 125, 125)
RULE = (222, 222, 222)
PAPER = (255, 255, 255)

# First font found wins: macOS Helvetica Neue, Windows Arial, Linux DejaVu.
FONT_FILES = {
    "regular": [
        ("/System/Library/Fonts/HelveticaNeue.ttc", 0),
        ("/System/Library/Fonts/Helvetica.ttc", 0),
        (r"C:\Windows\Fonts\arial.ttf", 0),
        ("DejaVuSans.ttf", 0),
    ],
    "bold": [
        ("/System/Library/Fonts/HelveticaNeue.ttc", 1),
        ("/System/Library/Fonts/Helvetica.ttc", 1),
        (r"C:\Windows\Fonts\arialbd.ttf", 0),
        ("DejaVuSans-Bold.ttf", 0),
    ],
}

_font_cache = {}


def load_font(weight, size):
    key = (weight, size)
    if key not in _font_cache:
        for path, index in FONT_FILES[weight]:
            try:
                _font_cache[key] = ImageFont.truetype(path, size, index=index)
                break
            except OSError:
                continue
        else:
            _font_cache[key] = ImageFont.load_default(size=size)
    return _font_cache[key]


def in_rect(x, y, rect):
    x0, y0, x1, y1 = rect
    return x0 <= x <= x1 and y0 <= y <= y1


class Slider:
    """Thin-rule slider with -/+ steppers. `curve` > 1 gives finer control at the low end."""

    def __init__(self, label, lo, hi, value, step, curve=1.0, unit=" px"):
        self.label, self.lo, self.hi, self.step, self.curve = label, lo, hi, step, curve
        self.unit = unit
        self.value = value
        self.x, self.y = 0, 0  # label baseline; set by the layout
        self.minus_rect = self.plus_rect = (0, 0, 0, 0)  # set while drawing

    @property
    def value(self):
        return int(round(self.lo + (self.hi - self.lo) * self.t ** self.curve))

    @value.setter
    def value(self, v):
        v = min(max(v, self.lo), self.hi)
        self.t = ((v - self.lo) / (self.hi - self.lo)) ** (1 / self.curve)

    @property
    def track_y(self):
        return self.y + 22

    def hit(self, x, y):
        return self.x - 8 <= x <= self.x + SIDE_W + 8 and abs(y - self.track_y) <= 12

    def set_from_x(self, x):
        self.t = min(max((x - self.x) / SIDE_W, 0.0), 1.0)

    def nudge(self, direction):
        self.value = self.value + direction * self.step


class Link:
    """A clickable uppercase word, underlined when active or hovered."""

    def __init__(self, text, action, x, baseline, font):
        self.action, self.x, self.baseline, self.font = action, x, baseline, font
        self.set_text(text)

    def set_text(self, text):
        self.text = text
        self.width = int(self.font.getlength(text))

    def hit(self, x, y):
        return (self.x - 6 <= x <= self.x + self.width + 6
                and self.baseline - self.font.size - 6 <= y <= self.baseline + 8)




class Interface:
    def __init__(self, source_label):
        self.source_label = source_label
        self.fonts = {
            "word": load_font("bold", 32),
            "nav": load_font("regular", 13),
            "head": load_font("bold", 13),
            "body": load_font("regular", 13),
        }
        # Both measured in detector pixels (PROC_W wide). Min size is the side
        # of the smallest square of movement counted; max jump is how far an
        # object may move between frames and keep its number.
        self.sliders = [
            Slider("Min size", 4, 160, 28, step=1, curve=2.0),
            Slider("Max jump", 10, 2000, 120, step=5, curve=2.0),
            Slider("Chime rate", 0, 100, 50, step=5, unit="%"),
        ]
        for i, s in enumerate(self.sliders):
            s.x, s.y = MARGIN, CAM_Y + 44 + i * 64

        self.options_y = CAM_Y + 260
        self.links = [
            Link(text, action, MARGIN, self.options_y + 28 + i * 20, self.fonts["body"])
            for i, (text, action) in enumerate((
                ("Live camera", "live"), ("Motion mask", "mask"),
                ("Full screen", "fullscreen"), ("Reset background", "reset"),
                ("Sound: FL Studio", "sound"), ("Grid: Off", "grid"), ("Labels: On", "labels"),
            ))
        ]
        self.exit_link = Link("EXIT FULL SCREEN", "fullscreen", 44, 47, load_font("regular", 18))

        self.view = "live"
        self.fullscreen = False
        self.sound_label = "FL Studio"   # set by the main loop
        self.sound_status = ""
        self.tube_strikes = [-1.0] * len(TUBE_NAMES)  # when each chime last rang; set by the main loop
        self.show_grid = False
        self.show_labels = True
        self.mouse = (-1, -1)
        self.dragging = None
        self.actions = []
        self.texts = []
        self.last_move = time.monotonic()
        self.reveal = 1.0  # 1 = panels fully shown, 0 = slid away

    # ----- input -------------------------------------------------------

    @property
    def min_size(self):
        return self.sliders[0].value

    @property
    def max_jump(self):
        return self.sliders[1].value

    @property
    def chime_rate(self):
        return self.sliders[2].value / 100

    def wake(self):
        self.last_move = time.monotonic()

    def on_mouse(self, event, x, y, flags, _param):
        if (x, y) != self.mouse or event != cv2.EVENT_MOUSEMOVE:
            self.wake()
        self.mouse = (x, y)

        if event == cv2.EVENT_LBUTTONDOWN:
            # Ignore clicks on panels that are still sliding in.
            if self.fullscreen:
                if self.reveal > 0.5 and self.exit_link.hit(x, y):
                    self.actions.append("fullscreen")
                return
            if self.reveal < 0.9:
                return
            for s in self.sliders:
                if in_rect(x, y, s.minus_rect):
                    s.nudge(-1)
                    return
                if in_rect(x, y, s.plus_rect):
                    s.nudge(1)
                    return
                if s.hit(x, y):
                    self.dragging = s
                    s.set_from_x(x)
                    return
            for link in self.links:
                if link.hit(x, y):
                    self.actions.append(link.action)
        elif event == cv2.EVENT_MOUSEMOVE and self.dragging:
            self.dragging.set_from_x(x)
        elif event == cv2.EVENT_LBUTTONUP:
            self.dragging = None

    def take_actions(self):
        actions, self.actions = self.actions, []
        return actions

    @staticmethod
    def fit(frame):
        """Shrink the camera frame to the detector's working size."""
        h, w = frame.shape[:2]
        return cv2.resize(frame, (PROC_W, round(h * PROC_W / w)), interpolation=cv2.INTER_AREA)

    # ----- drawing -----------------------------------------------------

    def text(self, xy, text, font="body", colour=INK, anchor="ls", spacing=0):
        if isinstance(font, str):
            font = self.fonts[font]
        self.texts.append((xy, text, font, colour, anchor, spacing))

    def underline(self, canvas, x, width, baseline, colour=INK, thickness=1):
        cv2.line(canvas, (x, baseline + 4), (x + width, baseline + 4), colour, thickness)

    def update_reveal(self):
        idle = time.monotonic() - self.last_move
        target = 1.0 if self.dragging or idle < IDLE_SECONDS else 0.0
        self.reveal += (target - self.reveal) * 0.18
        if abs(target - self.reveal) < 0.002:
            self.reveal = target
        r = self.reveal
        return 1.0 - r * r * (3 - 2 * r)  # eased "how hidden", 0..1

    def render(self, frame, mask, tracks, fps, raw=None):
        hidden = self.update_reveal()
        if self.fullscreen:
            return self.render_full(frame, mask, tracks, raw, hidden)

        # Camera at rest: as large as fits beside the left column.
        fh, fw = frame.shape[:2]
        s = min(CAM_W / fw, CAM_H / fh)
        rest = (CAM_X, CAM_Y + (CAM_H - int(fh * s)) // 2, CAM_X + int(fw * s), 0)
        rest = (rest[0], rest[1], rest[2], rest[1] + int(fh * s))

        panels = self.render_panels(rest, tracks, fps)
        out = panels if hidden == 0 else self.slide_panels(panels, rest, hidden)

        # As the panels slide away the camera grows to fill the window.
        rect = tuple(round(a + (b - a) * hidden) for a, b in zip(rest, (0, 0, W, H)))
        self.draw_camera(out, rect, frame, mask, tracks, raw)
        if self.view == "mask" and hidden < 1:
            x0, y0, x1, y1 = rect
            cv2.rectangle(out, (x0 - 1, y0 - 1), (x1, y1), RULE, 1)
        return out

    def render_panels(self, rest, tracks, fps):
        canvas = np.full((H, W, 3), PAPER, np.uint8)
        self.texts = []
        self.draw_header(len(tracks), fps)
        self.draw_controls(canvas)
        self.draw_objects(canvas, tracks)
        self.draw_caption(rest[0], rest[3], len(tracks))
        return self.flush_text(canvas)

    @staticmethod
    def slide_panels(page, rest, hidden):
        """Slide each panel off its nearest edge."""
        out = np.full_like(page, PAPER)
        bottom = rest[3]
        bands = [
            ((0, 0, W, CAM_Y - 2), 0, -CAM_Y),                     # header up
            ((0, CAM_Y - 2, CAM_X - 2, H), -CAM_X, 0),             # left column left
            ((CAM_X - 2, bottom + 2, W, H), 0, H - bottom),        # caption down
        ]
        for (x0, y0, x1, y1), dx, dy in bands:
            dx, dy = int(dx * hidden), int(dy * hidden)
            tx0, ty0 = max(x0 + dx, 0), max(y0 + dy, 0)
            tx1, ty1 = min(x1 + dx, W), min(y1 + dy, H)
            if tx0 < tx1 and ty0 < ty1:
                out[ty0:ty1, tx0:tx1] = page[ty0 - dy:ty1 - dy, tx0 - dx:tx1 - dx]
        return out

    def draw_camera(self, canvas, rect, frame, mask, tracks, raw):
        """Fill `rect` with the camera image (cropping to cover it) and draw the boxes."""
        x0, y0, x1, y1 = rect
        rw, rh = x1 - x0, y1 - y0
        if rw <= 0 or rh <= 0:
            return
        if self.view == "mask":
            src = mask
        else:
            src = raw if raw is not None else frame
        sh, sw = src.shape[:2]
        s = max(rw / sw, rh / sh)
        size = (max(rw, round(sw * s)), max(rh, round(sh * s)))
        interp = cv2.INTER_NEAREST if src is mask else (cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR)
        img = cv2.resize(src, size, interpolation=interp)
        cx, cy = (size[0] - rw) // 2, (size[1] - rh) // 2
        img = img[cy:cy + rh, cx:cx + rw]
        if src is mask:
            img = cv2.cvtColor(255 - img, cv2.COLOR_GRAY2BGR)
        region = canvas[y0:y1, x0:x1]
        region[:] = img

        # Same cover mapping, from detector pixels to this region.
        fh, fw = frame.shape[:2]
        k = max(rw / fw, rh / fh)
        offx, offy = (rw - fw * k) / 2, (rh - fh * k) / 2
        self.texts = []
        now = time.monotonic()
        if self.show_grid:
            self.draw_grid(region, tracks, fw, k, offx, now)
        self.draw_boxes(region, tracks, k, offx, offy, now)
        if self.show_labels:
            for t in tracks:
                self.draw_tag(region, t, fw, k, offx, offy, now)
        region[:] = self.flush_text(region)

    def draw_grid(self, region, tracks, fw, k, offx, now):
        """One strip per chime, labelled with its note; strips glow when they ring."""
        rh, rw = region.shape[:2]
        n = len(TUBE_NAMES)
        edges = [int(i * fw / n * k + offx) for i in range(n + 1)]
        occupied = {tube_for(t.hotspot[0], fw) for t in tracks}
        u = max(1.0, k * 0.8)
        font = load_font("bold", round(11 * u))
        for i, name in enumerate(TUBE_NAMES):
            x0, x1 = max(edges[i], 0), min(edges[i + 1], rw)
            if x1 <= x0:
                continue
            age = now - self.tube_strikes[i]
            if age < STRIP_FLASH:
                a = 0.22 * (1 - age / STRIP_FLASH)
                strip = region[:, x0:x1]
                strip[:] = (strip * (1 - a) + 255 * a).astype(np.uint8)
            # Note tag at the top of the strip; inverted when something is moving in it.
            tw, th = int(font.getlength(name) + 14 * u), int(18 * u)
            tx, ty = (x0 + x1 - tw) // 2, int(8 * u)
            fill, ink = (INK, PAPER) if i in occupied else (PAPER, INK)
            cv2.rectangle(region, (tx, ty), (tx + tw, ty + th), fill, -1)
            self.text(((x0 + x1) // 2, ty + int(13 * u)), name, font, ink, anchor="ms")
        for x in edges[1:-1]:
            if 0 <= x < rw:
                region[:, x] = (region[:, x] * 0.45 + 255 * 0.55).astype(np.uint8)

    def draw_tag(self, region, t, fw, k, offx, offy, now):
        """White info tag on each box: number and note, speed and time moving, depth bar."""
        rh, rw = region.shape[:2]
        u = max(1.0, k * 0.8)
        bold, regular = load_font("bold", round(10 * u)), load_font("regular", round(10 * u))
        line1 = f"NO. {t.id:02d}   {TUBE_NAMES[tube_for(t.hotspot[0], fw)]}"
        line2 = f"{t.drive:.0f} px/f  ·  {now - t.born:.1f} s"
        pad = int(7 * u)
        tw = int(max(bold.getlength(line1), regular.getlength(line2))) + 2 * pad + int(2 * u)
        th = int(37 * u)
        bx, by = int(t.box[0] * k + offx), int(t.box[1] * k + offy)
        tx = min(max(bx, 0), max(rw - tw, 0))
        ty = by - th - int(3 * u) if by - th - int(3 * u) >= 0 else max(by, 0) + int(3 * u)

        cv2.rectangle(region, (tx, ty), (tx + tw, ty + th), PAPER, -1)
        cv2.rectangle(region, (tx, ty), (tx + max(2, int(2 * u)), ty + th), t.colour, -1)
        left = tx + pad + int(2 * u)
        self.text((left, ty + int(14 * u)), line1, bold)
        self.text((left, ty + int(26 * u)), line2, regular, GREY)
        # Depth bar: how much delay/reverb this object's chimes have built up.
        bar_y, bar_w = ty + int(31 * u), tw - (left - tx) - pad
        cv2.line(region, (left, bar_y), (left + bar_w, bar_y), RULE, max(1, int(2 * u)))
        bar_colour = INK if t.colour == PAPER else t.colour  # white wouldn't show on the tag
        cv2.line(region, (left, bar_y), (left + int(bar_w * depth_for(t, now)), bar_y),
                 bar_colour, max(1, int(2 * u)))

    @staticmethod
    def draw_boxes(region, tracks, k, offx, offy, now):
        thin, thick = max(1, round(k)), max(3, round(3 * k))
        for t in tracks:
            bx, by, bw, bh = t.box
            x, y = int(bx * k + offx), int(by * k + offy)
            w, h = int(bw * k), int(bh * k)
            # Colour wash that fades out after the object rings a chime.
            age = now - t.struck_at
            if age < STRIKE_FLASH:
                alpha = 0.4 * (1 - age / STRIKE_FLASH)
                rh, rw = region.shape[:2]
                roi = region[max(y, 0):min(y + h, rh), max(x, 0):min(x + w, rw)]
                roi[:] = (roi * (1 - alpha) + np.array(t.colour) * alpha).astype(np.uint8)
            cv2.rectangle(region, (x, y), (x + w, y + h), t.colour, thin, cv2.LINE_AA)
            # Heavier corner brackets over the hairline box.
            arm = int(max(6 * k, min(16 * k, w // 4, h // 4)))
            for cx, cy, dx, dy in ((x, y, 1, 1), (x + w, y, -1, 1), (x, y + h, 1, -1), (x + w, y + h, -1, -1)):
                cv2.line(region, (cx, cy), (cx + dx * arm, cy), t.colour, thick)
                cv2.line(region, (cx, cy), (cx, cy + dy * arm), t.colour, thick)
            # Hot spot: the part of this object that's actually moving (what plays the note).
            hx, hy = int(t.hotspot[0] * k + offx), int(t.hotspot[1] * k + offy)
            r = max(4, int(4 * k))
            cv2.circle(region, (hx, hy), r + thin, PAPER, -1, cv2.LINE_AA)
            cv2.circle(region, (hx, hy), r, t.colour, -1, cv2.LINE_AA)

    def render_full(self, frame, mask, tracks, raw, hidden):
        """Camera and boxes only, edge to edge."""
        fh, fw = frame.shape[:2]
        full_h = round(fh * FULL_W / fw)
        canvas = np.empty((full_h, FULL_W, 3), np.uint8)
        self.draw_camera(canvas, (0, 0, FULL_W, full_h), frame, mask, tracks, raw)
        if hidden >= 1:
            return canvas

        # White bar that drops in on mouse movement.
        top = -int(FULL_BAR_H * hidden)
        cv2.rectangle(canvas, (0, top), (FULL_W, top + FULL_BAR_H - 1), PAPER, -1)
        self.texts = []
        link = self.exit_link
        self.text((link.x, top + link.baseline), link.text, link.font)
        if link.hit(*self.mouse):
            self.underline(canvas, link.x, link.width, top + link.baseline, thickness=2)
        self.text((FULL_W // 2, top + 51), "CHIMES", load_font("bold", 30), anchor="ms", spacing=1)
        self.text((FULL_W - 44, top + 47), f"TRACKING ({len(tracks)})", link.font, anchor="rs")
        return self.flush_text(canvas)

    def draw_header(self, count, fps):
        self.text((MARGIN, NAV_BASELINE), self.source_label, "nav")
        self.text((MARGIN + int(self.fonts["nav"].getlength(self.source_label)) + 28, NAV_BASELINE),
                  f"{fps:.0f} FPS", "nav")
        self.text((W // 2, NAV_BASELINE + 6), "CHIMES", "word", anchor="ms", spacing=1)
        tracking = f"TRACKING ({count})"
        self.text((W - MARGIN, NAV_BASELINE), tracking, "nav", anchor="rs")
        self.text((W - MARGIN - int(self.fonts["nav"].getlength(tracking)) - 28, NAV_BASELINE),
                  self.sound_status, "nav", anchor="rs")

    def draw_controls(self, canvas):
        self.text((MARGIN, CAM_Y + 10), "SENSITIVITY", "head")
        body = self.fonts["body"]
        for s in self.sliders:
            self.text((s.x, s.y), s.label, "body")

            # Right side reads "-  28 px  +".
            right = s.x + SIDE_W
            value = f"{s.value:,}{s.unit}"
            plus_x = right - int(body.getlength("+"))
            value_right = plus_x - 12
            minus_x = value_right - int(body.getlength(value)) - 12 - int(body.getlength("–"))
            self.text((plus_x, s.y), "+", "body")
            self.text((value_right, s.y), value, "body", GREY, anchor="rs")
            self.text((minus_x, s.y), "–", "body")
            s.plus_rect = (plus_x - 6, s.y - 16, right + 6, s.y + 6)
            s.minus_rect = (minus_x - 6, s.y - 16, minus_x + 14, s.y + 6)

            ty = s.track_y
            hx = s.x + int(s.t * SIDE_W)
            cv2.line(canvas, (s.x, ty), (s.x + SIDE_W, ty), RULE, 1)
            cv2.line(canvas, (s.x, ty), (hx, ty), INK, 2)
            cv2.rectangle(canvas, (hx - 5, ty - 5), (hx + 5, ty + 5), INK, -1)

        # Options list, styled like a sort menu: the active view is underlined.
        self.text((MARGIN, self.options_y), "OPTIONS", "head")
        mx, my = self.mouse
        for link in self.links:
            if link.action == "sound":
                link.set_text(f"Sound: {self.sound_label}")
            elif link.action == "grid":
                link.set_text(f"Grid: {'On' if self.show_grid else 'Off'}")
            elif link.action == "labels":
                link.set_text(f"Labels: {'On' if self.show_labels else 'Off'}")
            self.text((link.x, link.baseline), link.text, "body")
            if link.action == self.view or link.hit(mx, my):
                self.underline(canvas, link.x, link.width, link.baseline)

    def draw_objects(self, canvas, tracks):
        """List of tracked objects under the options, with colour swatches."""
        top = self.links[-1].baseline + 52
        self.text((MARGIN, top), "OBJECTS", "head")
        all_label = f"ALL ({len(tracks)})"
        self.text((MARGIN, top + 28), all_label, "body")
        self.underline(canvas, MARGIN, int(self.fonts["body"].getlength(all_label)), top + 28)

        if not tracks:
            self.text((MARGIN, top + 52), "Nothing moving", "body", GREY)
            return

        rows = sorted(tracks, key=lambda t: t.id)
        max_rows = (H - MARGIN - (top + 52)) // 20 + 1
        if len(rows) > max_rows:
            shown = rows[:max_rows - 1]
        else:
            shown = rows
        for i, t in enumerate(shown):
            y = top + 52 + i * 20
            cv2.rectangle(canvas, (MARGIN, y - 9), (MARGIN + 8, y - 1), t.colour, -1)
            cv2.rectangle(canvas, (MARGIN, y - 9), (MARGIN + 8, y - 1), RULE, 1)
            self.text((MARGIN + 18, y), f"No. {t.id:02d}  {t.colour_name}", "body")
            self.text((MARGIN + SIDE_W, y), f"{t.drive:.0f} px/f", "body", GREY, anchor="rs")
        if len(shown) < len(rows):
            self.text((MARGIN + 18, top + 52 + len(shown) * 20), f"+ {len(rows) - len(shown)} more",
                      "body", GREY)

    def draw_caption(self, x, bottom, count):
        y = bottom + 30
        self.text((x, y), "CHIMES", "body")
        self.text((x, y + 20), "Motion Mask" if self.view == "mask" else "Live Camera", "body")
        noun = "object" if count == 1 else "objects"
        self.text((x, y + 40), f"{count} {noun} in motion", "body")

    def flush_text(self, canvas):
        img = Image.fromarray(canvas)
        draw = ImageDraw.Draw(img)
        for (x, y), text, font, colour, anchor, spacing in self.texts:
            if not spacing:
                draw.text((x, y), text, font=font, fill=colour, anchor=anchor)
                continue
            # Letter-spaced text (the wordmark), drawn one character at a time.
            total = sum(font.getlength(c) for c in text) + spacing * (len(text) - 1)
            cx = x - total / 2 if anchor[0] == "m" else x
            for c in text:
                draw.text((cx, y), c, font=font, fill=colour, anchor="l" + anchor[1])
                cx += font.getlength(c) + spacing
        return np.array(img)
