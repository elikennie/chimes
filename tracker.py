"""
Motion tracker: finds moving things in the camera frame, draws a coloured
box around each one, and follows it with an ID and a trail.

Step 1 of motion-chimes. Later the tracked objects will drive MIDI notes.

Run:   python tracker.py               (default webcam)
       python tracker.py --camera 1    (another camera)
       python tracker.py --video clip.mp4

Sound: chimes go to FL Studio over a MIDI port called "Chimes" (see README),
       or to the built-in synth with --sound builtin. Click "Sound" to switch.

Keys:  q        quit
       f        full screen on/off (Esc also leaves full screen)
       m        switch between camera and motion mask (what the computer "sees")
       r        reset the background (use after moving the camera)
       s        switch sound: FL Studio / built-in / off
       g        show the chime grid (which note each strip plays)
       l        show/hide the info tags on each box
The sliders and the Options list on the left can also be clicked.
Leave the mouse still and the panels slide away while the camera fills the window.
"""

import argparse
import math
import os
import sys
import time
from collections import deque

import cv2
import numpy as np

import ui as page
from chimes import SoundEngine
from ui import Interface

WINDOW = "Chimes"

MIN_MOTION_PX = 30      # changed pixels needed for a patch to count as the hot spot
MAX_HOTSPOT_STEP = 60   # cap on hot spot movement per frame, so jumps between body parts don't spike

# Distinct BGR colours, cycled through as new objects appear.
PALETTE = [
    ("Red", (48, 59, 255)), ("Blue", (255, 132, 10)), ("Yellow", (10, 214, 255)),
    ("Green", (88, 209, 48)), ("Orange", (10, 159, 255)), ("Purple", (242, 90, 191)),
    ("Cyan", (255, 210, 100)), ("White", (255, 255, 255)),
]


class Track:
    def __init__(self, track_id, box):
        self.id = track_id
        self.box = box  # (x, y, w, h)
        self.colour_name, self.colour = PALETTE[track_id % len(PALETTE)]
        self.trail = deque([self.centre], maxlen=40)
        self.missed = 0
        self.speed = 0.0  # pixels per frame, smoothed
        # Hot spot: the centre of the biggest patch of frame-to-frame change inside
        # the box. On a person that's the waving hand, not the torso the box centres on.
        self.hotspot = self.centre
        self.hot_speed = 0.0
        self.hot_fresh = True
        self.drive = 0.0  # speed that drives the chimes: box or hot spot, whichever is faster
        self.struck_at = -1.0  # when this object last rang a chime
        self.born = time.monotonic()  # when it started moving

    @property
    def centre(self):
        x, y, w, h = self.box
        return (x + w // 2, y + h // 2)

    def update(self, box):
        old = self.centre
        self.box = box
        new = self.centre
        step = math.dist(old, new)
        self.speed = 0.7 * self.speed + 0.3 * step
        self.trail.append(new)
        self.missed = 0


    def measure(self, motion):
        """Find the hot spot in this frame's pixel changes and how fast it's moving."""
        x, y, w, h = self.box
        patch = motion[y:y + h, x:x + w]
        n, _labels, stats, centres = cv2.connectedComponentsWithStats(patch)
        best = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA])) if n > 1 else 0
        if best and stats[best, cv2.CC_STAT_AREA] >= MIN_MOTION_PX:
            new = (x + centres[best][0], y + centres[best][1])
            if not self.hot_fresh:
                step = min(math.dist(self.hotspot, new), MAX_HOTSPOT_STEP)
                self.hot_speed = 0.6 * self.hot_speed + 0.4 * step
            self.hotspot, self.hot_fresh = new, False
        else:
            self.hotspot, self.hot_fresh = self.centre, True
            self.hot_speed *= 0.8
        self.drive = max(self.speed, self.hot_speed)


class CentroidTracker:
    """Matches each frame's boxes to existing tracks by nearest centre."""

    def __init__(self, max_distance=120, max_missed=10):
        self.tracks = {}
        self.next_id = 0
        self.max_distance = max_distance
        self.max_missed = max_missed

    def update(self, boxes):
        centres = [(x + w // 2, y + h // 2) for x, y, w, h in boxes]
        unmatched = set(range(len(boxes)))

        # Greedy matching: closest (track, box) pairs first.
        pairs = sorted(
            (math.dist(t.centre, c), tid, i)
            for tid, t in self.tracks.items()
            for i, c in enumerate(centres)
        )
        matched_tracks = set()
        for dist, tid, i in pairs:
            if dist > self.max_distance:
                break
            if tid in matched_tracks or i not in unmatched:
                continue
            self.tracks[tid].update(boxes[i])
            matched_tracks.add(tid)
            unmatched.discard(i)

        for tid in list(self.tracks):
            if tid not in matched_tracks:
                self.tracks[tid].missed += 1
                if self.tracks[tid].missed > self.max_missed:
                    del self.tracks[tid]

        for i in unmatched:
            self.tracks[self.next_id] = Track(self.next_id, boxes[i])
            self.next_id += 1

        # Only report tracks seen this frame.
        return [t for t in self.tracks.values() if t.missed == 0]

    def measure(self, motion):
        for t in self.tracks.values():
            if t.missed == 0:
                t.measure(motion)


class MotionDetector:
    def __init__(self):
        self.reset()

    def reset(self):
        # Learns what the "still" background looks like and flags what differs.
        self.bg = cv2.createBackgroundSubtractorMOG2(
            history=500, varThreshold=32, detectShadows=True
        )
        self.prev_gray = None

    def detect(self, frame, min_area):
        small = cv2.GaussianBlur(frame, (5, 5), 0)
        mask = self.bg.apply(small)
        # Shadows are marked 127; keep only real foreground (255).
        _, mask = cv2.threshold(mask, 200, 255, cv2.THRESH_BINARY)
        # Remove speckle, then join nearby pieces of the same object.
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        mask = cv2.dilate(mask, np.ones((15, 15), np.uint8), iterations=2)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        boxes = [cv2.boundingRect(c) for c in contours if cv2.contourArea(c) >= min_area]

        # Frame-to-frame change: shows which part of a shape is actually moving.
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        if self.prev_gray is None or self.prev_gray.shape != gray.shape:
            self.prev_gray = gray
        _, motion = cv2.threshold(cv2.absdiff(gray, self.prev_gray), 20, 255, cv2.THRESH_BINARY)
        motion = cv2.dilate(motion, np.ones((5, 5), np.uint8))
        self.prev_gray = gray
        return boxes, mask, motion


def open_capture(args):
    if args.video:
        return cv2.VideoCapture(args.video)
    # DirectShow opens webcams much faster on Windows than the default backend.
    if sys.platform.startswith("win"):
        return cv2.VideoCapture(args.camera, cv2.CAP_DSHOW)
    return cv2.VideoCapture(args.camera)


def main():
    parser = argparse.ArgumentParser(description="Highlight and track moving objects.")
    parser.add_argument("--camera", type=int, default=0, help="camera index (default 0)")
    parser.add_argument("--video", help="use a video file instead of a camera")
    parser.add_argument("--sound", choices=SoundEngine.MODES, default="fl",
                        help="fl = MIDI to FL Studio (default), builtin = built-in chimes, off")
    args = parser.parse_args()

    cap = open_capture(args)
    if not cap.isOpened():
        sys.exit("Could not open the camera/video. On macOS, allow camera access for "
                 "your terminal app in System Settings > Privacy & Security > Camera.")

    source = os.path.basename(args.video).upper() if args.video else f"CAMERA {args.camera}"
    ui = Interface(source)
    detector = MotionDetector()
    tracker = CentroidTracker()
    sound = SoundEngine(args.sound)

    # A resizable window is needed for full screen to work.
    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
    cv2.resizeWindow(WINDOW, page.W, page.H)
    cv2.setMouseCallback(WINDOW, ui.on_mouse)
    keys = {ord("m"): "toggle", ord("r"): "reset", ord("f"): "fullscreen", ord("s"): "sound", ord("g"): "grid", ord("l"): "labels",
            ord("q"): "quit", 27: "escape"}

    def set_fullscreen(on):
        ui.fullscreen = on
        ui.wake()
        cv2.setWindowProperty(WINDOW, cv2.WND_PROP_FULLSCREEN,
                              cv2.WINDOW_FULLSCREEN if on else cv2.WINDOW_NORMAL)
        if not on:
            cv2.resizeWindow(WINDOW, page.W, page.H)

    fps, last = 0.0, time.perf_counter()

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        raw, frame = frame, ui.fit(frame)
        tracker.max_distance = ui.max_jump
        boxes, mask, motion = detector.detect(frame, ui.min_size ** 2)
        tracks = tracker.update(boxes)
        tracker.measure(motion)

        sound.brain.activity = ui.chime_rate
        now = time.monotonic()
        for t in sound.update(tracks, frame.shape[1]):
            t.struck_at = now
        ui.sound_label, ui.sound_status = sound.label, sound.status
        ui.tube_strikes = sound.brain.last_strike

        now = time.perf_counter()
        fps = 0.9 * fps + 0.1 / max(now - last, 1e-6)
        last = now

        cv2.imshow(WINDOW, ui.render(frame, mask, tracks, fps, raw))

        key = cv2.waitKey(1) & 0xFF
        actions = ui.take_actions() + ([keys[key]] if key in keys else [])
        if ("quit" in actions or ("escape" in actions and not ui.fullscreen)
                or cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1):
            break
        for action in actions:
            if action == "toggle":
                ui.view = "live" if ui.view == "mask" else "mask"
            elif action == "fullscreen":
                set_fullscreen(not ui.fullscreen)
            elif action == "escape":
                set_fullscreen(False)
            elif action in ("live", "mask"):
                ui.view = action
            elif action == "grid":
                ui.show_grid = not ui.show_grid
            elif action == "labels":
                ui.show_labels = not ui.show_labels
            elif action == "sound":
                sound.cycle()
            elif action == "reset":
                detector.reset()
                tracker = CentroidTracker(tracker.max_distance)

    sound.close()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
