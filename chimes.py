"""
Turns tracked motion into wind chime strikes.

The frame is split into vertical strips, one per tuned tube. Each moving
object can knock the tube under it: the faster it moves, the more likely a
strike each frame and the harder it hits. The longer an object has been
moving, the more delay and reverb its chimes get ("depth"), so steady motion
builds into a washed-out, ethereal sound.

Strikes go out as MIDI (for FL Studio) or to a small built-in chime synth.
"""

import random
import sys
import threading
import time

import numpy as np

# C major pentatonic, high register: any combination sounds consonant.
TUBES = [72, 74, 76, 79, 81, 84, 86, 88]

TUBE_COOLDOWN = 0.15    # seconds before the same tube can ring again
MAX_STRIKES_PER_FRAME = 3
NOTE_LENGTH = 2.0       # seconds before MIDI note-off, doubled at full depth
PORT_NAME = "Chimes"

DEPTH_MIN = 0.1         # a little space even on a brand new object
DEPTH_SECONDS = 6.0     # seconds of movement to reach full delay + reverb
CC_REVERB = 91          # MIDI CCs carrying depth, to link to FL Studio knobs
CC_DELAY = 92

NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def note_name(midi):
    return f"{NOTE_NAMES[midi % 12]}{midi // 12 - 1}"


TUBE_NAMES = [note_name(n) for n in TUBES]


def tube_for(x, frame_width):
    """Which tube's strip a horizontal position falls in."""
    return min(max(int(x / frame_width * len(TUBES)), 0), len(TUBES) - 1)


def depth_for(track, now):
    """0..1: how much delay/reverb this object's chimes get, growing as it keeps moving."""
    return DEPTH_MIN + (1 - DEPTH_MIN) * min(1.0, (now - track.born) / DEPTH_SECONDS)


class ChimeBrain:
    """Decides which tubes get struck from the current tracked objects."""

    def __init__(self):
        self.activity = 0.5  # 0..1, from the Chime rate slider
        self.last_strike = [-1.0] * len(TUBES)

    def update(self, tracks, frame_width, now):
        strikes = []
        for t in sorted(tracks, key=lambda t: -t.drive):
            if len(strikes) >= MAX_STRIKES_PER_FRAME:
                break
            if t.drive < 1.0:
                continue
            # Faster movement = more likely to ring this frame.
            chance = min(1.0, (t.drive / 25) ** 1.2) * self.activity * 0.6
            if random.random() > chance:
                continue
            tube = tube_for(t.hotspot[0], frame_width)  # where it's moving, not the box centre
            # Tubes swing into their neighbours now and then.
            if random.random() < 0.3:
                tube = min(max(tube + random.choice((-1, 1)), 0), len(TUBES) - 1)
            if now - self.last_strike[tube] < TUBE_COOLDOWN:
                continue
            self.last_strike[tube] = now
            velocity = int(np.clip(35 + t.drive * 3 + random.uniform(-10, 10), 20, 120))
            strikes.append((tube, velocity, depth_for(t, now), t))
        return strikes


class MidiOut:
    """Sends notes to FL Studio (or any app) through a MIDI port named Chimes."""

    def __init__(self):
        import mido
        self.mido = mido
        if sys.platform.startswith("win"):
            # Windows can't create ports itself; use one made in loopMIDI.
            names = [n for n in mido.get_output_names() if PORT_NAME.lower() in n.lower()]
            if not names:
                raise RuntimeError("no MIDI port")
            self.port = mido.open_output(names[0])
        else:
            # macOS/Linux: create a virtual port that FL Studio sees as an input.
            self.port = mido.open_output(PORT_NAME, virtual=True)
        self.note_offs = {}  # note -> time to release
        self.sent_depth = None

    def strike(self, tube, velocity, depth, now):
        note = TUBES[tube]
        if note in self.note_offs:
            self.port.send(self.mido.Message("note_off", note=note))
        self.port.send(self.mido.Message("note_on", note=note, velocity=velocity))
        self.note_offs[note] = now + NOTE_LENGTH * (1 + depth)

    def set_depth(self, depth):
        """CCs apply to the whole channel, so send the overall depth, only when it changes."""
        value = int(round(depth * 127))
        if self.sent_depth is None or abs(value - self.sent_depth) >= 2:
            self.port.send(self.mido.Message("control_change", control=CC_REVERB, value=value))
            self.port.send(self.mido.Message("control_change", control=CC_DELAY, value=value))
            self.sent_depth = value

    def tick(self, now):
        for note, due in list(self.note_offs.items()):
            if now >= due:
                self.port.send(self.mido.Message("note_off", note=note))
                del self.note_offs[note]

    def close(self):
        for note in self.note_offs:
            self.port.send(self.mido.Message("note_off", note=note))
        self.note_offs.clear()
        self.port.close()


def render_chime(note, rate, seconds=4.0):
    """A tubular chime: a few inharmonic partials, each fading at its own speed."""
    freq = 440.0 * 2 ** ((note - 69) / 12)
    t = np.arange(int(rate * seconds)) / rate
    # Partial ratios of a free-free bar (what a chime tube is), with level and decay time.
    partials = ((1.0, 1.0, 3.0), (2.756, 0.45, 1.3), (5.404, 0.22, 0.55), (8.933, 0.1, 0.25))
    wave = sum(level * np.sin(2 * np.pi * freq * ratio * t) * np.exp(-t / decay)
               for ratio, level, decay in partials if freq * ratio < rate / 2)
    wave[: int(rate * 0.002)] *= np.linspace(0, 1, int(rate * 0.002))  # soften the click
    return (wave / np.abs(wave).max()).astype(np.float32)


# ----- effects for the built-in synth ----------------------------------------
# Each works on blocks no longer than its delay, which lets numpy process a
# whole block at once instead of looping sample by sample.

EFFECT_BLOCK = 256


class _Line:
    """Circular delay buffer."""

    def __init__(self, length):
        self.buf = np.zeros(length, np.float32)
        self.pos = 0

    def indices(self, n):
        return (self.pos + np.arange(n)) % len(self.buf)

    def advance(self, n):
        self.pos = (self.pos + n) % len(self.buf)


class Comb(_Line):
    """Feedback comb with a gentle low-pass in the loop (darker, smoother tail)."""

    def __init__(self, length, feedback):
        super().__init__(length)
        self.feedback = feedback
        self.prev = 0.0

    def process(self, x):
        idx = self.indices(len(x))
        out = self.buf[idx]
        damped = 0.6 * out + 0.4 * np.concatenate(([self.prev], out[:-1]))
        self.prev = out[-1]
        self.buf[idx] = x + damped * self.feedback
        self.advance(len(x))
        return out


class Allpass(_Line):
    def process(self, x):
        idx = self.indices(len(x))
        delayed = self.buf[idx]
        self.buf[idx] = x + delayed * 0.5
        self.advance(len(x))
        return delayed - x


class Reverb:
    """Freeverb-style: parallel combs into series allpasses, one set per ear."""

    COMBS = (1116, 1188, 1277, 1356, 1422, 1491, 1557, 1617)
    ALLPASSES = (556, 441, 341)

    def __init__(self, rate, feedback=0.86):
        scale = rate / 44100
        self.sides = []
        for spread in (0, 23):  # slightly different lengths left/right = wide stereo
            combs = [Comb(int((c + spread) * scale), feedback) for c in self.COMBS]
            alls = [Allpass(int((a + spread) * scale)) for a in self.ALLPASSES]
            self.sides.append((combs, alls))

    def process(self, x):
        out = []
        for combs, alls in self.sides:
            y = sum(c.process(x) for c in combs) * 0.12
            for a in alls:
                y = a.process(y)
            out.append(y)
        return np.stack(out, axis=1)


class StereoDelay:
    """Echoes at different times left and right, each repeat a little darker."""

    def __init__(self, rate, left=0.42, right=0.63, feedback=0.5):
        self.lines = [_Line(int(rate * left)), _Line(int(rate * right))]
        self.feedback = feedback
        self.prev = [0.0, 0.0]

    def process(self, x):
        out = np.empty((len(x), 2), np.float32)
        for ch, line in enumerate(self.lines):
            idx = line.indices(len(x))
            echo = line.buf[idx]
            dark = 0.5 * echo + 0.5 * np.concatenate(([self.prev[ch]], echo[:-1]))
            self.prev[ch] = echo[-1]
            line.buf[idx] = x + dark * self.feedback
            line.advance(len(x))
            out[:, ch] = dark
        return out


class BuiltinChimes:
    """Plays chimes straight out of the speakers, no other software needed."""

    def __init__(self):
        import sounddevice as sd
        self.rate = 48000
        self.samples = [render_chime(n, self.rate) for n in TUBES]
        self.voices = []  # [sample, position, left gain, right gain, effects send]
        self.lock = threading.Lock()
        self.reverb = Reverb(self.rate)
        self.delay = StereoDelay(self.rate)
        self.stream = sd.OutputStream(samplerate=self.rate, channels=2, blocksize=1024,
                                      latency="high", dtype="float32", callback=self._fill)
        self.stream.start()

    def strike(self, tube, velocity, depth, now):
        gain = 0.35 * (velocity / 127) ** 1.5
        pan = tube / (len(TUBES) - 1)  # low tubes left, high tubes right
        # Deeper chimes sit further back: a little quieter dry, much more in the effects.
        dry = 1.0 - 0.45 * depth
        with self.lock:
            self.voices.append([self.samples[tube], 0, gain * dry * (1.2 - pan * 0.7),
                                gain * dry * (0.5 + pan * 0.7), gain * depth])
            del self.voices[:-24]

    def set_depth(self, depth):
        pass  # depth is applied per chime in strike()

    def _fill(self, out, frames, _time, _status):
        dry = np.zeros((frames, 2), np.float32)
        send = np.zeros(frames, np.float32)
        with self.lock:
            for v in self.voices:
                sample, pos, left, right, fx = v
                chunk = sample[pos:pos + frames]
                dry[:len(chunk), 0] += chunk * left
                dry[:len(chunk), 1] += chunk * right
                send[:len(chunk)] += chunk * fx
                v[1] = pos + frames
            self.voices = [v for v in self.voices if v[1] < len(v[0])]

        wet = np.empty((frames, 2), np.float32)
        for start in range(0, frames, EFFECT_BLOCK):
            block = send[start:start + EFFECT_BLOCK]
            echoes = self.delay.process(block)
            # Echoes feed the reverb too, so repeats blur into the tail.
            verb = self.reverb.process(block + echoes.mean(axis=1) * 0.6)
            wet[start:start + EFFECT_BLOCK] = echoes * 0.3 + verb * 0.4
        out[:] = np.tanh(dry + wet)  # gentle limiter when many tubes ring at once

    def tick(self, now):
        pass

    def close(self):
        self.stream.stop()
        self.stream.close()


class SoundEngine:
    MODES = ("fl", "builtin", "off")
    LABELS = {"fl": "FL Studio", "builtin": "Built-in", "off": "Off"}
    OUTPUTS = {"fl": MidiOut, "builtin": BuiltinChimes}

    def __init__(self, mode):
        self.brain = ChimeBrain()
        self.output = None
        self.error = None
        self.mode = None
        self.space = 0.0  # smoothed overall depth, for the MIDI CCs
        self.set_mode(mode)

    @property
    def label(self):
        return self.LABELS[self.mode]

    @property
    def status(self):
        if self.error:
            return self.error
        return f"SOUND: {self.label.upper()}"

    def set_mode(self, mode):
        if self.output:
            self.output.close()
        self.output, self.error, self.mode = None, None, mode
        if mode == "off":
            return
        try:
            self.output = self.OUTPUTS[mode]()
        except Exception:  # missing port, no audio device, etc.
            self.error = "NO MIDI PORT" if mode == "fl" else "NO AUDIO OUTPUT"

    def cycle(self):
        self.set_mode(self.MODES[(self.MODES.index(self.mode) + 1) % len(self.MODES)])

    def update(self, tracks, frame_width):
        """Strike chimes for this frame; returns the tracks that rang one."""
        if not self.output:
            return []
        now = time.monotonic()
        self.output.tick(now)

        # Overall depth follows the longest-moving object, easing up and down.
        target = max((depth_for(t, now) for t in tracks), default=0.0)
        self.space += (target - self.space) * (0.05 if target > self.space else 0.02)
        self.output.set_depth(self.space)

        strikes = self.brain.update(tracks, frame_width, now)
        for tube, velocity, depth, _track in strikes:
            self.output.strike(tube, velocity, depth, now)
        return [s[3] for s in strikes]

    def close(self):
        if self.output:
            self.output.close()
