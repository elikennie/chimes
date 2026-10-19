# motion-chimes

Camera motion → wind chime sounds (via MIDI into FL Studio).

`tracker.py` boxes and tracks moving objects in the camera frame; `chimes.py`
turns that motion into wind chime notes, sent to FL Studio over MIDI or played
by a built-in chime synth.

## Setup

Needs Python 3.10+ (python.org). Run these from this folder.

### macOS
```
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python tracker.py
```
First run: allow camera access for your terminal app
(System Settings → Privacy & Security → Camera), then run again.

### Windows
```
py -m venv .venv
.venv\Scripts\pip install -r requirements.txt
.venv\Scripts\python tracker.py
```
If the camera doesn't open: Settings → Privacy & security → Camera →
allow desktop apps to access your camera.

## Using it

- `--camera 1` picks a different camera, `--video clip.mp4` uses a recording.
- Keep the camera **still**. It learns the background for a few seconds, so
  don't move in front of it right at the start.
- **Min size** (px): smallest patch of movement counted, as the side of a square.
  Raise it to ignore flicker (leaves, noise), lower it to catch small things.
- **Max jump** (px): how far an object can move between frames and keep its number.
  Raise it for fast things like cars.
- Drag the sliders, or click `–` / `+` for exact steps (1 px / 5 px).
- The Options list (left) switches camera / motion mask, full screen, and resets the background.
- Leave the mouse still for 5 seconds and the panels slide away while the camera
  grows to fill the window;
  move it to bring them back.
- Keys: `f` full screen (camera and boxes only; `Esc` leaves it), `m` motion mask,
  `r` reset the background, `q` quit.

## Sound

Each moving object can knock the chime "tube" under its **hot spot** (the dot
on its box): the part of it that's actually moving, found from frame-to-frame
pixel changes. So a waving hand plays its own note even when the whole body is
one box. The frame is split
into 8 vertical strips, low notes on the left and high on the right (C major
pentatonic). Faster movement rings more often and harder; a box glows when it
rings a chime. **Chime rate** sets how busy it is overall.

The longer an object keeps moving, the more delay and reverb its chimes get
(the coloured bar on its tag). New movement starts fairly dry and reaches full
depth after about 6 seconds (`DEPTH_SECONDS` in `chimes.py`).

On the camera, each box's tag shows its number, the note it's over, its speed
and how long it's been moving. **Grid** (or `g`) shows the 8 chime strips with
their notes; a strip glows when its chime rings. **Labels** (or `l`) hides the tags.

Click **Sound** in Options (or press `s`) to switch FL Studio / Built-in / Off,
or start with `--sound builtin` or `--sound off`.

### FL Studio

FL Studio must be installed (the free trial works for live playing).

1. **Make the MIDI port**
   - macOS: nothing to do. The app creates a port called **Chimes** while it's running.
   - Windows: install [loopMIDI](https://www.tobias-erichsen.de/software/loopmidi.html)
     (free), and add a port named **Chimes** (keep loopMIDI running).
2. Start `tracker.py` **first**, then open FL Studio.
3. In FL Studio: **Options → MIDI Settings**. In the Input list, select
   **Chimes** and click **Enable**.
4. Pick a chime sound on the selected channel: drag a wind chime sample onto
   a Sampler channel, or load a bell preset in a synth (e.g. Sytrus or FLEX).
   Notes play on whichever channel is selected.
5. Add a reverb (e.g. Fruity Reeverb 2) on that channel's mixer track. Long
   tails are what make it sound like real chimes.
6. **Depth (delay + reverb):** the app sends MIDI **CC 91** and **CC 92** (0–127),
   rising the longer things keep moving. Link them once: right-click the
   reverb's **Wet** knob → **Link to controller…**, then set the **Ctrl** number to
   91 (port: Chimes). Do the same for your delay's **Wet** or **Feedback** knob
   with 92.
7. To record, press record in FL Studio and let the camera play.

If the header says **NO MIDI PORT**, the Chimes port wasn't found (on Windows,
check loopMIDI is running). Fix that, then click Sound until it comes back
around to FL Studio to try again.
