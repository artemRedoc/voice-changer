# Voice Changer

A system-wide voice changer for Windows. Your microphone goes through
real-time effects (pitch, robot, echo, reverb, telephone, megaphone, 8-bit
and more) and comes out of a **virtual microphone** that every app can use:
Discord, games, Zoom, Teams, OBS, Telegram, anything with a mic picker.

It is a small Python app (numpy + scipy + sounddevice, tkinter GUI) and can
be packed into a single `VoiceChanger.exe`.

## How it works

```
your mic  ->  Voice Changer (effects)  ->  "CABLE Input"   (virtual cable, in)
                                              |
Discord / game / Zoom  <-  microphone = "CABLE Output" (virtual cable, out)
```

Windows has no built-in virtual microphone, so the app uses the free
**VB-CABLE** driver as the pipe between itself and other apps.

## Setup on Windows

1. **Install VB-CABLE** (free): https://vb-audio.com/Cable/
   Unzip, right-click `VBCABLE_Setup_x64.exe`, *Run as administrator*,
   click *Install Driver*, then **reboot**.
2. **Install Python 3.11 or newer**: https://www.python.org/downloads/
   In the installer tick **"Add python.exe to PATH"**.
3. Copy this folder to the PC and double-click **`run_windows.bat`**.
   The first run creates a `.venv` folder and installs the dependencies
   (about a minute). After that it opens straight away.

## Using it

1. **Microphone**: your real mic.
2. **Send to (virtual mic)**: `CABLE Input (VB-Audio Virtual Cable)`.
   The app picks it automatically when it is installed.
3. **Hear myself on**: your headphones, if you want to hear the result.
   Use headphones, not speakers, or the mic will pick the output up again.
4. Click **Start**, then pick a voice. Move the dials to tune it. Everything
   changes live.
5. In Discord / the game / Zoom choose
   **`CABLE Output (VB-Audio Virtual Cable)`** as the microphone.

**EFFECTS ON/OFF** switches between the changed voice and your normal voice
without stopping the stream. Your last devices and dials are remembered.

Total latency is about 20 to 40 ms (5 ms per audio block plus the cable),
which is unnoticeable in voice chat.

## Build a single .exe

Run **`build_windows.bat`** once. It produces `dist\VoiceChanger.exe`, a
standalone file you can copy anywhere, no Python needed on that PC.
(VB-CABLE still has to be installed on the PC that runs it.)

## Troubleshooting

- **"Could not open audio devices"**: set both the mic and the cable to the
  same format. Windows *Sound settings → More sound settings*, select
  `CABLE Input` → *Properties → Advanced* → `16 bit, 48000 Hz`. Do the same
  for `CABLE Output` on the Recording tab and for your mic.
- **Other apps hear nothing**: make sure the app shows *Running* and
  **Send to** is `CABLE Input`, and that the other app's mic is
  `CABLE Output` (not `CABLE Input`).
- **Crackling**: close other audio apps, or in Windows Sound settings turn
  off *Allow applications to take exclusive control* for the cable.
- **Echo in calls**: you are monitoring on speakers. Set *Hear myself on*
  to headphones or `(off)`.
- **Robotic artifacts at big pitch shifts**: that is the price of a
  low-latency shifter. Keep pitch within ±7 semitones for natural results;
  the extreme settings are meant to sound monstrous.

## Run on macOS (for development)

```
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python app.py
```

Install BlackHole (https://existential.audio/blackhole/) as the virtual
cable; the app detects it the same way it detects VB-CABLE.

## Files

- `app.py`: GUI, device handling, audio streams.
- `dsp.py`: the effects engine and the voice presets.
- `run_windows.bat`: first-time setup and launch.
- `build_windows.bat`: makes `dist\VoiceChanger.exe`.
