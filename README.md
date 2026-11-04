# Voice Changer

A system-wide voice changer for Windows. Your microphone goes through
**116 effect voices** modelled on the lists of Voicemod, Clownfish and
MorphVOX (Hacker, Banana, Minion, Demon, Magic Chords, T-Pain, Walkie-Talkie,
Ghost, Underwater, Cyborg, Gas Mask and many more) and/or an **AI voice**
(97 ready-made RVC voices: Trump, Obama, Morgan Freeman, Darth Vader,
SpongeBob, Goku, Miku, anime, games, or any RVC model you drop in), and
comes out of a **virtual microphone** that every app can use: Discord,
games, Zoom, Teams, OBS, Telegram, anything with a mic picker.

It is a Python app (tkinter GUI, numpy/scipy effects, PyTorch RVC engine)
and can be packed into a single `VoiceChanger.exe`.

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

### Effect voices

Pick a tab in **VOICES** and click a voice. Every voice is built from the
dials in **FINE TUNE**, so you can start from one and adjust it; the
button then shows *Custom*. All voices are level-matched to your own
loudness and add almost no delay (about 30 ms for pitch voices).

- **Popular**: Normal, Hacker, Anonymous, Banana, Minion, Chipmunk, Helium, Baby, Deep, Robot, Darth Lord, Auto-Tune, Magic Chords, Demon, Alien, Cave
- **Gender & age**: Man to Woman, Woman to Man, Child, Cute, Teen, Nerd, Tough Guy, Radio Announcer, Old Man, Old Lady, Santa, Elf, Giant, Ogre, Kong, Gnome
- **Funny**: Hamsters, Duck, Kazoo, Drunk, Crazy, Party Time, Mutation, Fast Mutation, Slow Mutation, Sleepyhead, Helicopter, Fan, Clone, 2x1, Wah-Wah, Vibrato, Speech Jammer, Underwater, Tin Can, Metal Pipe, Gibberish
- **Sci-fi**: Android, Bot, Cyborg, Police Bot, Martian, Spacemen, Drone, Exo, Orbital, Cybertune, Transformer, Gas Mask, VR, Glitch, Evil AI, Second Life, 8-bit, Atari, Game Over
- **Horror**: Hell Demon, Dark Angel, Possessed, Zombie, Living Dead, Ghost, Spirit, Lost Soul, Creepy Doll, Crazy Clown, Evil, Venom, Monster, Villain, Witch, Radio Demon, Fear, Franky, Upside Down, Whisper
- **Radio & places**: Telephone, Radio, Old Radio, Megaphone, Cop Radio, Walkie-Talkie, Out of Range, Team Radio, Speakerphone, Cathedral, Student Hall, Stadium, Bathroom, Canyon
- **Music**: T-Pain, Voicetune, Cool Tune, Harmony, Octaver, Choir, Trap Star, Lil' Mod, Barbershop, Gospel

The dials: pitch (PSOLA, keeps your timbre), formant (vocal-tract size,
for gender and age), auto-tune with glide, two harmony voices, whisper,
hiss, robot ring modulation, metal comb, drive, bit crush, lo-fi sample
rate, low and high cut, wobble, pitch swing and random jumps, chorus,
flanger, tremolo, wah, echo, reverb and connection cut-outs.

### AI voices

1. In **AI VOICE** pick a voice from the list and click **Use voice**.
   The first time this downloads the shared speech model (about 380 MB)
   and then the voice (50 to 60 MB). Voices are kept in
   `%USERPROFILE%\.voicechanger\voices`.
2. Tick **AI voice on**. Set **Pitch key** to match your own voice to the
   target: a male speaker using a female voice needs about +12, a female
   speaker using a male voice about -12. Same type of voice: 0.
3. **Latency** is the AI block size. Pick the smallest value where the
   status line does not report gaps. With an NVIDIA GPU 0.25 s is fine;
   on a CPU use 0.35 to 0.5 s.
4. **Download index** also fetches the voice's retrieval index (often
   100 to 600 MB). It makes the timbre match the target more closely.
5. **Add .pth file...** loads any RVC v1/v2 model you downloaded elsewhere
   (weights.gg, voice-models.com, Hugging Face). If a `.index` file sits
   in the same folder it is picked up too.

The AI voice runs before the effects, so you can stack Robot or Cave on
top of an AI voice.

### GPU

With an NVIDIA card run **`install_gpu.bat`** once. It swaps in the CUDA
build of PyTorch; the app then shows *Runs on CUDA* and converts each
block in well under 100 ms. Without a GPU everything runs on the CPU,
which works on a modern desktop CPU at 0.35 to 0.5 s latency.

Total latency with effects only is about 20 to 40 ms (5 ms per audio block
plus the cable), which is unnoticeable in voice chat. With an AI voice it
is the chosen Latency value plus the model time, typically 0.4 to 0.8 s on
a CPU and about 0.3 s on a GPU.

## Build a single .exe

Run **`build_windows.bat`** once. It produces `dist\VoiceChanger.exe`, a
standalone file you can copy anywhere, no Python needed on that PC.
(VB-CABLE still has to be installed on the PC that runs it.) With the AI
engine bundled the file is large (about 1 GB).

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
- **AI voice sounds wrong / wrong gender**: adjust *Pitch key* (+12 or
  -12 when switching gender). Speak clearly and close to the mic; RVC
  copies breath and room noise too.
- **Status shows gaps**: the PC is too slow for the chosen Latency. Pick a
  larger one, close other apps, or install the GPU build.
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
- `dsp.py`: the effects engine (PSOLA pitch, YIN pitch tracker, STFT formant
  and whisper, modulation, lo-fi, space) and all 116 voice presets.
- `ai_voice.py`: the real-time RVC engine (ContentVec + FCPE pitch +
  RVC synthesizer, SOLA block stitching) and the voice library.
- `voices.json`: the voice catalog (Hugging Face repo and file per voice).
- `rvc/`: the RVC synthesizer network, vendored from RVC-Project (MIT).
- `install_gpu.bat`: installs the CUDA build of PyTorch.
- `run_windows.bat`: first-time setup and launch.
- `build_windows.bat`: makes `dist\VoiceChanger.exe`.
