"""Real-time voice effects engine.

Everything works on mono float32 blocks and keeps its own state between
blocks, so it can be driven from a low-latency audio callback.
"""

from dataclasses import dataclass, asdict, replace
import numpy as np
from scipy.signal import butter, sosfilt, sosfilt_zi, lfilter


@dataclass
class Params:
    enabled: bool = True
    # voice
    pitch: float = 0.0        # semitones -12..12
    formant: float = 1.0      # 0.6..1.6  vocal tract size (x<1 bigger, x>1 smaller)
    tune: float = 0.0         # 0..1 auto-tune amount
    tune_speed: float = 0.2   # 0 hard (T-Pain) .. 1 slow and natural
    harm1: float = 0.0        # semitones of extra voice 1 (0 = off)
    harm2: float = 0.0        # semitones of extra voice 2 (0 = off)
    harm_mix: float = 0.0     # 0..1
    whisper: float = 0.0      # 0..1 breathy / whisper
    # character
    ring_mix: float = 0.0     # robot amount
    ring_freq: float = 60.0
    comb_mix: float = 0.0     # metallic resonance
    comb_freq: float = 200.0
    comb_fb: float = 0.7
    drive: float = 0.0
    crush: float = 0.0        # bit depth reduction
    decimate: float = 1.0     # sample-and-hold factor 1..24
    noise: float = 0.0        # hiss that follows the voice
    # movement
    vib_depth: float = 0.0
    vib_rate: float = 5.0
    pitch_lfo: float = 0.0    # semitones of slow pitch swing
    pitch_rnd: float = 0.0    # semitones of random pitch jumps
    pitch_lfo_rate: float = 1.0
    chorus_mix: float = 0.0
    chorus_rate: float = 1.2
    flange_mix: float = 0.0
    flange_rate: float = 0.3
    trem_depth: float = 0.0
    trem_rate: float = 6.0
    wah_mix: float = 0.0
    wah_rate: float = 2.0
    # space and tone
    echo_mix: float = 0.0
    echo_time: float = 0.3
    echo_fb: float = 0.3
    reverb_mix: float = 0.0
    reverb_size: float = 1.5
    low_cut: float = 20.0
    high_cut: float = 20000.0
    dropout: float = 0.0      # 0..1 bad connection
    volume: float = 1.0

    def to_dict(self):
        return asdict(self)

    @staticmethod
    def from_dict(d):
        base = Params()
        known = {k: v for k, v in d.items() if k in base.__dict__}
        return replace(base, **known)


# ---------------------------------------------------------------- voices
# Inspired by the voice lists of Voicemod, Clownfish and MorphVOX.

CATEGORIES = {
    "Popular": {
        "Normal": {},
        "Hacker": {"pitch": -6, "formant": 0.85, "ring_mix": 0.35, "ring_freq": 35, "crush": 0.25,
                   "decimate": 2, "low_cut": 120, "high_cut": 6000, "flange_mix": 0.25, "flange_rate": 0.2},
        "Anonymous": {"pitch": -4, "formant": 0.88, "chorus_mix": 0.5, "chorus_rate": 0.8, "drive": 0.15,
                      "ring_mix": 0.2, "ring_freq": 30, "reverb_mix": 0.15},
        "Banana": {"pitch": 9, "formant": 1.45, "tune": 0.7, "tune_speed": 0.1, "vib_depth": 0.3, "vib_rate": 6.5},
        "Minion": {"pitch": 10, "formant": 1.5, "pitch_lfo": 1.5, "pitch_lfo_rate": 2.5, "chorus_mix": 0.2},
        "Chipmunk": {"pitch": 7, "formant": 1.25},
        "Helium": {"pitch": 10, "formant": 1.4},
        "Baby": {"pitch": 9, "formant": 1.45, "high_cut": 7000},
        "Deep": {"pitch": -5, "formant": 0.9},
        "Robot": {"ring_mix": 0.9, "ring_freq": 60, "high_cut": 5000},
        "Darth Lord": {"pitch": -5, "formant": 0.8, "low_cut": 60, "high_cut": 4500, "comb_mix": 0.3,
                       "comb_freq": 70, "comb_fb": 0.6, "reverb_mix": 0.15, "noise": 0.15},
        "Auto-Tune": {"tune": 1.0, "tune_speed": 0.0},
        "Magic Chords": {"tune": 1.0, "tune_speed": 0.05, "harm1": 4, "harm2": 7, "harm_mix": 0.7,
                         "reverb_mix": 0.3},
        "Demon": {"pitch": -9, "formant": 0.7, "drive": 0.35, "harm1": -12, "harm_mix": 0.5,
                  "reverb_mix": 0.3},
        "Alien": {"pitch": 4, "ring_mix": 0.5, "ring_freq": 150, "vib_depth": 0.5, "vib_rate": 5},
        "Cave": {"reverb_mix": 0.7, "reverb_size": 2.5, "echo_mix": 0.3, "echo_time": 0.4, "echo_fb": 0.4},
    },
    "Gender & age": {
        "Man to Woman": {"pitch": 5, "formant": 1.18},
        "Woman to Man": {"pitch": -5, "formant": 0.85},
        "Child": {"pitch": 5, "formant": 1.25, "high_cut": 9000},
        "Cute": {"pitch": 5, "formant": 1.2, "tune": 0.3, "tune_speed": 0.4},
        "Teen": {"pitch": 2, "formant": 1.07},
        "Nerd": {"pitch": 2, "formant": 1.1, "low_cut": 350, "high_cut": 7000},
        "Tough Guy": {"pitch": -3, "formant": 0.92, "drive": 0.15},
        "Radio Announcer": {"pitch": -1, "formant": 0.96, "low_cut": 70, "drive": 0.1,
                            "reverb_mix": 0.1, "reverb_size": 0.8},
        "Old Man": {"pitch": -2, "formant": 0.95, "vib_depth": 0.3, "vib_rate": 6, "high_cut": 6000,
                    "whisper": 0.15},
        "Old Lady": {"pitch": 3, "formant": 1.1, "vib_depth": 0.35, "vib_rate": 6.5, "whisper": 0.1},
        "Santa": {"pitch": -3, "formant": 0.9, "reverb_mix": 0.15},
        "Elf": {"pitch": 6, "formant": 1.15, "chorus_mix": 0.2},
        "Giant": {"pitch": -12, "formant": 0.7, "reverb_mix": 0.25, "reverb_size": 2},
        "Ogre": {"pitch": -8, "formant": 0.7, "drive": 0.2},
        "Kong": {"pitch": -10, "formant": 0.65, "drive": 0.25, "low_cut": 40},
        "Gnome": {"pitch": 8, "formant": 1.35, "drive": 0.1},
    },
    "Funny": {
        "Hamsters": {"pitch": 12, "formant": 1.5, "chorus_mix": 0.6, "chorus_rate": 3},
        "Duck": {"pitch": 4, "formant": 1.3, "low_cut": 700, "high_cut": 3500, "drive": 0.4},
        "Kazoo": {"drive": 0.7, "low_cut": 500, "high_cut": 3500, "ring_mix": 0.2, "ring_freq": 200,
                  "tune": 0.6, "tune_speed": 0.1},
        "Drunk": {"pitch_lfo": 1.5, "pitch_rnd": 0.8, "pitch_lfo_rate": 0.7, "vib_depth": 0.3,
                  "vib_rate": 1.2, "echo_mix": 0.15, "echo_time": 0.15},
        "Crazy": {"pitch_rnd": 4, "pitch_lfo_rate": 4},
        "Party Time": {"pitch_lfo": 6, "pitch_lfo_rate": 0.5, "reverb_mix": 0.3},
        "Mutation": {"pitch_rnd": 6, "pitch_lfo_rate": 1.5},
        "Fast Mutation": {"pitch_rnd": 7, "pitch_lfo_rate": 6},
        "Slow Mutation": {"pitch_lfo": 5, "pitch_lfo_rate": 0.3},
        "Sleepyhead": {"pitch": -3, "pitch_lfo": 1.0, "pitch_lfo_rate": 0.3, "high_cut": 4000},
        "Helicopter": {"trem_depth": 0.95, "trem_rate": 14, "noise": 0.15},
        "Fan": {"trem_depth": 0.7, "trem_rate": 22, "noise": 0.2, "ring_mix": 0.1, "ring_freq": 22},
        "Clone": {"chorus_mix": 0.9, "chorus_rate": 1.2},
        "2x1": {"harm1": 0.15, "harm_mix": 0.8, "chorus_mix": 0.4},
        "Wah-Wah": {"wah_mix": 0.9, "wah_rate": 2},
        "Vibrato": {"vib_depth": 0.8, "vib_rate": 6},
        "Speech Jammer": {"echo_mix": 1.0, "echo_time": 0.2, "echo_fb": 0.0},
        "Underwater": {"pitch": -1, "high_cut": 900, "vib_depth": 0.6, "vib_rate": 1.5,
                       "flange_mix": 0.5, "flange_rate": 0.3, "reverb_mix": 0.3},
        "Tin Can": {"comb_mix": 0.7, "comb_freq": 600, "comb_fb": 0.8, "low_cut": 400, "high_cut": 5000},
        "Metal Pipe": {"comb_mix": 0.8, "comb_freq": 220, "comb_fb": 0.9, "reverb_mix": 0.2},
        "Gibberish": {"pitch_rnd": 3, "pitch_lfo_rate": 9, "trem_depth": 0.4, "trem_rate": 9},
    },
    "Sci-fi": {
        "Android": {"tune": 1.0, "tune_speed": 0.0, "ring_mix": 0.3, "ring_freq": 90, "comb_mix": 0.4,
                    "comb_freq": 150, "comb_fb": 0.6},
        "Bot": {"ring_mix": 0.6, "ring_freq": 100, "crush": 0.3},
        "Cyborg": {"pitch": -3, "comb_mix": 0.6, "comb_freq": 110, "comb_fb": 0.7, "ring_mix": 0.3,
                   "ring_freq": 50, "flange_mix": 0.3},
        "Police Bot": {"ring_mix": 0.7, "ring_freq": 80, "low_cut": 400, "high_cut": 3200, "drive": 0.3},
        "Martian": {"pitch": 3, "ring_mix": 0.8, "ring_freq": 330, "vib_depth": 0.4, "vib_rate": 9},
        "Spacemen": {"low_cut": 300, "high_cut": 3000, "noise": 0.2, "drive": 0.2, "comb_mix": 0.3,
                     "comb_freq": 900, "comb_fb": 0.5},
        "Drone": {"pitch": -4, "flange_mix": 0.8, "flange_rate": 0.15, "ring_mix": 0.4, "ring_freq": 50},
        "Exo": {"pitch": -5, "flange_mix": 0.6, "flange_rate": 0.5, "drive": 0.2, "comb_mix": 0.4,
                "comb_freq": 90, "comb_fb": 0.6},
        "Orbital": {"flange_mix": 0.9, "flange_rate": 0.1, "chorus_mix": 0.5, "reverb_mix": 0.4,
                    "reverb_size": 2},
        "Cybertune": {"tune": 1.0, "tune_speed": 0.0, "harm1": 12, "harm_mix": 0.3, "flange_mix": 0.3},
        "Transformer": {"pitch": -4, "formant": 0.85, "comb_mix": 0.5, "comb_freq": 140, "comb_fb": 0.7,
                        "flange_mix": 0.4, "drive": 0.3},
        "Gas Mask": {"low_cut": 500, "high_cut": 3000, "comb_mix": 0.5, "comb_freq": 450, "comb_fb": 0.6,
                     "noise": 0.15, "drive": 0.2},
        "VR": {"flange_mix": 0.5, "flange_rate": 2, "reverb_mix": 0.2, "echo_mix": 0.2, "echo_time": 0.08},
        "Glitch": {"dropout": 0.5, "decimate": 4, "pitch_rnd": 3, "pitch_lfo_rate": 8, "crush": 0.4},
        "Evil AI": {"pitch": -2, "ring_mix": 0.3, "ring_freq": 80, "dropout": 0.25, "crush": 0.3,
                    "tune": 1.0, "tune_speed": 0.0},
        "Second Life": {"pitch": 1, "chorus_mix": 0.5, "reverb_mix": 0.3, "high_cut": 8000},
        "8-bit": {"decimate": 8, "crush": 0.85, "tune": 1.0, "tune_speed": 0.0},
        "Atari": {"decimate": 12, "crush": 0.9, "ring_mix": 0.3, "ring_freq": 440},
        "Game Over": {"pitch": -12, "decimate": 6, "crush": 0.7, "reverb_mix": 0.3},
    },
    "Horror": {
        "Hell Demon": {"pitch": -7, "harm1": -12, "harm_mix": 0.7, "ring_mix": 0.2, "ring_freq": 30,
                       "drive": 0.3, "reverb_mix": 0.35},
        "Dark Angel": {"pitch": -4, "harm1": 12, "harm_mix": 0.4, "reverb_mix": 0.6, "reverb_size": 2.5,
                       "chorus_mix": 0.4},
        "Possessed": {"pitch": -8, "harm1": -12, "harm2": 7, "harm_mix": 0.6, "pitch_rnd": 1,
                      "pitch_lfo_rate": 2, "drive": 0.3, "reverb_mix": 0.3},
        "Zombie": {"pitch": -5, "formant": 0.85, "drive": 0.45, "whisper": 0.3, "low_cut": 120,
                   "trem_depth": 0.3, "trem_rate": 5},
        "Living Dead": {"pitch": -6, "formant": 0.8, "drive": 0.5, "comb_mix": 0.3, "comb_freq": 80,
                        "whisper": 0.25, "reverb_mix": 0.2},
        "Ghost": {"pitch": -2, "whisper": 0.5, "reverb_mix": 0.8, "reverb_size": 3, "vib_depth": 0.3,
                  "vib_rate": 3, "echo_mix": 0.4, "echo_time": 0.5, "echo_fb": 0.5},
        "Spirit": {"pitch": 2, "whisper": 0.7, "reverb_mix": 0.7, "reverb_size": 3, "flange_mix": 0.3},
        "Lost Soul": {"pitch": -1, "whisper": 0.5, "echo_mix": 0.5, "echo_time": 0.6, "echo_fb": 0.6,
                      "reverb_mix": 0.6, "vib_depth": 0.4, "vib_rate": 2},
        "Creepy Doll": {"pitch": 6, "formant": 1.35, "vib_depth": 0.5, "vib_rate": 4, "decimate": 2,
                        "crush": 0.3, "reverb_mix": 0.3},
        "Crazy Clown": {"pitch": 4, "formant": 1.2, "pitch_lfo": 2, "pitch_lfo_rate": 3,
                        "chorus_mix": 0.3, "reverb_mix": 0.2},
        "Evil": {"pitch": -6, "harm1": -12, "harm_mix": 0.4, "drive": 0.25, "reverb_mix": 0.25},
        "Venom": {"pitch": -10, "formant": 0.6, "drive": 0.5, "harm1": -12, "harm_mix": 0.5,
                  "whisper": 0.2},
        "Monster": {"pitch": -9, "formant": 0.75, "drive": 0.3, "reverb_mix": 0.2},
        "Villain": {"pitch": -7, "formant": 0.85, "low_cut": 100, "high_cut": 5000, "drive": 0.25,
                    "ring_mix": 0.25, "ring_freq": 40},
        "Witch": {"pitch": 4, "formant": 1.1, "vib_depth": 0.4, "vib_rate": 7, "drive": 0.2,
                  "reverb_mix": 0.3},
        "Radio Demon": {"pitch": -3, "low_cut": 300, "high_cut": 3500, "drive": 0.4, "noise": 0.3,
                        "harm1": -12, "harm_mix": 0.3, "tune": 0.5, "tune_speed": 0.1},
        "Fear": {"trem_depth": 0.6, "trem_rate": 9, "vib_depth": 0.4, "vib_rate": 9, "reverb_mix": 0.3},
        "Franky": {"pitch": -6, "formant": 0.8, "drive": 0.3, "reverb_mix": 0.2},
        "Upside Down": {"pitch": -3, "flange_mix": 0.5, "flange_rate": 0.2, "reverb_mix": 0.5,
                        "reverb_size": 2.5, "high_cut": 3000},
        "Whisper": {"whisper": 1.0, "low_cut": 300},
    },
    "Radio & places": {
        "Telephone": {"low_cut": 400, "high_cut": 3200, "drive": 0.2},
        "Radio": {"low_cut": 300, "high_cut": 4500, "drive": 0.35, "crush": 0.2},
        "Old Radio": {"low_cut": 300, "high_cut": 3500, "noise": 0.25, "crush": 0.3, "drive": 0.25,
                      "trem_depth": 0.1, "trem_rate": 4},
        "Megaphone": {"low_cut": 500, "high_cut": 4000, "drive": 0.6},
        "Cop Radio": {"low_cut": 400, "high_cut": 3000, "drive": 0.5, "noise": 0.3, "crush": 0.2},
        "Walkie-Talkie": {"low_cut": 500, "high_cut": 2800, "drive": 0.6, "noise": 0.4, "decimate": 2},
        "Out of Range": {"low_cut": 400, "high_cut": 3200, "drive": 0.3, "noise": 0.35, "dropout": 0.6},
        "Team Radio": {"low_cut": 350, "high_cut": 3800, "drive": 0.45, "crush": 0.25, "noise": 0.1},
        "Speakerphone": {"low_cut": 250, "high_cut": 6000, "reverb_mix": 0.25, "reverb_size": 0.7,
                         "drive": 0.1},
        "Cathedral": {"reverb_mix": 0.75, "reverb_size": 3, "low_cut": 100},
        "Student Hall": {"reverb_mix": 0.45, "reverb_size": 1.8},
        "Stadium": {"echo_mix": 0.3, "echo_time": 0.35, "echo_fb": 0.3, "reverb_mix": 0.5,
                    "reverb_size": 2.5},
        "Bathroom": {"reverb_mix": 0.5, "reverb_size": 0.7, "high_cut": 9000},
        "Canyon": {"echo_mix": 0.5, "echo_time": 0.7, "echo_fb": 0.5, "reverb_mix": 0.2},
    },
    "Music": {
        "T-Pain": {"tune": 1.0, "tune_speed": 0.0, "reverb_mix": 0.15},
        "Voicetune": {"tune": 0.8, "tune_speed": 0.3},
        "Cool Tune": {"tune": 1.0, "tune_speed": 0.1, "chorus_mix": 0.3, "reverb_mix": 0.2},
        "Harmony": {"tune": 0.6, "tune_speed": 0.2, "harm1": 4, "harm_mix": 0.6},
        "Octaver": {"harm1": -12, "harm_mix": 0.9},
        "Choir": {"harm1": -12, "harm2": 12, "harm_mix": 0.45, "chorus_mix": 0.7, "reverb_mix": 0.5,
                  "reverb_size": 2.2},
        "Trap Star": {"tune": 1.0, "tune_speed": 0.0, "pitch": -2, "drive": 0.1, "reverb_mix": 0.2,
                      "echo_mix": 0.25, "echo_time": 0.25},
        "Lil' Mod": {"pitch": 3, "tune": 1.0, "tune_speed": 0.0, "echo_mix": 0.2, "echo_time": 0.18},
        "Barbershop": {"tune": 0.8, "tune_speed": 0.15, "harm1": 3, "harm2": 8, "harm_mix": 0.5},
        "Gospel": {"tune": 0.7, "harm1": 5, "harm2": 9, "harm_mix": 0.5, "reverb_mix": 0.5},
    },
}

PRESETS = {name: p for cat in CATEGORIES.values() for name, p in cat.items()}


def preset_params(name):
    return Params.from_dict(PRESETS.get(name, {}))


# ---------------------------------------------------------------- building blocks

class DelayLine:
    """Circular buffer. read_int() looks back from the *current* write position,
    so call it before write() for a feedback topology."""

    def __init__(self, size):
        self.size = 1 << (int(size) - 1).bit_length()
        self.mask = self.size - 1
        self.buf = np.zeros(self.size, dtype=np.float32)
        self.pos = 0

    def write(self, x):
        n = len(x)
        idx = (self.pos + np.arange(n)) & self.mask
        self.buf[idx] = x
        self.pos = (self.pos + n) & self.mask

    def read_int(self, n, delay):
        idx = (self.pos + np.arange(n) - delay) & self.mask
        return self.buf[idx]

    def read_frac(self, n, delay):
        """Fractional delay per sample, relative to the last n written samples."""
        base = self.pos - n + np.arange(n)
        rp = base - delay
        i0 = np.floor(rp).astype(np.int64)
        frac = (rp - i0).astype(np.float32)
        a = self.buf[i0 & self.mask]
        b = self.buf[(i0 + 1) & self.mask]
        return a + (b - a) * frac


class LFO:
    def __init__(self, sr):
        self.sr = sr
        self.phase = 0.0

    def sine(self, n, rate):
        step = 2 * np.pi * rate / self.sr
        ph = self.phase + np.arange(n) * step
        self.phase = (self.phase + n * step) % (2 * np.pi)
        return np.sin(ph).astype(np.float32)

    def value(self, n, rate):
        """One value per block (block-rate modulation)."""
        v = np.sin(self.phase)
        self.phase = (self.phase + 2 * np.pi * rate * n / self.sr) % (2 * np.pi)
        return float(v)


class Biquad:
    def __init__(self, sr, kind):
        self.sr = sr
        self.kind = kind
        self.fc = None
        self.sos = None
        self.zi = None

    def set(self, fc):
        fc = float(np.clip(fc, 10.0, self.sr * 0.45))
        if fc == self.fc:
            return
        self.fc = fc
        self.sos = butter(2, fc, btype=self.kind, fs=self.sr, output="sos")
        if self.zi is None:
            self.zi = sosfilt_zi(self.sos) * 0.0

    def process(self, x):
        y, self.zi = sosfilt(self.sos, x, zi=self.zi)
        return y.astype(np.float32)


class PitchShifter:
    """Two overlapping grains read from a delay line at a different speed,
    Hann cross-faded so the sum stays flat.

    When the voice's pitch period is known, the grain length is chosen so
    each splice jumps by a whole number of periods. The splice is then
    phase-coherent, which keeps small shifts (auto-tune) accurate and makes
    large shifts much less rough."""

    def __init__(self, sr, window_s=0.06):
        self.sr = sr
        self.win0 = window_s * sr
        self.win = self.win0
        self.line = DelayLine(int(sr * 0.6) + 4096)
        self.phase = 0.0

    def process(self, x, semitones, period=None):
        n = len(x)
        self.line.write(x)
        ratio = 2.0 ** (semitones / 12.0)
        dr = abs(ratio - 1.0)
        if period and dr > 1e-4:
            # the two grains are half a grain apart, so the jump D must be an
            # even number of periods for both to stay in phase
            k = max(1, round(dr * self.win0 / (2 * period)))
            win = 2 * k * period / dr
            self.win = float(np.clip(win, 0.02 * self.sr, 2.0 * self.sr))
        elif not period:
            self.win = self.win0
        win = self.win
        D = dr * win
        p1 = (self.phase + np.arange(n) / win) % 1.0
        p2 = (p1 + 0.5) % 1.0
        self.phase = (self.phase + n / win) % 1.0
        if ratio < 1.0:
            d1, d2 = D * p1, D * p2
        else:
            d1, d2 = D * (1.0 - p1), D * (1.0 - p2)
        g1 = 0.5 * (1.0 - np.cos(2 * np.pi * p1))
        g2 = 0.5 * (1.0 - np.cos(2 * np.pi * p2))
        r1 = self.line.read_frac(n, d1)
        r2 = self.line.read_frac(n, d2)
        return (g1 * r1 + g2 * r2).astype(np.float32)


class PSOLA:
    """Real-time TD-PSOLA pitch shifter.

    One Hann grain two periods long is cut around each analysis mark
    (marks spaced one pitch period apart) and laid down at synthesis marks
    spaced period / ratio apart. Formants stay where they are, so pitch and
    vocal-tract size are controlled independently. Pitch comes from the
    engine's tracker history, so each grain uses the period of its own
    moment in time. Latency is about 30 ms.
    """

    SIZE = 1 << 16

    def __init__(self, sr, history):
        self.sr = sr
        self.hist = history              # engine-owned list of (time, period or 0)
        self.pmax = int(sr / 60)
        self.L = 2 * self.pmax + 256
        self.default_p = int(sr * 0.006)
        self.inb = np.zeros(self.SIZE, dtype=np.float32)
        self.acc = np.zeros(self.SIZE, dtype=np.float32)
        self.mask = self.SIZE - 1
        self.t_in = 0
        self.syn = float(self.L)
        self.ana = float(self.L)
        self.corr = 0.0
        self.windows = {}

    def _period_at(self, t):
        h = self.hist
        for i in range(len(h) - 1, -1, -1):
            if h[i][0] <= t:
                return h[i][1]
        return h[0][1] if h else 0

    def _win(self, P):
        w = self.windows.get(P)
        if w is None:
            w = np.hanning(2 * P + 1).astype(np.float32)
            self.windows[P] = w
        return w

    def process(self, x, semis, tune=0.0, tune_speed=0.0):
        n = len(x)
        m = self.mask
        idx = (self.t_in + np.arange(n)) & m
        self.inb[idx] = x
        self.t_in += n
        e_end = self.t_in - self.L
        alpha_base = None
        while self.syn < e_end + self.pmax:
            ts = self.syn
            P0 = self._period_at(ts)
            P = int(round(P0)) if P0 else self.default_p
            P = int(np.clip(P, 32, self.pmax))
            r_semis = semis
            if tune > 0.0:
                if P0:
                    midi = 69.0 + 12.0 * np.log2((self.sr / P0) / 440.0)
                    target = np.round(midi) - midi
                    tau = tune_speed * 0.15 * self.sr
                    a = 1.0 if tau < 1.0 else 1.0 - np.exp(-P / tau)
                    self.corr += a * (target - self.corr)
                r_semis = semis + tune * self.corr
            ratio = 2.0 ** (r_semis / 12.0)
            while self.ana + P <= ts + P / 2:
                self.ana += P
            if self.ana < ts - 2 * P:
                self.ana = ts
            ta = int(round(self.ana))
            tsi = int(round(ts))
            w = self._win(P)
            seg = self.inb[(ta - P + np.arange(2 * P + 1)) & m]
            self.acc[(tsi - P + np.arange(2 * P + 1)) & m] += seg * w * (1.0 / ratio)
            self.syn += P / ratio
        oidx = (e_end - n + np.arange(n)) & m
        y = self.acc[oidx].copy()
        self.acc[oidx] = 0.0
        return y


class Spectral:
    """STFT stage: formant shift (cepstral envelope warp) and whisper
    (phase randomisation). Constant latency of N samples."""

    N = 1024
    H = 256

    def __init__(self, sr):
        self.sr = sr
        N, H = self.N, self.H
        self.win = np.sqrt(np.hanning(N + 1)[:N]).astype(np.float32)
        self.scale = H / np.sum(self.win ** 2)
        self.inbuf = np.zeros(N, dtype=np.float32)
        self.ola = np.zeros(N, dtype=np.float32)
        self.pending = np.zeros(0, dtype=np.float32)
        self.outq = np.zeros(N, dtype=np.float32)   # latency priming
        self.lifter = max(20, int(sr / 600))
        self.k = np.arange(N // 2 + 1, dtype=np.float64)
        self.rng = np.random.default_rng(7)

    def _frame(self, ratio, whisper):
        N = self.N
        X = np.fft.rfft(self.inbuf * self.win)
        if abs(ratio - 1.0) > 1e-3:
            logm = np.log(np.abs(X) + 1e-7)
            c = np.fft.irfft(logm, N)
            L = self.lifter
            c[L:N - L] = 0.0
            env = np.fft.rfft(c).real
            warped = np.interp(self.k / ratio, self.k, env, right=env[-1] - 6.0)
            gain = np.exp(np.clip(warped - env, -4.0, 2.5))
            X = X * gain
        if whisper > 0.0:
            ph = np.exp(1j * self.rng.uniform(0, 2 * np.pi, X.shape))
            X = (1.0 - whisper) * X + whisper * np.abs(X) * ph
        return (np.fft.irfft(X, N) * self.win).astype(np.float32)

    def process(self, x, ratio, whisper):
        H, N = self.H, self.N
        self.pending = np.concatenate([self.pending, x])
        produced = []
        while len(self.pending) >= H:
            hop = self.pending[:H]
            self.pending = self.pending[H:]
            self.inbuf = np.concatenate([self.inbuf[H:], hop])
            self.ola += self._frame(ratio, whisper) * self.scale
            produced.append(self.ola[:H].copy())
            self.ola = np.concatenate([self.ola[H:], np.zeros(H, dtype=np.float32)])
        if produced:
            self.outq = np.concatenate([self.outq] + produced)
        n = len(x)
        y = self.outq[:n]
        self.outq = self.outq[n:]
        return y


class PitchTracker:
    """YIN pitch detector on the most recent 2048 samples (FFT-based)."""

    W = 1024

    def __init__(self, sr, threshold=0.2):
        self.sr = sr
        self.buf = np.zeros(2048, dtype=np.float32)
        self.lo = int(sr / 900)
        self.hi = min(int(sr / 65), 2048 - self.W - 1)
        self.th = threshold

    def update(self, x):
        n = len(x)
        if n >= len(self.buf):
            self.buf[:] = x[-len(self.buf):]
        else:
            self.buf[:-n] = self.buf[n:]
            self.buf[-n:] = x
        W, hi = self.W, self.hi
        sig = self.buf.astype(np.float64)
        frame = sig[:W]
        if float(np.dot(frame, frame)) < 1e-4:
            return 0.0
        # difference function d(tau) = sum (x[j] - x[j+tau])^2 over j < W
        nfft = 4096
        F = np.fft.rfft(frame, nfft)
        G = np.fft.rfft(sig[: W + hi], nfft)
        cross = np.fft.irfft(np.conj(F) * G, nfft)[: hi + 1]
        sq = np.concatenate([[0.0], np.cumsum(sig[: W + hi] ** 2)])
        e0 = sq[W]
        et = sq[np.arange(hi + 1) + W] - sq[np.arange(hi + 1)]
        d = e0 + et - 2.0 * cross
        d[0] = 0.0
        cm = np.cumsum(d[1:])
        cmnd = np.ones(hi + 1)
        cmnd[1:] = d[1:] * np.arange(1, hi + 1) / np.maximum(cm, 1e-12)
        seg = cmnd[self.lo: hi]
        below = np.where(seg < self.th)[0]
        if len(below) == 0:
            return 0.0
        i = below[0]
        while i + 1 < len(seg) and seg[i + 1] < seg[i]:
            i += 1
        tau = float(i + self.lo)
        if 1 <= i < len(seg) - 1:
            a, b, c = seg[i - 1], seg[i], seg[i + 1]
            den = a - 2 * b + c
            if abs(den) > 1e-12:
                tau += 0.5 * (a - c) / den
        return self.sr / tau


class Vibrato:
    def __init__(self, sr):
        self.sr = sr
        self.line = DelayLine(sr // 10)
        self.lfo = LFO(sr)
        self.base = 0.006 * sr
        self.span = 0.004 * sr

    def process(self, x, depth, rate):
        self.line.write(x)
        delay = self.base + self.span * depth * self.lfo.sine(len(x), rate)
        return self.line.read_frac(len(x), delay)


class Chorus:
    def __init__(self, sr, base=0.022, span=0.006):
        self.sr = sr
        self.line = DelayLine(sr // 5)
        self.l1, self.l2 = LFO(sr), LFO(sr)
        self.l2.phase = 2.0
        self.base = base * sr
        self.span = span * sr

    def process(self, x, mix, rate):
        n = len(x)
        self.line.write(x)
        t1 = self.line.read_frac(n, self.base + self.span * self.l1.sine(n, rate))
        t2 = self.line.read_frac(n, self.base * 1.3 + self.span * self.l2.sine(n, rate * 1.37))
        return x * (1.0 - 0.5 * mix) + mix * 0.5 * (t1 + t2)


class Flanger(Chorus):
    def __init__(self, sr):
        super().__init__(sr, base=0.0035, span=0.003)

    def process(self, x, mix, rate):
        n = len(x)
        self.line.write(x)
        t = self.line.read_frac(n, self.base + self.span * self.l1.sine(n, rate))
        return (x + mix * t) / (1.0 + 0.5 * mix)


class Comb:
    """Feedback comb filter: metallic, tube and helmet resonances."""

    def __init__(self, sr):
        self.sr = sr
        self.D = None
        self.fb = None
        self.zi = None

    def process(self, x, mix, freq, fb):
        D = max(1, int(round(self.sr / max(freq, 20.0))))
        fb = float(np.clip(fb, 0.0, 0.95))
        if D != self.D or fb != self.fb:
            self.a = np.zeros(D + 1)
            self.a[0] = 1.0
            self.a[D] = -fb
            if self.zi is None or len(self.zi) != D:
                self.zi = np.zeros(D)
            self.D, self.fb = D, fb
        y, self.zi = lfilter([1.0 - fb], self.a, x, zi=self.zi)
        return x * (1.0 - mix) + mix * y.astype(np.float32)


class RingMod:
    def __init__(self, sr):
        self.lfo = LFO(sr)

    def process(self, x, mix, freq):
        carrier = self.lfo.sine(len(x), freq)
        return x * (1.0 - mix) + x * carrier * mix


class Wah:
    """Auto-wah: a resonant band-pass swept by an LFO, updated per block."""

    def __init__(self, sr):
        self.sr = sr
        self.lfo = LFO(sr)
        self.zi = np.zeros(2)

    def process(self, x, mix, rate):
        v = self.lfo.value(len(x), rate)
        f = 350.0 * 2 ** (2.5 * (0.5 + 0.5 * v))
        w0 = 2 * np.pi * f / self.sr
        q = 3.0
        alpha = np.sin(w0) / (2 * q)
        b = np.array([alpha, 0.0, -alpha])
        a = np.array([1 + alpha, -2 * np.cos(w0), 1 - alpha])
        y, self.zi = lfilter(b / a[0], a / a[0], x, zi=self.zi)
        return x * (1.0 - mix) + mix * 3.0 * y.astype(np.float32)


class Echo:
    def __init__(self, sr):
        self.sr = sr
        self.line = DelayLine(sr * 2)

    def process(self, x, mix, time_s, fb):
        n = len(x)
        T = max(n, int(time_s * self.sr))
        d = self.line.read_int(n, T)
        self.line.write(x + fb * d)
        return x + mix * d


class Reverb:
    """Schroeder reverb: four damped feedback combs, two all-passes."""

    COMBS = (1116, 1188, 1277, 1356)   # at 44.1 kHz
    ALLPASS = (556, 441)

    def __init__(self, sr, block):
        self.sr = sr
        self.block = block
        self.damp = 0.3
        self._build(1.5)

    def _build(self, size):
        self.size = size
        scale = self.sr / 44100.0 * size
        self.comb_len = [max(self.block, int(L * scale)) for L in self.COMBS]
        self.ap_len = [max(self.block, int(L * self.sr / 44100.0)) for L in self.ALLPASS]
        self.combs = [DelayLine(L + self.block) for L in self.comb_len]
        self.aps = [DelayLine(L + self.block) for L in self.ap_len]
        self.comb_zi = [np.zeros(1) for _ in self.COMBS]
        self.fb = float(np.clip(0.72 + 0.09 * size, 0.6, 0.96))

    def process(self, x, mix, size):
        if abs(size - self.size) > 1e-6:
            self._build(size)
        n = len(x)
        acc = np.zeros(n, dtype=np.float32)
        b = [1.0 - self.damp]
        a = [1.0, -self.damp]
        for i, line in enumerate(self.combs):
            d = line.read_int(n, self.comb_len[i])
            d_lp, self.comb_zi[i] = lfilter(b, a, d, zi=self.comb_zi[i])
            line.write(x + self.fb * d_lp.astype(np.float32))
            acc += d
        acc *= 0.25
        g = 0.5
        for i, line in enumerate(self.aps):
            d = line.read_int(n, self.ap_len[i])
            y = -g * acc + d
            line.write(acc + g * y)
            acc = y
        return x * (1.0 - 0.5 * mix) + mix * acc


def drive_curve(x, amount):
    k = 1.0 + amount * 40.0
    return ((1.0 + k) * x / (1.0 + k * np.abs(x))) / (1.0 + amount * 0.5)


def crush(x, amount):
    bits = 12.0 - amount * 9.0          # 12 bits .. 3 bits
    steps = 2.0 ** bits
    return np.round(x * steps) / steps


class Decimator:
    def __init__(self):
        self.count = 0
        self.held = 0.0

    def process(self, x, factor):
        k = max(1, int(round(factor)))
        n = len(x)
        t = self.count + np.arange(n)
        mark = (t % k) == 0
        idx = np.where(mark, np.arange(n), -1)
        idx = np.maximum.accumulate(idx)
        y = np.where(idx >= 0, x[np.clip(idx, 0, None)], self.held).astype(np.float32)
        self.held = float(y[-1])
        self.count = (self.count + n) % k
        return y


class Dropout:
    """Random cut-outs like a bad radio or call connection."""

    def __init__(self, sr, seed=11):
        self.sr = sr
        self.rng = np.random.default_rng(seed)
        self.remaining = 0
        self.gain = 1.0

    def process(self, x, amount):
        n = len(x)
        if self.remaining <= 0 and self.rng.random() < amount * 0.04 * n / 256:
            self.remaining = int(self.sr * self.rng.uniform(0.03, 0.25))
        target = 0.0 if self.remaining > 0 else 1.0
        self.remaining -= n
        g = np.linspace(self.gain, target, n, dtype=np.float32)
        self.gain = target
        return x * g


class VoiceEngine:
    def __init__(self, sample_rate=48000, block_size=256):
        self.sr = sample_rate
        self.block = block_size
        sr = sample_rate
        self.params = Params()
        self.hp = Biquad(sr, "highpass")
        self.lp = Biquad(sr, "lowpass")
        self.spectral = Spectral(sr)
        self.tracker = PitchTracker(sr)
        self.hist = []
        self.t = 0
        self.pitch = PSOLA(sr, self.hist)
        self.h1 = PSOLA(sr, self.hist)
        self.h2 = PSOLA(sr, self.hist)
        self.ring = RingMod(sr)
        self.vib = Vibrato(sr)
        self.chorus = Chorus(sr)
        self.flanger = Flanger(sr)
        self.comb = Comb(sr)
        self.wah = Wah(sr)
        self.trem = LFO(sr)
        self.pitch_lfo = LFO(sr)
        self.decim = Decimator()
        self.drop = Dropout(sr)
        self.echo = Echo(sr)
        self.reverb = Reverb(sr, block_size)
        self.rng = np.random.default_rng(3)
        self._smooth = {}
        self.tune_corr = 0.0
        self.f0 = 0.0
        self.period = None
        self.last_voiced = 0
        self.rnd_target = 0.0
        self.rnd_value = 0.0
        self.rnd_timer = 0
        self.env = 0.0
        self.lv_in = 0.0
        self.lv_out = 0.0
        self.match_gain = 1.0
        self.in_peak = 0.0
        self.out_peak = 0.0

    def set_params(self, p: Params):
        self.params = p

    def _ramp(self, key, target, n):
        prev = self._smooth.get(key, target)
        self._smooth[key] = target
        if prev == target:
            return target
        return np.linspace(prev, target, n, dtype=np.float32)

    def _pitch_offset(self, x, p):
        """Track pitch for this block and return the semitone offset from
        pitch movement (wobble, random jumps). Auto-tune is applied per grain."""
        n = len(x)
        self.t += n
        f0 = self.tracker.update(x)
        if f0 > 0:
            self.last_voiced = self.t
            self.period = self.sr / f0
        elif self.period and self.t - self.last_voiced > 0.06 * self.sr:
            self.period = None          # hold the last period through short gaps only
        self.f0 = f0
        # the YIN frame is the older half of the 2048-sample buffer
        self.hist.append((self.t - 1400, self.period or 0))
        if len(self.hist) > 400:
            del self.hist[:200]
        off = 0.0
        if p.pitch_lfo > 0.0:
            off += p.pitch_lfo * self.pitch_lfo.value(n, p.pitch_lfo_rate)
        if p.pitch_rnd > 0.0:
            self.rnd_timer -= n
            if self.rnd_timer <= 0:
                self.rnd_target = self.rng.uniform(-1.0, 1.0) * p.pitch_rnd
                self.rnd_timer = int(self.sr / max(p.pitch_lfo_rate, 0.1))
            self.rnd_value += 0.15 * (self.rnd_target - self.rnd_value)
            off += self.rnd_value
        return off

    def process(self, x):
        x = np.asarray(x, dtype=np.float32)
        n = len(x)
        self.in_peak = float(np.max(np.abs(x))) if n else 0.0
        p = self.params
        if not p.enabled or n == 0:
            y = x * p.volume
            self.out_peak = float(np.max(np.abs(y))) if n else 0.0
            return y

        y = x
        if abs(p.formant - 1.0) > 1e-3 or p.whisper > 0.0:
            y = self.spectral.process(y, p.formant, p.whisper)

        off = self._pitch_offset(y, p)
        base = y
        semis = p.pitch + off
        pitch_on = abs(semis) > 0.01 or p.tune > 0 or p.pitch_lfo > 0 or p.pitch_rnd > 0
        if pitch_on or p.harm_mix > 0.0:
            # PSOLA always runs while any pitch effect is on, so the dry
            # path and the extra voices stay time-aligned
            y = self.pitch.process(base, semis, p.tune, p.tune_speed)
        if p.harm_mix > 0.0:
            extra = np.zeros(n, dtype=np.float32)
            if p.harm1 != 0.0:
                extra += self.h1.process(base, p.harm1 + semis, p.tune, p.tune_speed)
            if p.harm2 != 0.0:
                extra += self.h2.process(base, p.harm2 + semis, p.tune, p.tune_speed)
            y = (y + p.harm_mix * extra) / (1.0 + 0.5 * p.harm_mix)

        if p.ring_mix > 0.0:
            y = self.ring.process(y, self._ramp("ring", p.ring_mix, n), p.ring_freq)
        if p.vib_depth > 0.0:
            y = self.vib.process(y, p.vib_depth, p.vib_rate)
        if p.chorus_mix > 0.0:
            y = self.chorus.process(y, p.chorus_mix, p.chorus_rate)
        if p.flange_mix > 0.0:
            y = self.flanger.process(y, p.flange_mix, p.flange_rate)
        if p.comb_mix > 0.0:
            y = self.comb.process(y, p.comb_mix, p.comb_freq, p.comb_fb)
        if p.wah_mix > 0.0:
            y = self.wah.process(y, p.wah_mix, p.wah_rate)
        if p.drive > 0.0:
            y = drive_curve(y, p.drive)
        if p.noise > 0.0:
            blk = float(np.sqrt(np.mean(y * y)))
            self.env += (0.3 if blk > self.env else 0.05) * (blk - self.env)
            y = y + (self.rng.standard_normal(n).astype(np.float32)
                     * p.noise * (0.004 + 1.2 * self.env))
        if p.low_cut > 20.0:
            self.hp.set(p.low_cut)
            y = self.hp.process(y)
        if p.high_cut < 20000.0:
            self.lp.set(p.high_cut)
            y = self.lp.process(y)
        if p.crush > 0.0:
            y = crush(y, p.crush)
        if p.decimate > 1.0:
            y = self.decim.process(y, p.decimate)
        if p.trem_depth > 0.0:
            y = y * (1.0 - p.trem_depth * (0.5 + 0.5 * self.trem.sine(n, p.trem_rate)))
        if p.dropout > 0.0:
            y = self.drop.process(y, p.dropout)
        if p.echo_mix > 0.0:
            y = self.echo.process(y, self._ramp("echo", p.echo_mix, n), p.echo_time, p.echo_fb)
        if p.reverb_mix > 0.0:
            y = self.reverb.process(y, self._ramp("rev", p.reverb_mix, n), p.reverb_size)
        y = self._level_match(x, y)
        y = y * self._ramp("vol", p.volume, n)
        y = np.clip(y, -1.0, 1.0).astype(np.float32)
        self.out_peak = float(np.max(np.abs(y)))
        return y

    def _level_match(self, x, y):
        """Keep every voice about as loud as the speaker. The gain only
        adapts while someone is talking, so echo tails are left alone."""
        n = len(x)
        a = 1.0 - np.exp(-n / (0.4 * self.sr))
        self.lv_in += a * (float(np.mean(x * x)) - self.lv_in)
        self.lv_out += a * (float(np.mean(y * y)) - self.lv_out)
        if self.lv_in > 1e-6 and self.lv_out > 1e-9:
            target = float(np.clip(np.sqrt(self.lv_in / self.lv_out), 0.2, 4.0))
            self.match_gain += 0.05 * (target - self.match_gain)
        g = np.linspace(self._smooth.get("match", self.match_gain), self.match_gain, n,
                        dtype=np.float32)
        self._smooth["match"] = self.match_gain
        return y * g
