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
    pitch: float = 0.0        # semitones, -12..12
    ring_mix: float = 0.0     # 0..1  robot amount
    ring_freq: float = 60.0   # Hz    robot tone
    echo_mix: float = 0.0     # 0..1
    echo_time: float = 0.3    # seconds
    echo_fb: float = 0.3      # 0..0.9 repeats
    reverb_mix: float = 0.0   # 0..1
    reverb_size: float = 1.5  # 0.5..3
    low_cut: float = 20.0     # Hz   high-pass
    high_cut: float = 20000.0 # Hz   low-pass
    drive: float = 0.0        # 0..1 distortion
    crush: float = 0.0        # 0..1 bit crush
    vib_depth: float = 0.0    # 0..1
    vib_rate: float = 5.0     # Hz
    volume: float = 1.0       # 0..2

    def to_dict(self):
        return asdict(self)

    @staticmethod
    def from_dict(d):
        base = Params()
        known = {k: v for k, v in d.items() if k in base.__dict__}
        return replace(base, **known)


PRESETS = {
    "Normal":    {},
    "Chipmunk":  {"pitch": 7},
    "Child":     {"pitch": 5, "high_cut": 9000},
    "Deep":      {"pitch": -5},
    "Giant":     {"pitch": -12, "echo_mix": 0.25, "echo_time": 0.12, "echo_fb": 0.2},
    "Monster":   {"pitch": -9, "drive": 0.3, "reverb_mix": 0.2, "reverb_size": 1.5},
    "Villain":   {"pitch": -7, "low_cut": 100, "high_cut": 5000, "drive": 0.25,
                  "ring_mix": 0.25, "ring_freq": 40},
    "Robot":     {"ring_mix": 0.9, "ring_freq": 60, "high_cut": 5000},
    "Alien":     {"pitch": 4, "ring_mix": 0.5, "ring_freq": 150, "vib_depth": 0.5, "vib_rate": 5},
    "Ghost":     {"pitch": -2, "reverb_mix": 0.8, "reverb_size": 3, "vib_depth": 0.3,
                  "vib_rate": 3, "echo_mix": 0.4, "echo_time": 0.5, "echo_fb": 0.5},
    "Cave":      {"reverb_mix": 0.7, "reverb_size": 2.5, "echo_mix": 0.3, "echo_time": 0.4,
                  "echo_fb": 0.4},
    "Telephone": {"low_cut": 400, "high_cut": 3200, "drive": 0.2},
    "Radio":     {"low_cut": 300, "high_cut": 4500, "drive": 0.35, "crush": 0.2},
    "Megaphone": {"low_cut": 500, "high_cut": 4000, "drive": 0.6},
    "8-bit":     {"crush": 0.7, "high_cut": 6000},
}


def preset_params(name):
    return Params.from_dict(PRESETS.get(name, {}))


# ---------------------------------------------------------------- building blocks

class DelayLine:
    """Circular buffer. read() looks back from the *current* write position,
    so call it before write() for a feedback topology."""

    def __init__(self, size):
        self.size = 1 << (int(size) - 1).bit_length()
        self.mask = self.size - 1
        self.buf = np.zeros(self.size, dtype=np.float32)
        self.pos = 0  # next write index

    def write(self, x):
        n = len(x)
        idx = (self.pos + np.arange(n)) & self.mask
        self.buf[idx] = x
        self.pos = (self.pos + n) & self.mask

    def read_int(self, n, delay):
        """Block of n samples delayed by an integer `delay` (>= n), taken
        relative to the samples about to be written."""
        idx = (self.pos + np.arange(n) - delay) & self.mask
        return self.buf[idx]

    def read_frac(self, n, delay):
        """Fractional delay per sample, relative to the last n written
        samples (call after write())."""
        base = self.pos - n + np.arange(n)
        rp = base - delay
        i0 = np.floor(rp).astype(np.int64)
        frac = (rp - i0).astype(np.float32)
        a = self.buf[i0 & self.mask]
        b = self.buf[(i0 + 1) & self.mask]
        return a + (b - a) * frac


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
    Hann cross-faded so the sum stays flat. Cheap and low latency."""

    def __init__(self, sr, window_s=0.06):
        self.sr = sr
        self.win = int(window_s * sr)
        self.line = DelayLine(self.win * 4 + 4096)
        self.phase = 0.0

    def process(self, x, semitones):
        n = len(x)
        self.line.write(x)
        ratio = 2.0 ** (semitones / 12.0)
        D = abs(ratio - 1.0) * self.win
        p1 = (self.phase + np.arange(n) / self.win) % 1.0
        p2 = (p1 + 0.5) % 1.0
        self.phase = (self.phase + n / self.win) % 1.0
        if ratio < 1.0:
            d1, d2 = D * p1, D * p2
        else:
            d1, d2 = D * (1.0 - p1), D * (1.0 - p2)
        g1 = 0.5 * (1.0 - np.cos(2 * np.pi * p1))
        g2 = 0.5 * (1.0 - np.cos(2 * np.pi * p2))
        r1 = self.line.read_frac(n, d1)
        r2 = self.line.read_frac(n, d2)
        return (g1 * r1 + g2 * r2).astype(np.float32)


class Vibrato:
    def __init__(self, sr):
        self.sr = sr
        self.line = DelayLine(sr // 10)
        self.phase = 0.0
        self.base = 0.006 * sr
        self.span = 0.004 * sr

    def process(self, x, depth, rate):
        n = len(x)
        self.line.write(x)
        ph = self.phase + np.arange(n) * (2 * np.pi * rate / self.sr)
        self.phase = (ph[-1] + 2 * np.pi * rate / self.sr) % (2 * np.pi)
        delay = self.base + self.span * depth * np.sin(ph)
        return self.line.read_frac(n, delay)


class RingMod:
    def __init__(self, sr):
        self.sr = sr
        self.phase = 0.0

    def process(self, x, mix, freq):
        n = len(x)
        step = 2 * np.pi * freq / self.sr
        ph = self.phase + np.arange(n) * step
        self.phase = (ph[-1] + step) % (2 * np.pi)
        carrier = np.sin(ph).astype(np.float32)
        return x * (1.0 - mix) + x * carrier * mix


class Echo:
    def __init__(self, sr, block):
        self.sr = sr
        self.block = block
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
        self.size = None
        self.combs = []
        self.aps = []
        self.comb_zi = []
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


class VoiceEngine:
    def __init__(self, sample_rate=48000, block_size=256):
        self.sr = sample_rate
        self.block = block_size
        self.params = Params()
        self.hp = Biquad(sample_rate, "highpass")
        self.lp = Biquad(sample_rate, "lowpass")
        self.pitch = PitchShifter(sample_rate)
        self.ring = RingMod(sample_rate)
        self.vib = Vibrato(sample_rate)
        self.echo = Echo(sample_rate, block_size)
        self.reverb = Reverb(sample_rate, block_size)
        self._smooth = {}
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

    def process(self, x):
        x = np.asarray(x, dtype=np.float32)
        n = len(x)
        self.in_peak = float(np.max(np.abs(x))) if n else 0.0
        p = self.params
        if not p.enabled:
            y = x * p.volume
            self.out_peak = float(np.max(np.abs(y))) if n else 0.0
            return y

        y = x
        if p.low_cut > 20.0:
            self.hp.set(p.low_cut)
            y = self.hp.process(y)
        if p.high_cut < 20000.0:
            self.lp.set(p.high_cut)
            y = self.lp.process(y)
        if abs(p.pitch) > 0.01:
            y = self.pitch.process(y, p.pitch)
        if p.ring_mix > 0.0:
            y = self.ring.process(y, self._ramp("ring", p.ring_mix, n), p.ring_freq)
        if p.vib_depth > 0.0:
            y = self.vib.process(y, p.vib_depth, p.vib_rate)
        if p.drive > 0.0:
            y = drive_curve(y, p.drive)
        if p.crush > 0.0:
            y = crush(y, p.crush)
        if p.echo_mix > 0.0:
            y = self.echo.process(y, self._ramp("echo", p.echo_mix, n), p.echo_time, p.echo_fb)
        if p.reverb_mix > 0.0:
            y = self.reverb.process(y, self._ramp("rev", p.reverb_mix, n), p.reverb_size)
        y = y * self._ramp("vol", p.volume, n)
        y = np.clip(y, -1.0, 1.0).astype(np.float32)
        self.out_peak = float(np.max(np.abs(y))) if n else 0.0
        return y
