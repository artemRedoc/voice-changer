"""Voice Changer: real-time voice effects that feed a virtual microphone.

Run:  python app.py
Route: microphone -> effects -> output device (the virtual cable's input).
Other apps then pick the virtual cable's output as their microphone.
"""

import json
import os
import queue
import sys
import threading
import tkinter as tk
from tkinter import ttk, messagebox

import numpy as np
import sounddevice as sd

from dsp import VoiceEngine, Params, PRESETS, preset_params

APP_NAME = "Voice Changer"
SETTINGS_PATH = os.path.join(os.path.expanduser("~"), ".voicechanger.json")
BLOCK = 256
RATE_CANDIDATES = (48000, 44100)


def fmt_pct(v):
    return f"{v * 100:.0f} %"


def fmt_hz(v):
    return f"{v:.0f} Hz" if v < 1000 else f"{v / 1000:.1f} kHz"


SLIDERS = [
    # group, key, label, min, max, formatter
    ("Pitch", "pitch", "Shift", -12, 12, lambda v: f"{v:+.0f} st"),
    ("Robot", "ring_mix", "Amount", 0, 1, fmt_pct),
    ("Robot", "ring_freq", "Tone", 10, 400, fmt_hz),
    ("Echo", "echo_mix", "Amount", 0, 1, fmt_pct),
    ("Echo", "echo_time", "Time", 0.05, 1.0, lambda v: f"{v * 1000:.0f} ms"),
    ("Echo", "echo_fb", "Repeats", 0, 0.9, fmt_pct),
    ("Space", "reverb_mix", "Amount", 0, 1, fmt_pct),
    ("Space", "reverb_size", "Size", 0.5, 3.0, lambda v: f"{v:.1f} x"),
    ("Tone", "low_cut", "Low cut", 20, 1500, fmt_hz),
    ("Tone", "high_cut", "High cut", 1000, 20000, fmt_hz),
    ("Grit", "drive", "Drive", 0, 1, fmt_pct),
    ("Grit", "crush", "Crush", 0, 1, fmt_pct),
    ("Wobble", "vib_depth", "Depth", 0, 1, fmt_pct),
    ("Wobble", "vib_rate", "Speed", 0.5, 12, fmt_hz),
    ("Output", "volume", "Volume", 0, 2, fmt_pct),
]
INT_KEYS = {"pitch"}


# ------------------------------------------------------------------ devices

def list_devices():
    """Return (inputs, outputs) as lists of (index, label). On Windows the
    same device shows up once per host API; keep the WASAPI copies only."""
    devs = sd.query_devices()
    apis = sd.query_hostapis()
    names = [a["name"] for a in apis]
    preferred = None
    for cand in ("Windows WASAPI", "Core Audio", "ALSA"):
        if cand in names:
            preferred = names.index(cand)
            break
    inputs, outputs = [], []
    for i, d in enumerate(devs):
        if preferred is not None and d["hostapi"] != preferred:
            continue
        label = d["name"]
        if d["max_input_channels"] > 0:
            inputs.append((i, label))
        if d["max_output_channels"] > 0:
            outputs.append((i, label))
    return inputs, outputs


def guess_virtual_output(outputs):
    for i, label in outputs:
        low = label.lower()
        if "cable input" in low or "blackhole" in low or "virtual" in low:
            return i
    return None


# ------------------------------------------------------------------ audio

class AudioIO:
    def __init__(self):
        self.stream = None
        self.monitor = None
        self.engine = None
        self.mon_q = queue.Queue(maxsize=16)
        self.lock = threading.Lock()
        self.rate = None
        self.last_error = None

    @property
    def running(self):
        return self.stream is not None and self.stream.active

    def start(self, in_dev, out_dev, mon_dev, params):
        self.stop()
        out_ch = min(2, sd.query_devices(out_dev)["max_output_channels"])
        last = None
        rates = list(RATE_CANDIDATES)
        for dev in (in_dev, out_dev):
            r = int(sd.query_devices(dev)["default_samplerate"])
            if r not in rates:
                rates.append(r)
        for rate in rates:
            try:
                engine = VoiceEngine(rate, BLOCK)
                engine.set_params(params)
                stream = sd.Stream(
                    device=(in_dev, out_dev), samplerate=rate, blocksize=BLOCK,
                    dtype="float32", channels=(1, out_ch), latency="low",
                    callback=self._callback,
                )
                stream.start()
            except Exception as e:  # try the next rate
                last = e
                continue
            self.engine = engine
            self.stream = stream
            self.rate = rate
            self.out_ch = out_ch
            break
        else:
            raise RuntimeError(f"Could not open audio devices: {last}")

        if mon_dev is not None and mon_dev != out_dev:
            mon_ch = min(2, sd.query_devices(mon_dev)["max_output_channels"])
            self.mon_ch = mon_ch
            with self.mon_q.mutex:
                self.mon_q.queue.clear()
            try:
                self.monitor = sd.OutputStream(
                    device=mon_dev, samplerate=self.rate, blocksize=BLOCK,
                    dtype="float32", channels=mon_ch, latency="low",
                    callback=self._monitor_callback,
                )
                self.monitor.start()
            except Exception as e:
                self.monitor = None
                self.last_error = f"Monitor output failed: {e}"

    def stop(self):
        for s in (self.stream, self.monitor):
            if s is not None:
                try:
                    s.stop()
                    s.close()
                except Exception:
                    pass
        self.stream = None
        self.monitor = None

    def set_params(self, params):
        if self.engine is not None:
            self.engine.set_params(params)

    def _callback(self, indata, outdata, frames, time_info, status):
        y = self.engine.process(indata[:, 0])
        outdata[:] = y[:, None]
        if self.monitor is not None:
            try:
                self.mon_q.put_nowait(y)
            except queue.Full:
                try:
                    self.mon_q.get_nowait()
                    self.mon_q.put_nowait(y)
                except queue.Empty:
                    pass

    def _monitor_callback(self, outdata, frames, time_info, status):
        try:
            y = self.mon_q.get_nowait()
            outdata[:] = y[:frames, None]
        except queue.Empty:
            outdata.fill(0)

    def levels(self):
        if self.engine is None:
            return 0.0, 0.0
        return self.engine.in_peak, self.engine.out_peak

    def cpu_load(self):
        if self.stream is None:
            return 0.0
        try:
            return self.stream.cpu_load
        except Exception:
            return 0.0


# ------------------------------------------------------------------ settings

def load_settings():
    try:
        with open(SETTINGS_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def save_settings(data):
    try:
        with open(SETTINGS_PATH, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
    except Exception:
        pass


# ------------------------------------------------------------------ GUI

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title(APP_NAME)
        self.minsize(760, 560)
        self.audio = AudioIO()
        self.settings = load_settings()
        self.params = Params.from_dict(self.settings.get("params", {}))
        self.preset_name = self.settings.get("preset", "Normal")
        self.vars = {}
        self.value_labels = {}
        self.preset_buttons = {}
        self.updating = False
        self._save_job = None

        self._style()
        self._build()
        self._load_devices()
        self._apply_params_to_sliders()
        self._highlight_preset()
        self.protocol("WM_DELETE_WINDOW", self._on_close)
        self.after(60, self._tick)

    # ---------- layout
    def _style(self):
        style = ttk.Style(self)
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass
        self.configure(bg="#1b1e24")
        style.configure(".", background="#1b1e24", foreground="#e8e6e0", fieldbackground="#262a33")
        style.configure("TLabel", background="#1b1e24", foreground="#e8e6e0")
        style.configure("Dim.TLabel", foreground="#8b8f98")
        style.configure("Title.TLabel", font=("Segoe UI", 18, "bold"))
        style.configure("TLabelframe", background="#1b1e24", bordercolor="#333845")
        style.configure("TLabelframe.Label", background="#1b1e24", foreground="#ffb020",
                        font=("Segoe UI", 9, "bold"))
        style.configure("TButton", padding=(10, 5))
        style.configure("Preset.TButton", padding=(8, 6))
        style.configure("Active.Preset.TButton", background="#ffb020", foreground="#1b1e24")
        style.map("Active.Preset.TButton", background=[("active", "#ffc24a")])
        style.configure("On.TButton", background="#2fbf71", foreground="#0d1a12",
                        font=("Segoe UI", 11, "bold"), padding=(16, 8))
        style.map("On.TButton", background=[("active", "#3fd381")])
        style.configure("Off.TButton", background="#5a5f6a", foreground="#e8e6e0",
                        font=("Segoe UI", 11, "bold"), padding=(16, 8))
        style.configure("Horizontal.TScale", background="#1b1e24", troughcolor="#2b303a")
        style.configure("TCombobox", arrowcolor="#e8e6e0")

    def _build(self):
        pad = {"padx": 12, "pady": 6}
        top = ttk.Frame(self)
        top.pack(fill="x", **pad)
        ttk.Label(top, text=APP_NAME, style="Title.TLabel").pack(side="left")
        self.status_var = tk.StringVar(value="Stopped")
        ttk.Label(top, textvariable=self.status_var, style="Dim.TLabel").pack(side="left", padx=16)
        self.toggle_btn = ttk.Button(top, text="EFFECTS ON", style="On.TButton",
                                     command=self._toggle_enabled)
        self.toggle_btn.pack(side="right")
        self.run_btn = ttk.Button(top, text="Start", command=self._toggle_run)
        self.run_btn.pack(side="right", padx=8)

        dev = ttk.LabelFrame(self, text="DEVICES")
        dev.pack(fill="x", **pad)
        dev.columnconfigure(1, weight=1)
        dev.columnconfigure(3, weight=1)
        self.in_var = tk.StringVar()
        self.out_var = tk.StringVar()
        self.mon_var = tk.StringVar()
        ttk.Label(dev, text="Microphone").grid(row=0, column=0, sticky="w", padx=8, pady=4)
        self.in_box = ttk.Combobox(dev, textvariable=self.in_var, state="readonly")
        self.in_box.grid(row=0, column=1, sticky="ew", padx=8, pady=4)
        ttk.Label(dev, text="Send to (virtual mic)").grid(row=0, column=2, sticky="w", padx=8)
        self.out_box = ttk.Combobox(dev, textvariable=self.out_var, state="readonly")
        self.out_box.grid(row=0, column=3, sticky="ew", padx=8, pady=4)
        ttk.Label(dev, text="Hear myself on").grid(row=1, column=0, sticky="w", padx=8, pady=4)
        self.mon_box = ttk.Combobox(dev, textvariable=self.mon_var, state="readonly")
        self.mon_box.grid(row=1, column=1, sticky="ew", padx=8, pady=4)
        ttk.Button(dev, text="Refresh devices", command=self._load_devices).grid(
            row=1, column=3, sticky="e", padx=8, pady=4)
        self.hint_var = tk.StringVar(value="")
        ttk.Label(dev, textvariable=self.hint_var, style="Dim.TLabel", wraplength=700,
                  justify="left").grid(row=2, column=0, columnspan=4, sticky="w", padx=8, pady=(0, 6))

        meters = ttk.Frame(self)
        meters.pack(fill="x", padx=12)
        ttk.Label(meters, text="In", style="Dim.TLabel").grid(row=0, column=0, padx=(0, 6))
        self.in_meter = tk.Canvas(meters, height=10, bg="#262a33", highlightthickness=0)
        self.in_meter.grid(row=0, column=1, sticky="ew", pady=2)
        ttk.Label(meters, text="Out", style="Dim.TLabel").grid(row=1, column=0, padx=(0, 6))
        self.out_meter = tk.Canvas(meters, height=10, bg="#262a33", highlightthickness=0)
        self.out_meter.grid(row=1, column=1, sticky="ew", pady=2)
        meters.columnconfigure(1, weight=1)

        pre = ttk.LabelFrame(self, text="VOICES")
        pre.pack(fill="x", **pad)
        cols = 5
        for i, name in enumerate(PRESETS):
            b = ttk.Button(pre, text=name, style="Preset.TButton",
                           command=lambda n=name: self._pick_preset(n))
            b.grid(row=i // cols, column=i % cols, sticky="ew", padx=4, pady=4)
            self.preset_buttons[name] = b
        for c in range(cols):
            pre.columnconfigure(c, weight=1)

        dials = ttk.Frame(self)
        dials.pack(fill="both", expand=True, **pad)
        groups = {}
        for group, key, label, lo, hi, fmt in SLIDERS:
            if group not in groups:
                frame = ttk.LabelFrame(dials, text=group.upper())
                idx = len(groups)
                frame.grid(row=idx // 4, column=idx % 4, sticky="nsew", padx=4, pady=4)
                frame.columnconfigure(1, weight=1)
                groups[group] = frame
            frame = groups[group]
            r = len([k for g, k, *_ in SLIDERS if g == group and k in self.vars])
            var = tk.DoubleVar(value=getattr(self.params, key))
            self.vars[key] = var
            ttk.Label(frame, text=label).grid(row=r, column=0, sticky="w", padx=6, pady=2)
            s = ttk.Scale(frame, from_=lo, to=hi, variable=var, orient="horizontal",
                          command=lambda v, k=key: self._slider_changed(k))
            s.grid(row=r, column=1, sticky="ew", padx=4, pady=2)
            vl = ttk.Label(frame, text="", style="Dim.TLabel", width=8, anchor="e")
            vl.grid(row=r, column=2, sticky="e", padx=6)
            self.value_labels[key] = (vl, fmt)
        for c in range(4):
            dials.columnconfigure(c, weight=1)
        ttk.Button(dials, text="Reset dials", command=lambda: self._pick_preset("Normal")).grid(
            row=2, column=3, sticky="e", padx=4, pady=4)

    # ---------- devices
    def _load_devices(self):
        try:
            sd._terminate()
            sd._initialize()
        except Exception:
            pass
        self.inputs, self.outputs = list_devices()
        self.in_box["values"] = [l for _, l in self.inputs]
        self.out_box["values"] = [l for _, l in self.outputs]
        self.mon_box["values"] = ["(off)"] + [l for _, l in self.outputs]

        def pick(box, var, options, wanted, fallback_idx):
            labels = [l for _, l in options]
            if wanted in labels:
                var.set(wanted)
            elif fallback_idx is not None and any(i == fallback_idx for i, _ in options):
                var.set(dict(options)[fallback_idx])
            elif labels:
                var.set(labels[0])
            else:
                var.set("")

        try:
            default_in, default_out = sd.default.device
        except Exception:
            default_in, default_out = None, None
        pick(self.in_box, self.in_var, self.inputs, self.settings.get("input"), default_in)
        virtual = guess_virtual_output(self.outputs)
        pick(self.out_box, self.out_var, self.outputs, self.settings.get("output"), virtual)
        mon_saved = self.settings.get("monitor", "(off)")
        mon_labels = ["(off)"] + [l for _, l in self.outputs]
        self.mon_var.set(mon_saved if mon_saved in mon_labels else "(off)")

        if virtual is None:
            self.hint_var.set(
                "No virtual audio cable found. On Windows install VB-CABLE (free), then pick "
                "'CABLE Input' here and choose 'CABLE Output' as the microphone in Discord, "
                "games or Zoom. Without it, 'Send to' can only be your speakers or headphones.")
        else:
            self.hint_var.set(
                "Virtual cable detected. In Discord, games or Zoom pick the cable's "
                "output ('CABLE Output') as the microphone.")

    def _device_index(self, options, label):
        for i, l in options:
            if l == label:
                return i
        return None

    # ---------- run / stop
    def _toggle_run(self):
        if self.audio.running:
            self.audio.stop()
            self.run_btn.configure(text="Start")
            self.status_var.set("Stopped")
            return
        in_dev = self._device_index(self.inputs, self.in_var.get())
        out_dev = self._device_index(self.outputs, self.out_var.get())
        mon_dev = self._device_index(self.outputs, self.mon_var.get())
        if in_dev is None or out_dev is None:
            messagebox.showerror(APP_NAME, "Pick a microphone and an output device first.")
            return
        try:
            self.audio.start(in_dev, out_dev, mon_dev, self.params)
        except Exception as e:
            messagebox.showerror(APP_NAME, str(e))
            return
        self.run_btn.configure(text="Stop")
        if self.audio.last_error:
            messagebox.showwarning(APP_NAME, self.audio.last_error)
            self.audio.last_error = None
        self._schedule_save()

    def _toggle_enabled(self):
        self.params.enabled = not self.params.enabled
        self._push_params()
        self._refresh_toggle()

    def _refresh_toggle(self):
        if self.params.enabled:
            self.toggle_btn.configure(text="EFFECTS ON", style="On.TButton")
        else:
            self.toggle_btn.configure(text="EFFECTS OFF", style="Off.TButton")

    # ---------- params
    def _pick_preset(self, name):
        enabled = self.params.enabled
        self.params = preset_params(name)
        self.params.enabled = enabled
        self.preset_name = name
        self._apply_params_to_sliders()
        self._highlight_preset()
        self._push_params()

    def _highlight_preset(self):
        for name, b in self.preset_buttons.items():
            b.configure(style="Active.Preset.TButton" if name == self.preset_name
                        else "Preset.TButton")

    def _apply_params_to_sliders(self):
        self.updating = True
        for key, var in self.vars.items():
            var.set(getattr(self.params, key))
            self._refresh_value_label(key)
        self.updating = False
        self._refresh_toggle()

    def _refresh_value_label(self, key):
        lbl, fmt = self.value_labels[key]
        lbl.configure(text=fmt(self.vars[key].get()))

    def _slider_changed(self, key):
        if self.updating:
            return
        v = self.vars[key].get()
        if key in INT_KEYS:
            v = float(round(v))
            self.vars[key].set(v)
        setattr(self.params, key, v)
        self._refresh_value_label(key)
        if self.preset_name != "Custom":
            self.preset_name = "Custom"
            self._highlight_preset()
        self._push_params()

    def _push_params(self):
        self.audio.set_params(Params(**self.params.to_dict()))
        self._schedule_save()

    # ---------- persistence
    def _schedule_save(self):
        if self._save_job:
            self.after_cancel(self._save_job)
        self._save_job = self.after(800, self._save)

    def _save(self):
        self._save_job = None
        save_settings({
            "params": self.params.to_dict(),
            "preset": self.preset_name,
            "input": self.in_var.get(),
            "output": self.out_var.get(),
            "monitor": self.mon_var.get(),
        })

    # ---------- periodic
    def _tick(self):
        lin, lout = self.audio.levels()
        self._draw_meter(self.in_meter, lin)
        self._draw_meter(self.out_meter, lout)
        if self.audio.running:
            lat = BLOCK / self.audio.rate * 1000
            self.status_var.set(
                f"Running  {self.audio.rate} Hz  block {lat:.1f} ms  cpu {self.audio.cpu_load() * 100:.0f} %")
        self.after(60, self._tick)

    def _draw_meter(self, canvas, level):
        canvas.delete("all")
        w = canvas.winfo_width()
        h = canvas.winfo_height()
        db = 20 * np.log10(max(level, 1e-5))
        frac = float(np.clip((db + 60) / 60, 0, 1))
        color = "#2fbf71" if db < -6 else ("#ffb020" if db < -1 else "#ff4d4d")
        canvas.create_rectangle(0, 0, w * frac, h, fill=color, width=0)

    def _on_close(self):
        self.audio.stop()
        self._save()
        self.destroy()


def main():
    if sys.platform == "win32":
        try:
            import ctypes
            ctypes.windll.shcore.SetProcessDpiAwareness(1)
        except Exception:
            pass
    App().mainloop()


if __name__ == "__main__":
    main()
