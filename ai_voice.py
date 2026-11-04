"""AI voice: real-time RVC voice conversion.

Pipeline per block (same scheme as RVC's own realtime GUI):
  mic block -> rolling 16 kHz context window -> ContentVec features
  -> pitch (FCPE) -> RVC synthesizer -> resample -> SOLA crossfade -> out
"""

import json
import os
import threading
import time
import zipfile
from collections import deque

import numpy as np
import torch
import torch.nn.functional as F
from scipy.signal import resample_poly

APP_DIR = os.path.join(os.path.expanduser("~"), ".voicechanger")
VOICES_DIR = os.path.join(APP_DIR, "voices")
CATALOG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "voices.json")
HUBERT_REPO = "fwerkor/content-vec-best"   # safetensors build of lengyue233/content-vec-best


def pick_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


_shared = {}
_shared_lock = threading.Lock()


def get_hubert(device):
    key = ("hubert", str(device))
    with _shared_lock:
        if key not in _shared:
            from huggingface_hub import snapshot_download
            from torch import nn
            from transformers import HubertModel

            class HubertModelWithFinalProj(HubertModel):
                def __init__(self, config):
                    super().__init__(config)
                    self.final_proj = nn.Linear(config.hidden_size, config.classifier_proj_size)

            path = snapshot_download(HUBERT_REPO, allow_patterns=["*.json", "*.safetensors"])
            model = HubertModelWithFinalProj.from_pretrained(path, use_safetensors=True)
            _shared[key] = model.to(device).eval()
        return _shared[key]


def get_fcpe(device):
    key = ("fcpe", str(device))
    with _shared_lock:
        if key not in _shared:
            from torchfcpe import spawn_bundled_infer_model
            _shared[key] = spawn_bundled_infer_model(device=str(device))
        return _shared[key]


def warm_up(device, status=None):
    """Load the shared models once (first run downloads ContentVec, ~380 MB)."""
    if status:
        status("Loading ContentVec (first run downloads about 380 MB)...")
    get_hubert(device)
    if status:
        status("Loading pitch model...")
    get_fcpe(device)


# ------------------------------------------------------------------ RVC model

class RVCModel:
    def __init__(self, pth_path, device, index_path=None, index_rate=0.0):
        from rvc.models import (
            SynthesizerTrnMs256NSFsid,
            SynthesizerTrnMs256NSFsid_nono,
            SynthesizerTrnMs768NSFsid,
            SynthesizerTrnMs768NSFsid_nono,
        )
        self.device = device
        self.path = pth_path
        cpt = torch.load(pth_path, map_location="cpu", weights_only=True)
        cfg = list(cpt["config"])
        cfg[-3] = cpt["weight"]["emb_g.weight"].shape[0]
        self.tgt_sr = int(cfg[-1])
        self.if_f0 = int(cpt.get("f0", 1))
        self.version = cpt.get("version", "v1")
        if self.version == "v1":
            net = SynthesizerTrnMs256NSFsid(*cfg, is_half=False) if self.if_f0 else SynthesizerTrnMs256NSFsid_nono(*cfg)
        else:
            net = SynthesizerTrnMs768NSFsid(*cfg, is_half=False) if self.if_f0 else SynthesizerTrnMs768NSFsid_nono(*cfg)
        del net.enc_q
        net.load_state_dict(cpt["weight"], strict=False)
        net = net.float().eval().to(device)
        net.remove_weight_norm()
        self.net_g = net

        self.hubert = get_hubert(device)
        self.fcpe = get_fcpe(device)
        self.f0_min, self.f0_max = 50.0, 1100.0
        self.f0_mel_min = 1127 * np.log(1 + self.f0_min / 700)
        self.f0_mel_max = 1127 * np.log(1 + self.f0_max / 700)
        self.cache_pitch = torch.zeros(1024, device=device, dtype=torch.long)
        self.cache_pitchf = torch.zeros(1024, device=device, dtype=torch.float32)
        self.pitch_key = 0.0
        self.index = None
        self.index_rate = 0.0
        if index_path and index_rate > 0 and os.path.isfile(index_path):
            try:
                import faiss
                self.index = faiss.read_index(index_path)
                self.big_npy = self.index.reconstruct_n(0, self.index.ntotal)
                self.index_rate = float(index_rate)
            except Exception:
                self.index = None

    def reset(self):
        self.cache_pitch.zero_()
        self.cache_pitchf.zero_()

    def _f0_post(self, f0):
        f0 = torch.as_tensor(f0, dtype=torch.float32, device=self.device).squeeze()
        f0_mel = 1127 * torch.log(1 + f0 / 700)
        f0_mel[f0_mel > 0] = (f0_mel[f0_mel > 0] - self.f0_mel_min) * 254 / (
            self.f0_mel_max - self.f0_mel_min) + 1
        f0_mel[f0_mel <= 1] = 1
        f0_mel[f0_mel > 255] = 255
        return torch.round(f0_mel).long(), f0

    def get_f0(self, x, up_key):
        n = x.shape[0]
        f0 = self.fcpe.infer(
            x.float().view(1, -1), sr=16000, decoder_mode="local_argmax",
            threshold=0.006, f0_min=self.f0_min, f0_max=self.f0_max,
            interp_uv=False, output_interp_target_length=n // 160 + 1,
        )
        f0 = f0.squeeze().detach().cpu().numpy().astype(np.float64)
        uv = f0 == 0
        if np.any(~uv):
            f0[uv] = np.interp(np.where(uv)[0], np.where(~uv)[0], f0[~uv])
        f0 *= 2 ** (up_key / 12)
        return self._f0_post(f0)

    @torch.no_grad()
    def infer(self, input_wav, block_frame_16k, skip_head, return_length):
        """input_wav: 1-D 16 kHz tensor on device. Returns audio at tgt_sr."""
        feats = input_wav.float().view(1, -1)
        if self.version == "v1":
            out = self.hubert(input_values=feats, output_hidden_states=True, return_dict=True)
            feats = self.hubert.final_proj(out.hidden_states[9])
        else:
            feats = self.hubert(input_values=feats, return_dict=True).last_hidden_state
        feats = torch.cat((feats, feats[:, -1:, :]), 1)

        if self.index is not None and self.index_rate > 0:
            npy = feats[0][skip_head // 2:].cpu().numpy().astype("float32")
            score, ix = self.index.search(npy, k=8)
            if (ix >= 0).all():
                weight = np.square(1 / (score + 1e-9))
                weight /= weight.sum(axis=1, keepdims=True)
                npy = np.sum(self.big_npy[ix] * np.expand_dims(weight, axis=2), axis=1)
                feats[0][skip_head // 2:] = (
                    torch.from_numpy(npy).unsqueeze(0).to(self.device) * self.index_rate
                    + (1 - self.index_rate) * feats[0][skip_head // 2:])

        p_len = input_wav.shape[0] // 160
        if self.if_f0:
            f0_frame = block_frame_16k + 800
            pitch, pitchf = self.get_f0(input_wav[-f0_frame:], self.pitch_key)
            shift = block_frame_16k // 160
            self.cache_pitch[:-shift] = self.cache_pitch[shift:].clone()
            self.cache_pitchf[:-shift] = self.cache_pitchf[shift:].clone()
            self.cache_pitch[4 - pitch.shape[0]:] = pitch[3:-1]
            self.cache_pitchf[4 - pitch.shape[0]:] = pitchf[3:-1]
            cache_pitch = self.cache_pitch[None, -p_len:]
            cache_pitchf = self.cache_pitchf[None, -p_len:]

        feats = F.interpolate(feats.permute(0, 2, 1), scale_factor=2).permute(0, 2, 1)
        feats = feats[:, :p_len, :]
        p_len_t = torch.LongTensor([p_len]).to(self.device)
        sid = torch.LongTensor([0]).to(self.device)
        if self.if_f0:
            audio = self.net_g.infer(feats, p_len_t, cache_pitch, cache_pitchf, sid,
                                     skip_head, return_length, return_length)[0]
        else:
            audio = self.net_g.infer(feats, p_len_t, sid, skip_head, return_length, return_length)[0]
        return audio.squeeze().float()


# ------------------------------------------------------------------ streaming

class RealtimeConverter:
    """Feeds fixed-size blocks through the model with context and SOLA stitching."""

    def __init__(self, model, sr, block_time=0.3, crossfade_time=0.05, extra_time=1.0):
        self.model = model
        self.sr = sr
        zc = sr // 100
        self.zc = zc
        self.block_frame = max(1, round(block_time * sr / zc)) * zc
        self.block_frame_16k = 160 * self.block_frame // zc
        self.crossfade_frame = max(1, round(crossfade_time * sr / zc)) * zc
        self.sola_buffer_frame = min(self.crossfade_frame, 4 * zc)
        self.sola_search_frame = zc
        self.extra_frame = max(1, round(extra_time * sr / zc)) * zc
        total = self.extra_frame + self.crossfade_frame + self.sola_search_frame + self.block_frame
        self.input_wav = np.zeros(total, dtype=np.float32)
        self.input_wav_res = torch.zeros(160 * total // zc, device=model.device, dtype=torch.float32)
        self.sola_buffer = np.zeros(self.sola_buffer_frame, dtype=np.float32)
        self.skip_head = self.extra_frame // zc
        self.return_length = (self.block_frame + self.sola_buffer_frame + self.sola_search_frame) // zc
        self.fade_in = (np.sin(0.5 * np.pi * np.linspace(0, 1, self.sola_buffer_frame)) ** 2).astype(np.float32)
        self.fade_out = 1.0 - self.fade_in
        self.last_ms = 0.0
        model.reset()

    def process(self, indata):
        t0 = time.perf_counter()
        bf, zc = self.block_frame, self.zc
        self.input_wav[:-bf] = self.input_wav[bf:]
        self.input_wav[-bf:] = indata
        seg = self.input_wav[-bf - 2 * zc:]
        res = resample_poly(seg, 160, zc).astype(np.float32)
        b16 = self.block_frame_16k
        self.input_wav_res[:-b16] = self.input_wav_res[b16:].clone()
        self.input_wav_res[-b16 - 160:] = torch.from_numpy(res[160:]).to(self.model.device)

        out = self.model.infer(self.input_wav_res, b16, self.skip_head, self.return_length)
        out = out.cpu().numpy().astype(np.float32)
        if self.model.tgt_sr != self.sr:
            out = resample_poly(out, zc, self.model.tgt_sr // 100).astype(np.float32)
        expected = self.return_length * zc
        if len(out) < expected:
            out = np.pad(out, (0, expected - len(out)))
        out = out[:expected]

        sb, ss = self.sola_buffer_frame, self.sola_search_frame
        conv_input = out[: sb + ss]
        nom = np.correlate(conv_input, self.sola_buffer, mode="valid")
        den = np.sqrt(np.convolve(conv_input ** 2, np.ones(sb, dtype=np.float32), mode="valid") + 1e-8)
        off = int(np.argmax(nom / den))
        out = out[off:]
        out[:sb] = out[:sb] * self.fade_in + self.sola_buffer * self.fade_out
        self.sola_buffer = out[bf: bf + sb].copy()
        self.last_ms = (time.perf_counter() - t0) * 1000.0
        return out[:bf].copy()


class AIVoice:
    """Owns a worker thread. push() is safe to call from the audio callback:
    it hands blocks to the worker and returns converted audio as it arrives."""

    def __init__(self):
        self.model = None
        self.converter = None
        self.thread = None
        self.running = False
        self.lock = threading.Lock()
        self.in_q = deque()
        self.out_q = deque()
        self.out_off = 0
        self.acc = None
        self.acc_n = 0
        self.underruns = 0
        self.dropped = 0
        self.proc_ms = 0.0
        self.device = pick_device()
        self.status = ""

    def load_voice(self, pth_path, index_path=None, index_rate=0.0, status=None):
        warm_up(self.device, status)
        if status:
            status(f"Loading voice {os.path.basename(pth_path)}...")
        model = RVCModel(pth_path, self.device, index_path, index_rate)
        with self.lock:
            self.model = model
            if self.converter is not None:
                self.converter = RealtimeConverter(model, self.converter.sr,
                                                   self._block_time, self._extra_time)
        return model

    def set_pitch_key(self, key):
        if self.model is not None:
            self.model.pitch_key = float(key)

    def start(self, sr, block_time=0.3, extra_time=1.0):
        if self.model is None:
            raise RuntimeError("Load a voice first")
        self.stop()
        self._block_time, self._extra_time = block_time, extra_time
        self.converter = RealtimeConverter(self.model, sr, block_time, 0.05, extra_time)
        self.acc = np.zeros(self.converter.block_frame, dtype=np.float32)
        self.acc_n = 0
        self.in_q.clear()
        self.out_q.clear()
        self.out_off = 0
        self.underruns = 0
        self.dropped = 0
        self.buffering = True
        self.wait = 0
        self.margin = int(0.08 * sr)   # jitter cushion before playback starts
        self.running = True
        self.thread = threading.Thread(target=self._worker, daemon=True)
        self.thread.start()

    def stop(self):
        self.running = False
        if self.thread is not None:
            self.thread.join(timeout=2.0)
            self.thread = None

    def _worker(self):
        while self.running:
            with self.lock:
                conv = self.converter
                block = self.in_q.popleft() if self.in_q else None
            if block is None:
                time.sleep(0.005)
                continue
            try:
                y = conv.process(block)
                self.proc_ms = conv.last_ms
            except Exception as e:  # keep the stream alive
                self.status = f"AI error: {e}"
                y = np.zeros_like(block)
            with self.lock:
                self.out_q.append(y)

    def push(self, x):
        """x: float32 block from the audio callback. Returns same-length audio."""
        if not self.running or self.converter is None:
            return x
        n = len(x)
        bf = self.converter.block_frame
        pos = 0
        while pos < n:
            take = min(n - pos, bf - self.acc_n)
            self.acc[self.acc_n: self.acc_n + take] = x[pos: pos + take]
            self.acc_n += take
            pos += take
            if self.acc_n == bf:
                with self.lock:
                    self.in_q.append(self.acc.copy())
                    while len(self.in_q) > 2:   # worker too slow: keep latency bounded
                        self.in_q.popleft()
                        self.dropped += 1
                self.acc_n = 0
        y = np.zeros(n, dtype=np.float32)
        filled = 0
        with self.lock:
            if self.buffering:
                if self.out_q:
                    self.wait += n
                if self.wait < self.margin:
                    return y
                self.buffering = False
                self.wait = 0
            while filled < n and self.out_q:
                c = self.out_q[0]
                take = min(n - filled, len(c) - self.out_off)
                y[filled: filled + take] = c[self.out_off: self.out_off + take]
                filled += take
                self.out_off += take
                if self.out_off >= len(c):
                    self.out_q.popleft()
                    self.out_off = 0
        if filled < n:
            self.underruns += 1
            self.buffering = True     # refill the cushion instead of stuttering
        return y

    def latency_ms(self):
        if self.converter is None:
            return 0.0
        return (1000.0 * self.converter.block_frame / self.converter.sr + self.proc_ms
                + 1000.0 * self.margin / self.converter.sr)


# ------------------------------------------------------------------ voice library

def load_catalog():
    try:
        with open(CATALOG_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return []


def voice_dir(voice_id):
    return os.path.join(VOICES_DIR, voice_id)


def find_model_files(folder):
    """Return (pth_path, index_path) found under folder, or (None, None)."""
    pth = idx = None
    for root, _, files in os.walk(folder):
        for fn in files:
            low = fn.lower()
            p = os.path.join(root, fn)
            if low.endswith(".pth") and pth is None and os.path.getsize(p) > 10e6:
                pth = p
            elif low.endswith(".index") and idx is None:
                idx = p
    return pth, idx


def is_downloaded(voice):
    pth, _ = find_model_files(voice_dir(voice["id"]))
    return pth is not None


def download_voice(voice, status=None):
    """Fetch a catalog voice into VOICES_DIR/<id>/ and return (pth, index)."""
    from huggingface_hub import hf_hub_download
    target = voice_dir(voice["id"])
    os.makedirs(target, exist_ok=True)
    for fn in voice["files"]:
        if status:
            status(f"Downloading {os.path.basename(fn)}...")
        local = hf_hub_download(voice["repo"], fn, local_dir=target)
        if local.lower().endswith(".zip"):
            if status:
                status("Unpacking...")
            with zipfile.ZipFile(local) as z:
                for member in z.namelist():
                    low = member.lower()
                    if low.endswith(".pth") or low.endswith(".index"):
                        z.extract(member, target)
            os.remove(local)
    pth, idx = find_model_files(target)
    if pth is None:
        raise RuntimeError("Download finished but no .pth model was found")
    return pth, idx
