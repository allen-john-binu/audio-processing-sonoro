#!/usr/bin/env python3
"""
Offline ReSpeaker chirp detector and provisional DOA analysis.

Input: six-channel, 16 kHz PCM WAV.
Outputs next to input:
  *_chirps_listen.wav       mono snippets (channel 0) concatenated with gaps
  *_chirps_4mic.wav         four candidate raw mic channels, snippets concatenated
  *_chirp_detections.csv    detections and provisional DOA
  *_chirp_detection.png    overview plot

Important:
- Channel indices are zero-based. The default candidate raw mic channels are 1,2,3,4.
- DOA is PROVISIONAL until the channel-to-physical-microphone mapping and array orientation
  are verified. It uses an idealized four-mic circular geometry.
"""
import argparse
import csv
import os
import wave
import numpy as np
from scipy import signal
from scipy.ndimage import maximum_filter1d
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FS_EXPECTED = 16000
CHIRP_DURATION_S = 0.005
F0, F1 = 500.0, 3500.0
RADIUS_M = 0.0323
SOUND_SPEED = 343.0
MIC_CHANNELS = [1, 2, 3, 4]  # zero-based indices
PRE_S = 0.010
POST_S = 0.020

def read_wav(path):
    with wave.open(path, "rb") as wf:
        nch, sw, fs, nf = wf.getnchannels(), wf.getsampwidth(), wf.getframerate(), wf.getnframes()
        raw = wf.readframes(nf)
    if sw != 2:
        raise ValueError(f"Expected 16-bit PCM WAV, found {sw*8}-bit")
    data = np.frombuffer(raw, dtype="<i2").reshape(-1, nch).astype(np.float32) / 32768.0
    return data, fs, nch

def make_template(fs):
    n = max(8, int(round(CHIRP_DURATION_S * fs)))
    t = np.arange(n) / fs
    template = signal.chirp(t, f0=F0, f1=F1, t1=CHIRP_DURATION_S, method="linear")
    template *= signal.windows.tukey(n, alpha=0.5)
    template -= template.mean()
    template /= np.linalg.norm(template) + 1e-12
    return template.astype(np.float32)

def detect_chirps(data, fs, mic_channels, threshold, min_gap_s):
    template = make_template(fs)
    n = len(template)
    # Matched-filter score, normalized by local signal energy, independently per channel.
    scores = []
    for ch in mic_channels:
        x = data[:, ch].astype(np.float64)
        corr = signal.fftconvolve(x, template[::-1].astype(np.float64), mode="valid")
        energy = signal.fftconvolve(x*x, np.ones(n), mode="valid")
        score = np.abs(corr) / (np.sqrt(np.maximum(energy, 1e-15)) + 1e-12)
        scores.append(score)
    combined = np.max(np.vstack(scores), axis=0)
    # Pick peaks and retain only those above the normalized matched-filter threshold.
    distance = max(1, int(min_gap_s * fs))
    peaks, props = signal.find_peaks(combined, height=threshold, distance=distance, prominence=threshold*0.25)
    return peaks, combined, template

def gcc_phat_delay(a, b, fs, max_tau):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    a = a - np.mean(a); b = b - np.mean(b)
    n = len(a) + len(b)
    nfft = 1 << int(np.ceil(np.log2(max(2, n))))
    A = np.fft.rfft(a, n=nfft)
    B = np.fft.rfft(b, n=nfft)
    R = A * np.conj(B)
    R /= np.maximum(np.abs(R), 1e-12)
    cc = np.fft.irfft(R, n=nfft)
    max_shift = min(int(round(fs*max_tau)), nfft//2-1)
    cc = np.concatenate((cc[-max_shift:], cc[:max_shift+1]))
    k = int(np.argmax(np.abs(cc))) - max_shift
    return k / fs

def provisional_doa(segment, fs, mic_channels):
    # Idealized circular positions; these are assumptions, not verified channel mapping.
    angles = np.deg2rad(np.array([45., 135., 225., 315.]))
    xy = RADIUS_M * np.column_stack((np.cos(angles), np.sin(angles)))
    # Estimate pairwise delays, then find angle minimizing the far-field delay residual.
    delays = {}
    for i in range(4):
        for j in range(i+1, 4):
            max_tau = np.linalg.norm(xy[i]-xy[j]) / SOUND_SPEED
            delays[(i,j)] = gcc_phat_delay(segment[:, mic_channels[i]], segment[:, mic_channels[j]], fs, max_tau)
    theta_grid = np.arange(0., 360., 1.)
    errs = []
    for th in np.deg2rad(theta_grid):
        direction = np.array([np.cos(th), np.sin(th)])
        err = []
        for (i,j), tau in delays.items():
            # Sign convention can flip depending on correlation convention; compare both signs.
            predicted = np.dot(xy[i]-xy[j], direction) / SOUND_SPEED
            err.append((abs(tau-predicted)))
        errs.append(sum(err))
    errs = np.asarray(errs)
    idx = int(np.argmin(errs))
    # Also calculate opposite sign convention; choose whichever gives lower residual.
    errs2 = []
    for th in np.deg2rad(theta_grid):
        direction = np.array([np.cos(th), np.sin(th)])
        errs2.append(sum(abs(tau + np.dot(xy[i]-xy[j], direction)/SOUND_SPEED)
                         for (i,j), tau in delays.items()))
    errs2 = np.asarray(errs2)
    if errs2.min() < errs.min():
        idx = int(np.argmin(errs2))
    residual = float(min(errs.min(), errs2.min()))
    return float(theta_grid[idx]), residual

def write_wav(path, audio, fs):
    audio = np.asarray(audio, dtype=np.float32)
    audio = np.clip(audio, -1.0, 0.999969)
    pcm = (audio * 32768.0).astype("<i2")
    with wave.open(path, "wb") as wf:
        wf.setnchannels(pcm.shape[1] if pcm.ndim == 2 else 1)
        wf.setsampwidth(2); wf.setframerate(fs)
        wf.writeframes(pcm.reshape(-1).tobytes())

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("wav", help="path to six-channel WAV")
    ap.add_argument("--threshold", type=float, default=0.28, help="normalized matched-filter threshold (default 0.28)")
    ap.add_argument("--min-gap", type=float, default=0.08, help="minimum seconds between detections")
    ap.add_argument("--pre", type=float, default=PRE_S, help="seconds before each chirp")
    ap.add_argument("--post", type=float, default=POST_S, help="seconds after each chirp")
    ap.add_argument("--mic-channels", default="1,2,3,4", help="zero-based candidate mic channels")
    args = ap.parse_args()

    data, fs, nch = read_wav(args.wav)
    if fs != FS_EXPECTED:
        print(f"WARNING: sample rate is {fs}, expected {FS_EXPECTED}; analysis uses file sample rate.")
    mic_channels = [int(v) for v in args.mic_channels.split(",")]
    if len(mic_channels) != 4 or max(mic_channels) >= nch:
        raise ValueError(f"Need exactly four valid zero-based mic channel indices; file has {nch} channels.")
    if nch != 6:
        print(f"WARNING: expected six channels, found {nch}.")
    peaks, score, template = detect_chirps(data, fs, mic_channels, args.threshold, args.min_gap)
    base = os.path.splitext(args.wav)[0]
    listen_segments, four_segments, rows = [], [], []
    pre = int(round(args.pre*fs)); post = int(round(args.post*fs))
    for number, p in enumerate(peaks, 1):
        start = max(0, int(p)-pre)
        end = min(len(data), int(p)+len(template)+post)
        seg = data[start:end, :]
        # Remove DC and normalize only for listening; preserve relative channels in the 4-mic file.
        listen = seg[:, 0:1].copy()
        listen -= listen.mean(axis=0, keepdims=True)
        peak = np.max(np.abs(listen)) if listen.size else 0
        if peak > 0: listen *= min(0.95/peak, 1.0)
        listen_segments.append(np.vstack([listen, np.zeros((int(0.05*fs),1), dtype=np.float32)]))
        four = seg[:, mic_channels].copy()
        four_segments.append(np.vstack([four, np.zeros((int(0.05*fs),4), dtype=np.float32)]))
        # DOA window includes the chirp plus a small margin; same samples for every channel.
        doa_start = max(start, int(p)-int(0.002*fs))
        doa_end = min(end, int(p)+len(template)+int(0.003*fs))
        doa_seg = data[doa_start:doa_end, :]
        theta, residual = provisional_doa(doa_seg, fs, mic_channels) if len(doa_seg) > 16 else (float("nan"), float("nan"))
        rows.append({
            "chirp_number": number,
            "time_s": round(p/fs, 6),
            "matched_filter_score": round(float(score[p]), 5),
            "snippet_start_s": round(start/fs, 6),
            "snippet_end_s": round(end/fs, 6),
            "provisional_doa_deg": round(theta, 1),
            "doa_residual_s": round(residual, 8),
            "note": "DOA assumes ideal circular geometry and candidate channel order; verify before interpreting"
        })

    listen_path = base + "_chirps_listen.wav"
    four_path = base + "_chirps_4mic.wav"
    csv_path = base + "_chirp_detections.csv"
    plot_path = base + "_chirp_detection.png"
    if listen_segments:
        write_wav(listen_path, np.vstack(listen_segments), fs)
        write_wav(four_path, np.vstack(four_segments), fs)
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [
            "chirp_number","time_s","matched_filter_score","snippet_start_s","snippet_end_s",
            "provisional_doa_deg","doa_residual_s","note"])
        writer.writeheader(); writer.writerows(rows)
    fig, ax = plt.subplots(figsize=(14,4))
    times = np.arange(len(score))/fs
    ax.plot(times[::10], score[::10], linewidth=0.7)
    if len(peaks):
        ax.scatter(peaks/fs, score[peaks], s=16)
    ax.axhline(args.threshold, linestyle="--", linewidth=1)
    ax.set(xlabel="Time (s)", ylabel="Normalized matched-filter score",
           title=f"Chirp candidates: {len(peaks)} | threshold={args.threshold:g}")
    ax.grid(True, alpha=0.25); fig.tight_layout(); fig.savefig(plot_path, dpi=140); plt.close(fig)
    print(f"Input: {args.wav}")
    print(f"Sample rate: {fs} Hz | channels: {nch} | duration: {len(data)/fs:.2f} s")
    print(f"Detected {len(peaks)} candidate chirps.")
    print(f"Listening WAV: {listen_path if listen_segments else 'not written (no detections)'}")
    print(f"Four-mic snippets: {four_path if four_segments else 'not written (no detections)'}")
    print(f"Detection/DOA CSV: {csv_path}")
    print(f"Detection plot: {plot_path}")
    print("WARNING: DOA values are provisional; verify physical channel mapping and array orientation.")
    if len(peaks) == 0:
        print("No detections. Try --threshold 0.18; if there are too many, try 0.35 or 0.40.")

if __name__ == "__main__":
    main()
