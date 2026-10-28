"""charts.py - figures for chapter 31 (fictional call, real ASR outputs)."""
import sys; sys.path.insert(0, "../common")
import json, wave
import numpy as np, matplotlib.pyplot as plt
from fa_plot import setup, fa, save
setup()
rows = json.load(open("wer_results.json"))
models = ["tiny", "base", "small"]
audios = [("call.wav", "بدون نویز"), ("call_snr10.wav", "SNR = 10 dB"), ("call_snr0.wav", "SNR = 0 dB")]
W = {(r["model"], r["audio"]): r for r in rows}

fig, ax = plt.subplots(figsize=(8, 4))
x = np.arange(3); cols = ["#4a7bb7", "#e0a458", "#d6806b"]
for k, (a, lab) in enumerate(audios):
    v = [W[(m, a)]["wer_norm"] for m in models]
    ax.bar(x + (k - 1) * 0.27, v, 0.27, label=fa(lab), color=cols[k])
    for xi, vi in zip(x, v):
        ax.text(xi + (k - 1) * 0.27, vi + 0.02, f"{vi:.2f}", ha="center", fontsize=8)
ax.set_xticks(x); ax.set_xticklabels(models)
ax.set_ylabel(fa("WER (بعد از نرمال‌سازی)")); ax.set_ylim(0, 1.15)
ax.legend(loc="upper right"); ax.set_title(fa("کیفیت صدا بیشتر از اندازه‌ی مدل اثر دارد"))
save(fig, "ch32", "wer_by_model_noise")

fig, ax = plt.subplots(figsize=(7, 3.6))
rtf = [W[(m, "call.wav")]["rtf"] for m in models]
ax.bar(models, rtf, color="#4a7bb7")
for i, v in enumerate(rtf):
    ax.text(i, v + 0.002, f"{v:.2f}", ha="center")
ax.set_ylabel(fa("ضریب زمان واقعی (RTF)"))
ax.set_title(fa("زمان پردازش ÷ طول صدا، روی CPU جعبه‌ی ما (int8)"))
save(fig, "ch32", "rtf")

turns = json.load(open("turns.json"))
segs = json.load(open("out_small_call.json"))["segments"]
with wave.open("call.wav") as w:
    a = np.frombuffer(w.readframes(w.getnframes()), np.int16) / 32768
fig, axs = plt.subplots(3, 1, figsize=(9, 4.4), sharex=True,
                        gridspec_kw={"height_ratios": [2, 1, 1]})
t = np.arange(len(a)) / 16000
axs[0].plot(t[::20], a[::20], lw=0.4, color="#555")
axs[0].set_yticks([]); axs[0].set_ylabel(fa("موج صدا"))
c = {"SARAH": "#4a7bb7", "MARKUS": "#d6806b"}
for tr in turns:
    axs[1].barh(0, tr["end"] - tr["start"], left=tr["start"], height=0.6, color=c[tr["speaker"]])
axs[1].set_yticks([0]); axs[1].set_yticklabels([fa("گوینده")])
for s in segs:
    axs[2].barh(0, s["end"] - s["start"], left=s["start"], height=0.6, color="#9db8d6", edgecolor="k", lw=0.5)
axs[2].set_yticks([0]); axs[2].set_yticklabels([fa("بخش‌های ASR")])
axs[2].set_xlabel(fa("زمان (ثانیه)"))
from matplotlib.patches import Patch
axs[1].legend(handles=[Patch(color=c["SARAH"], label="Sarah"), Patch(color=c["MARKUS"], label="Markus")],
              loc="upper right", ncol=2, fontsize=8, bbox_to_anchor=(1, 1.6))
for ax in axs[1:]: ax.set_ylim(-0.5, 0.5)
fig.tight_layout(); save(fig, "ch32", "diarization_timeline")

fig, ax = plt.subplots(figsize=(7, 3.6))
sizes = ["tiny", "base", "small", "medium", "large"]
params = [39, 74, 244, 769, 1550]
ax.bar(sizes, params, color=["#4a7bb7"] * 3 + ["#9db8d6"] * 2)
for i, v in enumerate(params):
    ax.text(i, v + 25, fa(f"{v}M"), ha="center")
ax.set_ylabel(fa("تعداد پارامتر (میلیون)"))
ax.set_title(fa("اندازه‌های Whisper (سه تای اول را خودمان اجرا کردیم)"))
save(fig, "ch32", "whisper_sizes")

fig, ax = plt.subplots(figsize=(8, 3.4))
lab = ["بی‌نویز", "10 dB", "0 dB"]
for m, mk in zip(models, ["o-", "s-", "^-"]):
    raw = [W[(m, a)]["wer_raw"] for a, _ in audios]
    nrm = [W[(m, a)]["wer_norm"] for a, _ in audios]
    ax.plot(range(3), [r - n for r, n in zip(raw, nrm)], mk, label=m)
ax.axhline(0, color="k", lw=0.6)
ax.set_xticks(range(3)); ax.set_xticklabels([fa(l) for l in lab])
ax.set_ylabel(fa("WER خام منهای WER نرمال‌شده"))
ax.set_title(fa("نرمال‌سازی روی صدای تمیز مهم است؛ روی صدای خراب نه"))
ax.legend(); ax.grid(alpha=.3)
save(fig, "ch32", "normalisation_effect")
