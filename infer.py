"""Inference + variance-adaptor analysis shared by both languages.

For every test sentence: phonemise (language front end) -> FastSpeech 2 ->
HiFi-GAN -> wav, and dump the variance adaptor's *predicted* duration (frames),
pitch (F0, de-normalised to Hz) and energy as PNG plots + a JSON file.
"""
import argparse
import json
import os
import re

import matplotlib
import numpy as np
import torch
from scipy.io import wavfile

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from config import load_config, split_config  # noqa: E402
from model import FastSpeech2  # noqa: E402
from utils.model import get_vocoder, vocoder_infer  # noqa: E402
from text import text_to_sequence, _id_to_symbol  # noqa: E402

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def read_sentences(path):
    """File format: `ID|sentence` per line, '#' lines ignored."""
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line.strip() or line.startswith("#"):
                continue
            sid, text = line.split("|", 1)
            out.append((sid.strip(), text))
    return out


def find_checkpoint(ckpt_dir):
    for name in ("best_model.pth.tar", "final_model.pth.tar"):
        p = os.path.join(ckpt_dir, name)
        if os.path.exists(p):
            return p
    nums = [f for f in os.listdir(ckpt_dir) if re.fullmatch(r"\d+\.pth\.tar", f)]
    if not nums:
        raise FileNotFoundError("No checkpoint found in {}".format(ckpt_dir))
    return os.path.join(ckpt_dir, max(nums, key=lambda f: int(f.split(".")[0])))


def load_model(preprocess_config, model_config, ckpt_path):
    model = FastSpeech2(preprocess_config, model_config).to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])  # fp16 export is cast back to fp32
    model.eval()
    return model


def _boundaries(durations):
    edges = np.concatenate([[0], np.cumsum(durations)])
    return edges


def _annotate(ax, labels, edges, sec_per_frame):
    for e in edges[1:-1]:
        ax.axvline(e * sec_per_frame, color="gray", lw=0.4, ls=":")
    ymax = ax.get_ylim()[1]
    for i, lab in enumerate(labels):
        mid = (edges[i] + edges[i + 1]) / 2 * sec_per_frame
        ax.text(mid, ymax, lab, ha="center", va="bottom", fontsize=6, rotation=90)


def plot_duration(path, sid, text, labels, dur_frames, sec_per_frame):
    n = len(labels)
    fig, ax = plt.subplots(figsize=(max(9, 0.28 * n), 3.6))
    ax.bar(np.arange(n), dur_frames, color="tab:blue")
    ax.set_xticks(np.arange(n))
    ax.set_xticklabels(labels, rotation=90, fontsize=7)
    ax.set_ylabel("Predicted duration (frames)")
    sec = ax.secondary_yaxis("right", functions=(lambda f: f * sec_per_frame * 1000,
                                                 lambda m: m / (sec_per_frame * 1000)))
    sec.set_ylabel("ms")
    ax.set_title("Duration prediction - {}\n{}".format(sid, text[:110]), fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_contour(path, sid, text, labels, durations, values, ylabel, title, sec_per_frame, color):
    edges = _boundaries(durations)
    frame_vals = np.repeat(values, np.maximum(durations, 0))
    t = np.arange(len(frame_vals)) * sec_per_frame
    fig, ax = plt.subplots(figsize=(max(9, 0.28 * len(labels)), 3.8))
    ax.plot(t, frame_vals, color=color, lw=1.6)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel(ylabel)
    ax.set_xlim(0, t[-1] if len(t) else 1)
    lo, hi = np.min(values), np.max(values)
    pad = 0.15 * (hi - lo + 1e-6)
    ax.set_ylim(lo - pad, hi + 0.35 * (hi - lo + 1e-6) + pad)
    _annotate(ax, labels, edges, sec_per_frame)
    ax.set_title("{} - {}\n{}".format(title, sid, text[:110]), fontsize=9, pad=28)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


@torch.no_grad()
def synthesize_one(model, vocoder, preprocess_config, model_config, phone_string,
                   speaker_id, controls):
    seq = np.array(text_to_sequence(phone_string, preprocess_config["preprocessing"]["text"]["text_cleaners"]))
    texts = torch.from_numpy(seq).long().unsqueeze(0).to(device)
    src_lens = torch.tensor([len(seq)]).to(device)
    speakers = torch.tensor([speaker_id]).long().to(device)
    p_c, e_c, d_c = controls
    out = model(speakers, texts, src_lens, len(seq), p_control=p_c, e_control=e_c, d_control=d_c)
    mel = out[1]                      # post-net mel (1, T, 80)
    pitch, energy = out[2][0].cpu().numpy(), out[3][0].cpu().numpy()
    dur = out[5][0].cpu().numpy().astype(int)
    mel_len = int(out[9][0].item())
    hop = preprocess_config["preprocessing"]["stft"]["hop_length"]
    wav = vocoder_infer(mel.transpose(1, 2), vocoder, model_config, preprocess_config,
                        lengths=[mel_len * hop])[0]
    labels = [_id_to_symbol[int(i)].lstrip("@") for i in seq]
    return wav, labels, dur, pitch, energy


def run(language_dir, frontend, default_sentences, description):
    ap = argparse.ArgumentParser(description=description)
    ap.add_argument("--config", default=os.path.join(language_dir, "config.yaml"))
    ap.add_argument("--ckpt", default=None, help="checkpoint file (default: best_model, else latest)")
    ap.add_argument("--sentences", default=default_sentences, help="file with ID|text lines")
    ap.add_argument("--text", default=None, help="synthesise a single sentence instead")
    ap.add_argument("--results_dir", default=None, help="default: ../Results/<Language>")
    ap.add_argument("--speaker_id", type=int, default=None)
    ap.add_argument("--pitch_control", type=float, default=1.0)
    ap.add_argument("--energy_control", type=float, default=1.0)
    ap.add_argument("--duration_control", type=float, default=1.0)
    args = ap.parse_args()

    cfg = load_config(args.config)
    preprocess_config, model_config, train_config = split_config(cfg)
    ckpt = args.ckpt or find_checkpoint(train_config["path"]["ckpt_path"])
    results = args.results_dir or os.path.join("..", "Results", os.path.basename(language_dir))
    speaker_id = args.speaker_id if args.speaker_id is not None else cfg.get("inference", {}).get("speaker_id", 0)
    stats_file = os.path.join(preprocess_config["path"]["preprocessed_path"], "stats.json")
    ckpt_dir = os.path.dirname(os.path.abspath(ckpt))
    if not os.path.exists(stats_file) and os.path.exists(os.path.join(ckpt_dir, "stats.json")):
        preprocess_config["path"]["preprocessed_path"] = ckpt_dir  # stats.json/speakers.json saved with ckpt
    print("Checkpoint:", ckpt, "| speaker id:", speaker_id)

    model = load_model(preprocess_config, model_config, ckpt)
    vocoder = get_vocoder(model_config, device)
    sentences = [("single", args.text)] if args.text else read_sentences(args.sentences)

    pp = preprocess_config["preprocessing"]
    sr, hop = pp["audio"]["sampling_rate"], pp["stft"]["hop_length"]
    spf = hop / sr
    with open(os.path.join(preprocess_config["path"]["preprocessed_path"], "stats.json")) as f:
        stats = json.load(f)
    p_mean, p_std = stats["pitch"][2], stats["pitch"][3]
    e_mean, e_std = stats["energy"][2], stats["energy"][3]

    wav_dir = os.path.join(results, "test_generated_audios")
    plot_dir = os.path.join(results, "variance_adaptor_plots")
    data_dir = os.path.join(results, "variance_adaptor_data")
    for d in (wav_dir, data_dir, *[os.path.join(plot_dir, k) for k in ("duration", "pitch", "energy")]):
        os.makedirs(d, exist_ok=True)

    controls = (args.pitch_control, args.energy_control, args.duration_control)
    for sid, text in sentences:
        phones = frontend(text, preprocess_config)
        print("\n[{}] {}\n  phones: {}".format(sid, text, phones))
        wav, labels, dur, pitch, energy = synthesize_one(
            model, vocoder, preprocess_config, model_config, phones, speaker_id, controls)
        wavfile.write(os.path.join(wav_dir, "{}.wav".format(sid)), sr, wav)

        pitch_hz = pitch * p_std + p_mean       # undo z-score normalisation
        energy_dn = energy * e_std + e_mean
        plot_duration(os.path.join(plot_dir, "duration", sid + ".png"), sid, text, labels, dur, spf)
        plot_contour(os.path.join(plot_dir, "pitch", sid + ".png"), sid, text, labels, dur,
                     pitch_hz, "F0 (Hz)", "Predicted pitch (F0)", spf, "tab:red")
        plot_contour(os.path.join(plot_dir, "energy", sid + ".png"), sid, text, labels, dur,
                     energy_dn, "Energy (STFT magnitude norm)", "Predicted energy", spf, "tab:green")
        with open(os.path.join(data_dir, sid + ".json"), "w", encoding="utf-8") as f:
            json.dump({"id": sid, "text": text, "phones": labels,
                       "duration_frames": dur.tolist(),
                       "duration_ms": [round(float(d) * spf * 1000, 1) for d in dur],
                       "pitch_hz": [round(float(v), 2) for v in pitch_hz],
                       "energy": [round(float(v), 4) for v in energy_dn],
                       "total_seconds": round(float(dur.sum()) * spf, 3)},
                      f, ensure_ascii=False, indent=1)
    print("\nDone. Audio -> {}\nPlots -> {}".format(wav_dir, plot_dir))
