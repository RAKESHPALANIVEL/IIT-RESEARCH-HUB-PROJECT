"""Corpus subsampling (~5 h train + ~1 h validation) and raw_data preparation.

Strategy (identical for both languages, parameters in config.yaml -> data_selection):
  1. text filter  : non-empty cleaned transcript, >= min_words words
  2. de-duplicate : identical transcripts are kept once, so validation text never
                    appears in training
  3. round-robin  : candidates are shuffled per speaker and interleaved so all
                    speakers are represented evenly (seeded -> reproducible)
  4. audio filter : duration in [min_duration, max_duration] s and a heuristic
                    SNR >= min_snr_db; optional per-speaker cap
  5. stop at train_hours + val_hours; shuffle; first val_hours -> validation,
     rest -> training (utterance-level disjoint split, written to split.json)
Accepted audio is resampled to the model rate, peak-normalised, written to
raw_data/<speaker>/<id>.wav with the cleaned transcript in <id>.lab (MFA format).
"""
import json
import os
import random
from collections import defaultdict
from itertools import zip_longest

import librosa
import numpy as np
from scipy.io import wavfile
from tqdm import tqdm


def estimate_snr_db(y, frame=1024, hop=256):
    """Crude SNR: mean power of loudest 50 % frames vs quietest 10 % frames."""
    rms = librosa.feature.rms(y=y, frame_length=frame, hop_length=hop)[0] + 1e-8
    p = np.sort(rms ** 2)
    noise = p[: max(1, int(0.1 * len(p)))].mean()
    signal = p[int(0.5 * len(p)):].mean()
    return float(10 * np.log10(signal / noise))


def select_and_prepare(records, raw_dir, split_file, report_file, sel_cfg, audio_cfg):
    """records: list of dict(id, speaker, wav, text) with CLEANED text."""
    rng = random.Random(sel_cfg.get("seed", 1234))
    sr = audio_cfg["sampling_rate"]
    max_val = audio_cfg["max_wav_value"]
    # overshoot by one max-length clip so the validation split can round up without starving training
    target_sec = (sel_cfg["train_hours"] + sel_cfg["val_hours"]) * 3600 + sel_cfg["max_duration"]
    cap = sel_cfg.get("max_per_speaker_hours")
    cap_sec = cap * 3600 if cap else None

    # 1) text filter, 2) dedupe
    rng.shuffle(records)
    seen, cands = set(), []
    for r in records:
        t = r["text"].strip()
        if len(t.split()) < sel_cfg["min_words"] or t in seen:
            continue
        seen.add(t)
        cands.append(r)

    # 3) round-robin over speakers
    by_spk = defaultdict(list)
    for r in cands:
        by_spk[r["speaker"]].append(r)
    order = [r for tup in zip_longest(*[by_spk[s] for s in sorted(by_spk)]) for r in tup if r]

    accepted, total, spk_sec = [], 0.0, defaultdict(float)
    rejected = defaultdict(int)
    bar = tqdm(order, desc="Selecting")
    for r in bar:
        if total >= target_sec:
            break
        if cap_sec and spk_sec[r["speaker"]] >= cap_sec:
            rejected["speaker_cap"] += 1
            continue
        try:
            y, _ = librosa.load(r["wav"], sr=sr)
        except Exception as e:  # unreadable file
            rejected["unreadable"] += 1
            continue
        dur = len(y) / sr
        if not (sel_cfg["min_duration"] <= dur <= sel_cfg["max_duration"]):
            rejected["duration"] += 1
            continue
        if estimate_snr_db(y) < sel_cfg["min_snr_db"]:
            rejected["snr"] += 1
            continue
        y = y / max(np.abs(y).max(), 1e-6) * (max_val - 1)  # 32767, avoids int16 wrap
        out_dir = os.path.join(raw_dir, r["speaker"])
        os.makedirs(out_dir, exist_ok=True)
        wavfile.write(os.path.join(out_dir, r["id"] + ".wav"), sr, y.astype(np.int16))
        with open(os.path.join(out_dir, r["id"] + ".lab"), "w", encoding="utf-8") as f:
            f.write(r["text"])
        r = dict(r, duration=dur)
        accepted.append(r)
        total += dur
        spk_sec[r["speaker"]] += dur
        bar.set_postfix(hours="{:.2f}".format(total / 3600))

    # 5) split
    rng.shuffle(accepted)
    val, train, vsec = [], [], 0.0
    for r in accepted:
        if vsec < sel_cfg["val_hours"] * 3600:
            val.append(r)
            vsec += r["duration"]
        else:
            train.append(r)
    assert not ({r["id"] for r in val} & {r["id"] for r in train})
    os.makedirs(os.path.dirname(split_file) or ".", exist_ok=True)
    with open(split_file, "w") as f:
        json.dump({"train": [r["id"] for r in train], "val": [r["id"] for r in val]}, f)

    h = lambda rs: sum(r["duration"] for r in rs) / 3600
    report = {
        "candidates_after_text_filter_and_dedupe": len(cands),
        "accepted_utterances": len(accepted),
        "train_utterances": len(train), "val_utterances": len(val),
        "train_hours": round(h(train), 3), "val_hours": round(h(val), 3),
        "n_speakers_used": len({r["speaker"] for r in accepted}),
        "rejected": dict(rejected),
        "selection_config": sel_cfg,
    }
    os.makedirs(os.path.dirname(report_file) or ".", exist_ok=True)
    with open(report_file, "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))
    if report["train_hours"] < sel_cfg["train_hours"] * 0.95:
        print("WARNING: fewer training hours than requested were available with the current filters.")
    return report
