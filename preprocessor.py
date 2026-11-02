"""Acoustic feature extraction (mel, F0, energy, duration).

Adapted from ming024/FastSpeech2 `preprocessor/preprocessor.py`. Algorithm is
unchanged (WORLD DIO+StoneMask pitch, STFT-norm energy, MFA durations,
phoneme-level averaging, z-score normalisation). Changes:
  * MFA >= 2 writes silences as EMPTY phone labels -> treated as "sp".
  * librosa.load now receives sr= (reference loaded at 22050 regardless).
  * pitch/energy scaler only fitted on utterances that were actually processed
    (reference re-used stale values when a TextGrid was missing).
  * train/val split can be read from split.json (exact 5 h / 1 h, disjoint),
    falling back to the reference random split.
"""
import json
import os
import random

import librosa
import numpy as np
import pyworld as pw
import tgt
from scipy.interpolate import interp1d
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

import audio as Audio

SILENCE = {"sil", "sp", "spn", ""}


class Preprocessor:
    def __init__(self, config):
        self.config = config
        self.in_dir = config["path"]["raw_path"]
        self.out_dir = config["path"]["preprocessed_path"]
        self.split_file = config["path"].get("split_file")
        self.val_size = config["preprocessing"].get("val_size", 512)
        self.sampling_rate = config["preprocessing"]["audio"]["sampling_rate"]
        self.hop_length = config["preprocessing"]["stft"]["hop_length"]

        pp = config["preprocessing"]
        assert pp["pitch"]["feature"] in ["phoneme_level", "frame_level"]
        assert pp["energy"]["feature"] in ["phoneme_level", "frame_level"]
        self.pitch_phoneme_averaging = pp["pitch"]["feature"] == "phoneme_level"
        self.energy_phoneme_averaging = pp["energy"]["feature"] == "phoneme_level"
        self.pitch_normalization = pp["pitch"]["normalization"]
        self.energy_normalization = pp["energy"]["normalization"]

        self.STFT = Audio.stft.TacotronSTFT(
            pp["stft"]["filter_length"],
            pp["stft"]["hop_length"],
            pp["stft"]["win_length"],
            pp["mel"]["n_mel_channels"],
            pp["audio"]["sampling_rate"],
            pp["mel"]["mel_fmin"],
            pp["mel"]["mel_fmax"],
        )

    # ------------------------------------------------------------------
    def build_from_path(self):
        for sub in ("mel", "pitch", "energy", "duration"):
            os.makedirs(os.path.join(self.out_dir, sub), exist_ok=True)

        print("Processing Data ...")
        out = []  # (info_string, n_frames)
        pitch_scaler = StandardScaler()
        energy_scaler = StandardScaler()

        speakers = {}
        skipped = 0
        for i, speaker in enumerate(tqdm(sorted(os.listdir(self.in_dir)))):
            speakers[speaker] = i
            for wav_name in sorted(os.listdir(os.path.join(self.in_dir, speaker))):
                if not wav_name.endswith(".wav"):
                    continue
                basename = wav_name[: -len(".wav")]
                tg_path = os.path.join(
                    self.out_dir, "TextGrid", speaker, "{}.TextGrid".format(basename)
                )
                if not os.path.exists(tg_path):
                    skipped += 1
                    continue
                ret = self.process_utterance(speaker, basename)
                if ret is None:
                    skipped += 1
                    continue
                info, pitch, energy, n = ret
                out.append((info, n))
                if len(pitch) > 0:
                    pitch_scaler.partial_fit(pitch.reshape((-1, 1)))
                if len(energy) > 0:
                    energy_scaler.partial_fit(energy.reshape((-1, 1)))
        print("Utterances processed: {}, skipped (no/failed alignment): {}".format(len(out), skipped))

        print("Computing statistic quantities ...")
        if self.pitch_normalization:
            pitch_mean, pitch_std = pitch_scaler.mean_[0], pitch_scaler.scale_[0]
        else:
            pitch_mean, pitch_std = 0, 1
        if self.energy_normalization:
            energy_mean, energy_std = energy_scaler.mean_[0], energy_scaler.scale_[0]
        else:
            energy_mean, energy_std = 0, 1

        pitch_min, pitch_max = self.normalize(
            os.path.join(self.out_dir, "pitch"), pitch_mean, pitch_std
        )
        energy_min, energy_max = self.normalize(
            os.path.join(self.out_dir, "energy"), energy_mean, energy_std
        )

        with open(os.path.join(self.out_dir, "speakers.json"), "w") as f:
            json.dump(speakers, f)
        with open(os.path.join(self.out_dir, "stats.json"), "w") as f:
            json.dump(
                {
                    "pitch": [float(pitch_min), float(pitch_max), float(pitch_mean), float(pitch_std)],
                    "energy": [float(energy_min), float(energy_max), float(energy_mean), float(energy_std)],
                },
                f,
            )

        # ---- train / validation split ---------------------------------
        if self.split_file and os.path.exists(self.split_file):
            with open(self.split_file) as f:
                split = json.load(f)
            train_ids, val_ids = set(split["train"]), set(split["val"])
            assert not (train_ids & val_ids), "train/val overlap!"
            train = [(m, n) for m, n in out if m.split("|")[0] in train_ids]
            val = [(m, n) for m, n in out if m.split("|")[0] in val_ids]
        else:
            random.seed(1234)
            random.shuffle(out)
            val, train = out[: self.val_size], out[self.val_size:]

        for name, items in (("train.txt", train), ("val.txt", val)):
            with open(os.path.join(self.out_dir, name), "w", encoding="utf-8") as f:
                for m, _ in items:
                    f.write(m + "\n")

        hrs = lambda items: sum(n for _, n in items) * self.hop_length / self.sampling_rate / 3600
        summary = {
            "train_utterances": len(train), "val_utterances": len(val),
            "train_hours": hrs(train), "val_hours": hrs(val),
            "n_speakers": len(speakers),
        }
        with open(os.path.join(self.out_dir, "data_summary.json"), "w") as f:
            json.dump(summary, f, indent=2)
        print("Data summary:", json.dumps(summary, indent=2))
        return out

    # ------------------------------------------------------------------
    def process_utterance(self, speaker, basename):
        wav_path = os.path.join(self.in_dir, speaker, "{}.wav".format(basename))
        text_path = os.path.join(self.in_dir, speaker, "{}.lab".format(basename))
        tg_path = os.path.join(self.out_dir, "TextGrid", speaker, "{}.TextGrid".format(basename))

        textgrid = tgt.io.read_textgrid(tg_path)
        phone, duration, start, end = self.get_alignment(textgrid.get_tier_by_name("phones"))
        if start >= end or len(phone) == 0 or sum(duration) <= 0:
            return None
        text = "{" + " ".join(phone) + "}"

        wav, _ = librosa.load(wav_path, sr=self.sampling_rate)
        wav = wav[int(self.sampling_rate * start): int(self.sampling_rate * end)].astype(np.float32)

        with open(text_path, "r", encoding="utf-8") as f:
            raw_text = f.readline().strip("\n")

        pitch, t = pw.dio(
            wav.astype(np.float64), self.sampling_rate,
            frame_period=self.hop_length / self.sampling_rate * 1000,
        )
        pitch = pw.stonemask(wav.astype(np.float64), pitch, t, self.sampling_rate)
        pitch = pitch[: sum(duration)]
        if np.sum(pitch != 0) <= 1:
            return None

        mel_spectrogram, energy = Audio.tools.get_mel_from_wav(wav, self.STFT)
        mel_spectrogram = mel_spectrogram[:, : sum(duration)]
        energy = energy[: sum(duration)]

        if self.pitch_phoneme_averaging:
            nonzero_ids = np.where(pitch != 0)[0]
            interp_fn = interp1d(
                nonzero_ids, pitch[nonzero_ids],
                fill_value=(pitch[nonzero_ids[0]], pitch[nonzero_ids[-1]]),
                bounds_error=False,
            )
            pitch = interp_fn(np.arange(0, len(pitch)))
            pos = 0
            for i, d in enumerate(duration):
                pitch[i] = np.mean(pitch[pos: pos + d]) if d > 0 else 0
                pos += d
            pitch = pitch[: len(duration)]

        if self.energy_phoneme_averaging:
            pos = 0
            for i, d in enumerate(duration):
                energy[i] = np.mean(energy[pos: pos + d]) if d > 0 else 0
                pos += d
            energy = energy[: len(duration)]

        np.save(os.path.join(self.out_dir, "duration", "{}-duration-{}.npy".format(speaker, basename)), duration)
        np.save(os.path.join(self.out_dir, "pitch", "{}-pitch-{}.npy".format(speaker, basename)), pitch)
        np.save(os.path.join(self.out_dir, "energy", "{}-energy-{}.npy".format(speaker, basename)), energy)
        np.save(os.path.join(self.out_dir, "mel", "{}-mel-{}.npy".format(speaker, basename)), mel_spectrogram.T)

        return (
            "|".join([basename, speaker, text, raw_text]),
            self.remove_outlier(pitch),
            self.remove_outlier(energy),
            mel_spectrogram.shape[1],
        )

    def get_alignment(self, tier):
        phones, durations = [], []
        start_time = end_time = 0
        end_idx = 0
        for t in tier._objects:
            s, e, p = t.start_time, t.end_time, t.text
            if p == "":  # MFA >= 2.0 labels silence with an empty string
                p = "sp"
            if phones == []:  # trim leading silence
                if p in SILENCE:
                    continue
                start_time = s
            phones.append(p)
            if p not in SILENCE:
                end_time = e
                end_idx = len(phones)
            durations.append(
                int(np.round(e * self.sampling_rate / self.hop_length)
                    - np.round(s * self.sampling_rate / self.hop_length))
            )
        return phones[:end_idx], durations[:end_idx], start_time, end_time

    def remove_outlier(self, values):
        values = np.array(values)
        p25, p75 = np.percentile(values, 25), np.percentile(values, 75)
        lower, upper = p25 - 1.5 * (p75 - p25), p75 + 1.5 * (p75 - p25)
        return values[np.logical_and(values > lower, values < upper)]

    def normalize(self, in_dir, mean, std):
        max_value = np.finfo(np.float64).min
        min_value = np.finfo(np.float64).max
        for filename in os.listdir(in_dir):
            filename = os.path.join(in_dir, filename)
            values = (np.load(filename) - mean) / std
            np.save(filename, values)
            max_value = max(max_value, max(values))
            min_value = min(min_value, min(values))
        return min_value, max_value
