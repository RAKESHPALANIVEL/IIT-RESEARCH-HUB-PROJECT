# FastSpeech 2 TTS - English (Svarah) and Telugu (IndicTTS Telugu)

Two separate FastSpeech 2 models, built on the reference implementation
[ming024/FastSpeech2](https://github.com/ming024/FastSpeech2) (model, loss, optimiser, MFA-duration
preprocessing and HiFi-GAN vocoder are kept as in the reference; changes are listed below).

> **Status of this code base.** The code was written against the reference repo and the assignment spec.
> Verified here: syntax of every file, the English and Telugu text front ends (all phones map to
> symbols, dates/times/numbers are verbalised), the Telugu G2P, and the 5 h / 1 h data-selection logic
> on synthetic audio. **Not run here (no GPU, no dataset access, PyTorch could not be installed in the
> authoring sandbox):** feature extraction, training, inference and plotting on real data. Expect to fix
> small environment issues on the first run. **No checkpoints, audio or plots are included**; you
> generate them with the commands below and then fill in `Report/Technical_Report.md`.

## Folder structure

```
project/
├── English/        preprocess.py  train.py  inference.py  config.yaml  checkpoints/
│                   text/ (English front end)  lexicon/  test_sentences.txt
├── Telugu/         preprocess.py  train.py  inference.py  config.yaml  checkpoints/
│                   text/ (Telugu front end: G2P, numbers, symbols)  lexicon/  test_sentences.txt
├── Results/{English,Telugu}/   training_loss.png  validation_loss.png
│                   test_generated_audios/  variance_adaptor_plots/{duration,pitch,energy}/
├── Report/Technical_Report.md  (template -> export to Technical_Report.pdf)
├── common/         code shared by both languages (unchanged reference model + small fixes)
├── scripts/        plot_losses.py  export_model.py  setup_hifigan.sh
├── README.md  requirements.txt
```
`common/` holds the reference `model/`, `transformer/`, `audio/`, `hifigan/`, `utils/`, `dataset.py`,
`evaluate.py` once, instead of copying them into both language folders. Everything language-specific
lives in `English/` and `Telugu/`.

## What was reused and what was changed

| Part | Reference | Here |
|---|---|---|
| Model (encoder 4 / decoder 6 FFT blocks, variance adaptor, post-net), loss, Noam-style optimiser | `model/`, `transformer/` | **unchanged** |
| Features: 22.05 kHz, 80 mel, hop 256, WORLD pitch, STFT energy, phoneme-level, z-score | `preprocessor/` | same algorithm; fixes below |
| Alignment | MFA 1.x TextGrids | MFA >= 3 (empty label = silence handled); English: pretrained `english_us_arpa`; Telugu: acoustic model trained on the subset |
| Vocoder | HiFi-GAN | same (pretrained *universal* generator) for both languages |
| Text front end | ARPAbet + pinyin symbols | English: ARPAbet via LibriSpeech lexicon + g2p_en (pinyin removed) + date/time normaliser. **Telugu: new** rule-based G2P, 46-phone symbol set, Telugu number/date/time verbalisation, `telugu_cleaners` |
| Corpus | LJSpeech / AISHELL-3 / LibriTTS | Svarah, IndicTTS Telugu readers + 5 h / 1 h subsampler (`common/data_utils.py`) |
| Training schedule | 900k steps | 150k steps, annealing at 80k/110k/130k (5 h corpus), see `config.yaml` |
| Bug fixes | | `librosa.load(sr=)`; int16 overflow at +32768; stale pitch stats when an alignment is missing; modern librosa/torch APIs; `torch.load(weights_only=False)` |

Config files use the reference's three sections (`preprocess`, `model`, `train`) in one `config.yaml`
plus `data_selection` and `inference`.

## 1. Installation

```bash
conda create -n fs2 python=3.10 -y && conda activate fs2
pip install torch torchaudio            # pick the build for your CUDA version
pip install -r requirements.txt
python -c "import nltk; [nltk.download(x) for x in ('averaged_perceptron_tagger','cmudict','punkt')]"
bash scripts/setup_hifigan.sh           # downloads/unzips generator_universal.pth.tar

# Montreal Forced Aligner lives in its own env (conda only)
conda create -n aligner -c conda-forge montreal-forced-aligner -y
```

## 2. Datasets (you must obtain them)

* **English - Svarah** (AI4Bharat, Indian-accented English, multi-speaker). Set `preprocess.path.corpus_path`
  in `English/config.yaml`. Manifest field names are configurable under `preprocess.corpus`; run
  `python preprocess.py inspect` first and fix keys / speaker rule if needed.
* **Telugu - IndicTTS Telugu** (IIT Madras, requires registration/licence). Set `corpus_path` and the
  speaker folder / transcript paths under `preprocess.corpus` in `Telugu/config.yaml`
  (default assumes `<speaker>/wav/*.wav` + festvox `txt.done.data`). Use one speaker with >= 6 h.

## 3. English - run order

```bash
cd English
python preprocess.py inspect        # verify manifest parsing
python preprocess.py select         # ~5 h train + ~1 h val -> raw_data/, preprocessed_data/Svarah/split.json
conda activate aligner && python preprocess.py align && conda activate fs2   # MFA TextGrids
python preprocess.py features       # mel/pitch/energy/duration, train.txt, val.txt, stats.json
python train.py                     # checkpoints/ + output/log/{train,val}/*.csv  (resume: --restore_step N)
python inference.py                 # -> ../Results/English (audio, duration/pitch/energy plots)
cd .. && python scripts/plot_losses.py English
```
`preprocess.py all` chains select, align and features (run it inside an env where `mfa` is on PATH and the
Python packages are installed, or run stages separately as above).

## 4. Telugu - run order

```bash
cd Telugu
python preprocess.py inspect
python preprocess.py select
python preprocess.py lexicon        # rule-based G2P -> lexicon/telugu-lexicon.txt (MFA dictionary)
conda activate aligner && python preprocess.py align && conda activate fs2   # trains Telugu aligner on the subset
python preprocess.py features
python train.py
python inference.py
cd .. && python scripts/plot_losses.py Telugu
```
Check `mfa train --help` for your MFA version if the `align` stage complains about argument order.

## 5. Test sentences

`English/test_sentences.txt` holds E1, E2a, E2b, E3, E4 exactly as in the PDF (note E3's "rhyth m").
`Telugu/test_sentences.txt` holds T1, T2a, T2b, T3, T4 **repaired by hand** because the PDF text layer is
corrupted; the raw extraction is in `test_sentences_raw_pdf.txt`. **Compare with the original document
before generating final audio.** Custom text: `python inference.py --text "..." --results_dir out/`.

Outputs per sentence ID: `test_generated_audios/<ID>.wav`, `variance_adaptor_plots/{duration,pitch,energy}/<ID>.png`,
and `Results/<Lang>/variance_adaptor_data/<ID>.json` (phones, predicted durations in frames/ms, F0 in Hz, energy)
for writing the report. Pitch is de-normalised with `stats.json`.

## 6. Checkpoints and GitHub

Full training checkpoints (model + optimiser) are ~400 MB. For the repo use
`python scripts/export_model.py English/checkpoints/best_model.pth.tar English/checkpoints/best_model_fp16.pth.tar`
(~70 MB, loads transparently). Otherwise use Git LFS (`.gitattributes` is included) or an external link in this
README. Copy `stats.json` and `speakers.json` (saved automatically into `checkpoints/` by `train.py`) with the
checkpoint: inference needs them.

## 7. Known limitations / decisions to document in the report

* The universal HiFi-GAN is trained on English; Telugu quality may be limited by the vocoder, not the acoustic model.
* Telugu G2P is rule-based (no schwa/sandhi/loanword rules); Telugu number words were written from general
  knowledge - have a native speaker check them.
* Svarah is an ASR corpus (spontaneous, noisy, accented). The SNR filter is a heuristic; tune
  `data_selection.min_snr_db` and look at `selection_report.json`.
* The English split is utterance-level (same speakers in train/val, no shared transcripts or files). Change this if
  your instructor expects unseen speakers in validation.
