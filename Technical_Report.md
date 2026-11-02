---
title: "FastSpeech 2 Text-to-Speech for English and Telugu - Technical Report"
author: "<RAKESH P>"
date: "<date>"
---
<!-- Fill every <...> with your real results, then export: pandoc Technical_Report.md -o Technical_Report.pdf -->

# 1. Implementation approach
* Reference: FastSpeech 2 (Ren et al., ICLR 2021) and ming024/FastSpeech2 (commit <hash>).
* Components reused unchanged: <encoder/decoder FFT blocks, variance adaptor, length regulator, post-net, loss, optimiser, HiFi-GAN>.
* Modifications for English: <Svarah reader, text normaliser, subsampler, MFA 3 handling, bug fixes (see README table)>.
* Modifications for Telugu: <rule-based G2P and phone inventory, number verbalisation, cleaners, symbol table, MFA trained on subset>.

# 2. Data
| | English (Svarah) | Telugu (IndicTTS) |
|---|---|---|
| Selection / subsampling strategy | <from selection_report.json> | <...> |
| Speakers used | <n> | <n> |
| Train hours / utterances | <h / n> | <h / n> |
| Validation hours / utterances | <h / n> | <h / n> |
| Overlap check | none (split.json, disjoint ids and transcripts) | none |
| Preprocessing | 22.05 kHz, 80-mel, hop 256, phoneme-level F0/energy, z-norm | same |
| Alignment | MFA english_us_arpa | MFA trained on subset (<model>) |
| Training configuration | <batch, steps, lr schedule, GPU, wall-clock> | <...> |

# 3. Loss curves
<Insert Results/<Lang>/training_loss.png and validation_loss.png. Discuss: convergence step, minimum validation step, train/val gap, spikes or instability, signs of overfitting (val loss rising while train falls), per-component behaviour (mel, pitch, energy, duration).>

# 4. Generated speech (per test sentence)
| ID | Pronunciation | Overall quality (1-5 or MOS-style) | Notes |
|---|---|---|---|
| E1 | <long OOV word antidisestablishmentarianism> | | |
| E2a / E2b | <"conduct" noun vs verb: does stress change?> | | |
| E3 | <split token "rhyth m"> | | |
| E4 | <time, date, year, OOV "andaroo"> | | |
| T1 | <long loanword conjuncts> | | |
| T2a / T2b | <"పట్టు" grip vs silk: does context disambiguate?> | | |
| T3 | | | |
| T4 | <digits, ordinal, time verbalisation> | | |

# 5. Variance-adaptor analysis
For each sentence insert `duration`, `pitch`, `energy` plots and explain: <phone-level duration of long vs short vowels, geminates, pauses at commas; F0 declination / final fall / question rise; energy peaks on stressed syllables; differences between the two readings of E2 and T2; failures>.

# 6. English vs Telugu
<Comparison of data, front end, alignment quality, loss levels, naturalness, failures (heteronyms, OOV, numbers), limitations (G2P rules, vocoder, single vs multi-speaker, small data) and improvements (better alignment, Telugu vocoder fine-tuning, larger lexicon/G2P, more data, speaker-balanced validation).>

# 7. Reproducibility
* Software: Python <ver>, torch <ver>, librosa <ver>, MFA <ver> (see requirements.txt, README).
* Commands: see README sections 3 and 4 (exact commands used: <paste>).
* Config files: `English/config.yaml`, `Telugu/config.yaml`; seeds: data_selection.seed=1234.
* Checkpoints: <location / LFS / link>.
