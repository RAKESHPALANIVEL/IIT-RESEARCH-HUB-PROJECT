"""Config loading shared by English/ and Telugu/.

Each language folder has ONE config.yaml with four sections:
  preprocess      -> same schema as ming024/FastSpeech2 `preprocess.yaml`
  model           -> same schema as `model.yaml`
  train           -> same schema as `train.yaml`
  data_selection  -> (new) 5 h train / 1 h validation subsampling rules
All relative paths in the config are relative to the language folder, so every
script does os.chdir(<language folder>) before loading it.
"""
import os
import sys
import yaml

COMMON_DIR = os.path.dirname(os.path.abspath(__file__))


def setup_paths(language_dir):
    """Make `text` resolve to the language-specific front end and the rest
    (model, transformer, audio, hifigan, utils, dataset ...) to common/."""
    language_dir = os.path.abspath(language_dir)
    for p in (COMMON_DIR, language_dir):  # language_dir ends up first
        if p in sys.path:
            sys.path.remove(p)
        sys.path.insert(0, p)
    os.chdir(language_dir)
    return language_dir


def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg


def split_config(cfg):
    return cfg["preprocess"], cfg["model"], cfg["train"]
