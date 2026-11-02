"""FastSpeech 2 training loop.

Adapted from ming024/FastSpeech2 `train.py` (same model, loss, optimiser,
warm-up + step annealing schedule, gradient clipping/accumulation). Additions:
  * every `log_step` steps the *window-average* of the 6 losses is written to
    <log_path>/train/train_loss.csv, validation losses to val/val_loss.csv
    (used by scripts/plot_losses.py for training_loss.png / validation_loss.png)
  * keeps only the last `keep_last_ckpt` full checkpoints, always keeps
    best_model.pth.tar (lowest validation total loss, model weights only)
  * writes final_model.pth.tar (model weights only) when training finishes
  * returns instead of calling quit()
"""
import argparse
import csv
import os
import re

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from utils.model import get_model, get_vocoder, get_param_num
from utils.tools import to_device, log, synth_one_sample
from model import FastSpeech2Loss
from dataset import Dataset
from evaluate import evaluate

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LOSS_COLS = ["total", "mel", "mel_postnet", "pitch", "energy", "duration"]


def _prune(ckpt_dir, keep):
    files = [f for f in os.listdir(ckpt_dir) if re.fullmatch(r"\d+\.pth\.tar", f)]
    files.sort(key=lambda f: int(f.split(".")[0]))
    for f in files[:-keep]:
        os.remove(os.path.join(ckpt_dir, f))


def train(configs, restore_step=0):
    preprocess_config, model_config, train_config = configs
    print("Prepare training ...")

    dataset = Dataset("train.txt", preprocess_config, train_config, sort=True, drop_last=True)
    batch_size = train_config["optimizer"]["batch_size"]
    group_size = 4  # >1 enables length-sorting inside Dataset.collate_fn
    assert batch_size * group_size < len(dataset), "dataset smaller than a batch group"
    loader = DataLoader(dataset, batch_size=batch_size * group_size, shuffle=True,
                        collate_fn=dataset.collate_fn)

    args = argparse.Namespace(restore_step=restore_step)
    model, optimizer = get_model(args, configs, device, train=True)
    model = nn.DataParallel(model)
    print("Number of FastSpeech2 Parameters:", get_param_num(model))
    Loss = FastSpeech2Loss(preprocess_config, model_config).to(device)
    vocoder = get_vocoder(model_config, device)

    for p in train_config["path"].values():
        os.makedirs(p, exist_ok=True)
    ckpt_dir = train_config["path"]["ckpt_path"]
    train_log_path = os.path.join(train_config["path"]["log_path"], "train")
    val_log_path = os.path.join(train_config["path"]["log_path"], "val")
    os.makedirs(train_log_path, exist_ok=True)
    os.makedirs(val_log_path, exist_ok=True)
    train_logger, val_logger = SummaryWriter(train_log_path), SummaryWriter(val_log_path)

    mode = "a" if restore_step else "w"
    train_csv_f = open(os.path.join(train_log_path, "train_loss.csv"), mode, newline="")
    val_csv_f = open(os.path.join(val_log_path, "val_loss.csv"), mode, newline="")
    train_csv, val_csv = csv.writer(train_csv_f), csv.writer(val_csv_f)
    if not restore_step:
        train_csv.writerow(["step"] + LOSS_COLS)
        val_csv.writerow(["step"] + LOSS_COLS)

    s = train_config["step"]
    grad_acc_step = train_config["optimizer"]["grad_acc_step"]
    grad_clip_thresh = train_config["optimizer"]["grad_clip_thresh"]
    total_step, log_step = s["total_step"], s["log_step"]
    save_step, synth_step, val_step = s["save_step"], s["synth_step"], s["val_step"]
    keep_last = s.get("keep_last_ckpt", 3)
    sr = preprocess_config["preprocessing"]["audio"]["sampling_rate"]

    step, epoch = restore_step + 1, 1
    best_val = float("inf")
    window = torch.zeros(6)
    window_n = 0
    outer_bar = tqdm(total=total_step, desc="Training", position=0)
    outer_bar.n = restore_step
    outer_bar.update()

    while True:
        for batchs in loader:
            for batch in batchs:
                batch = to_device(batch, device)
                output = model(*(batch[2:]))
                losses = Loss(batch, output)

                (losses[0] / grad_acc_step).backward()
                if step % grad_acc_step == 0:
                    nn.utils.clip_grad_norm_(model.parameters(), grad_clip_thresh)
                    optimizer.step_and_update_lr()
                    optimizer.zero_grad()

                window += torch.tensor([l.item() for l in losses])
                window_n += 1

                if step % log_step == 0:
                    avg = (window / window_n).tolist()
                    window.zero_()
                    window_n = 0
                    msg = "Step {}/{}, Total Loss: {:.4f}, Mel Loss: {:.4f}, Mel PostNet Loss: {:.4f}, Pitch Loss: {:.4f}, Energy Loss: {:.4f}, Duration Loss: {:.4f}".format(step, total_step, *avg)
                    with open(os.path.join(train_log_path, "log.txt"), "a") as f:
                        f.write(msg + "\n")
                    outer_bar.write(msg)
                    log(train_logger, step, losses=avg)
                    train_csv.writerow([step] + ["{:.6f}".format(v) for v in avg])
                    train_csv_f.flush()

                if step % synth_step == 0:
                    fig, wav_rec, wav_pred, tag = synth_one_sample(
                        batch, output, vocoder, model_config, preprocess_config)
                    log(train_logger, fig=fig, tag="Training/step_{}_{}".format(step, tag))
                    log(train_logger, audio=wav_rec, sampling_rate=sr,
                        tag="Training/step_{}_{}_reconstructed".format(step, tag))
                    log(train_logger, audio=wav_pred, sampling_rate=sr,
                        tag="Training/step_{}_{}_synthesized".format(step, tag))

                if step % val_step == 0:
                    model.eval()
                    message, val_means = evaluate(model, step, configs, val_logger, vocoder)
                    with open(os.path.join(val_log_path, "log.txt"), "a") as f:
                        f.write(message + "\n")
                    outer_bar.write(message)
                    val_csv.writerow([step] + ["{:.6f}".format(v) for v in val_means])
                    val_csv_f.flush()
                    if val_means[0] < best_val:
                        best_val = val_means[0]
                        torch.save({"model": model.module.state_dict(), "step": step,
                                    "val_total_loss": best_val},
                                   os.path.join(ckpt_dir, "best_model.pth.tar"))
                    model.train()

                if step % save_step == 0:
                    torch.save({"model": model.module.state_dict(),
                                "optimizer": optimizer._optimizer.state_dict()},
                               os.path.join(ckpt_dir, "{}.pth.tar".format(step)))
                    _prune(ckpt_dir, keep_last)

                if step == total_step:
                    torch.save({"model": model.module.state_dict(), "step": step},
                               os.path.join(ckpt_dir, "final_model.pth.tar"))
                    train_csv_f.close()
                    val_csv_f.close()
                    print("Training finished.")
                    return
                step += 1
                outer_bar.update(1)
        epoch += 1
