"""Shrink a checkpoint for GitHub: keep model weights only, cast to fp16 (~70 MB < 100 MB limit).
Inference casts the weights back to fp32 on load.

    python scripts/export_model.py English/checkpoints/best_model.pth.tar English/checkpoints/best_model_fp16.pth.tar
"""
import sys

import torch

src, dst = sys.argv[1], sys.argv[2]
ckpt = torch.load(src, map_location="cpu", weights_only=False)
sd = {k: (v.half() if v.is_floating_point() else v) for k, v in ckpt["model"].items()}
torch.save({"model": sd, "step": ckpt.get("step")}, dst)
print("saved", dst)
