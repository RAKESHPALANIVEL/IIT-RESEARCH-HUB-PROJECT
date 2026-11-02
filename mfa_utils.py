"""Thin wrapper around the Montreal Forced Aligner CLI (MFA >= 3.0)."""
import shutil
import subprocess


def run(cmd):
    print("$ " + " ".join(cmd))
    if shutil.which(cmd[0]) is None:
        raise SystemExit(
            "\n`{}` not found on PATH. Install Montreal Forced Aligner first, e.g.\n"
            "  conda create -n aligner -c conda-forge montreal-forced-aligner\n"
            "  conda activate aligner\n"
            "then re-run this stage (or run the command above yourself).".format(cmd[0]))
    subprocess.run(cmd, check=True)
