#!/usr/bin/env bash
# Fetch + unzip the pretrained *universal* HiFi-GAN generator shipped in the reference repo.
set -euo pipefail
cd "$(dirname "$0")/../common/hifigan"
if [ ! -f generator_universal.pth.tar ]; then
  curl -L -o generator_universal.pth.tar.zip \
    https://github.com/ming024/FastSpeech2/raw/master/hifigan/generator_universal.pth.tar.zip
  unzip -o generator_universal.pth.tar.zip
  rm generator_universal.pth.tar.zip
fi
ls -lh generator_universal.pth.tar
