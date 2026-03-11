param()

$ErrorActionPreference = "Stop"

pyinstaller `
  --noconfirm `
  --clean `
  --windowed `
  --name FinalProjectVideoCutter `
  --paths src `
  --collect-data whisper `
  --collect-submodules final_project `
  --hidden-import whisper `
  --hidden-import faster_whisper `
  --hidden-import ffmpeg `
  --hidden-import librosa `
  --hidden-import cv2 `
  --hidden-import natasha `
  --hidden-import ruaccent `
  --hidden-import transformers `
  --hidden-import moviepy.video.fx.HeadBlur `
  --copy-metadata imageio `
  --copy-metadata imageio-ffmpeg `
  --copy-metadata moviepy `
  src\final_project\gui.py
