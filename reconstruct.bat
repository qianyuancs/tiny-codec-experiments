@echo off
mkdir results\recon

for %%f in (data\LibriSpeech\samples\sample_*.wav) do (
  echo Processing %%f ...
  ffmpeg -y -i "%%f" -c:a libopus -b:a 24k tmp.opus
  ffmpeg -y -i tmp.opus -ar 16000 -ac 1 "results\recon\%%~nxf"
)

del tmp.opus

