import torchaudio, os
from pathlib import Path

out_dir = Path("原始wav文件下载")
out_dir.mkdir(parents=True, exist_ok=True)

# 下载 test-clean （干净的小测试集）
ds = torchaudio.datasets.LIBRISPEECH("./data", url="test-clean", download=True)

# 保存前 50个样本
for i in range(50):
    waveform, sample_rate, _, _, _, _ = ds[i]
    wav = waveform.mean(0, keepdim=True)  # 转单声道
    wav = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)(wav)
    torchaudio.save(str(out_dir / f"sample_{i}.wav"), wav, 16000)

print("数据准备完成")

