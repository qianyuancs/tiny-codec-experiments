import librosa
import matplotlib.pyplot as plt
import numpy as np

# 写死路径
orig_path = "C:/Users/21978/Desktop/code/warm_up/data/LibriSpeech/samples/sample_0.wav"#原始语音
recon_path = "C:/Users/21978/Desktop/code/warm_up/results/recon/sample_0.wav"#重建语音（opus)
recon_path2=r"C:\Users\21978\Desktop\code\warm_up\results\recon\recon_step000000_sample_28.wav"#DAC重建语音
out_path = "C:/Users/21978/Desktop/code/warm_up/results/fig_sample1.png"

# 读取音频
y_orig, sr = librosa.load(orig_path, sr=16000)
y_recon, _ = librosa.load(recon_path, sr=16000)

# 对齐长度
min_len = min(len(y_orig), len(y_recon))
y_orig = y_orig[:min_len]
y_recon = y_recon[:min_len]

# 绘制波形对比
plt.figure(figsize=(12,6))
plt.subplot(2,1,1)
plt.plot(y_orig)
plt.title("Original waveform")
plt.subplot(2,1,2)
plt.plot(y_recon)
plt.title("Reconstructed waveform(DAC)")
plt.tight_layout()
plt.savefig(out_path)
print(f"图已保存到 {out_path}")
plt.show()

