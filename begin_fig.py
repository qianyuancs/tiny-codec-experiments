import librosa
import matplotlib.pyplot as plt

y, sr = librosa.load(r"C:\Users\21978\Desktop\code\warm_up\data\LibriSpeech\samples\sample_0.wav", sr=16000)
plt.figure(figsize=(10, 3))
plt.plot(y)
plt.title("Waveform Example (16kHz)")
plt.xlabel("Time (samples)")
plt.ylabel("Amplitude")
plt.show()
