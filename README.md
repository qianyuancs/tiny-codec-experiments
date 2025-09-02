# tiny-codec-experiments

##Codec 
语音信号的压缩与重建系统。它的目标是在尽可能小的码率下（低比特率），保留尽量高的语音质量和可懂度。
基本流程：
1.编码（Encode）：把连续的语音波形转化为一系列紧凑的表示（bitstream 或 tokens）；
2.传输/存储：低比特率的数据更节省带宽和存储；
3.解码（Decode）：从压缩表示中重建语音。
传统 Codec：Opus、AMR-WB、MP3 等 → 基于信号处理方法。
神经网络 Codec（Neural Codec）：近年来兴起，如 SoundStream、Encodec、DAC，它们利用 VQ-VAE、GAN 等模型，实现了比传统方法更自然的重建。

本文件是基于Windows11的CPU所运行，支持刚接触codec的朋友练手，通过与ffmpeg与opus经典算法做对比，体验训练DAC模型，调整参数带来的变化，主流评估值计算及其结果可视化分析
