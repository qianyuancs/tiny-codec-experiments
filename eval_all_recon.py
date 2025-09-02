import os
import csv
import soundfile as sf
from pystoi import stoi
from pesq import pesq

# 设置路径
recon_dir = "results/recon"
data_dir = "data/LibriSpeech/samples"
sr = 16000  # 采样率

# 获取重建文件列表
recon_files = sorted([f for f in os.listdir(recon_dir) if f.endswith(".wav")])

results = []
pesq_list = []
stoi_list = []

for recon_file in recon_files:
    recon_path = os.path.join(recon_dir, recon_file)
    
    # 找对应的原始音频
    # 假设文件名格式是 recon_stepxxxx_sample_X.wav
    sample_name = "sample_" + recon_file.split("_")[-1]
    ref_path = os.path.join(data_dir, sample_name)
    
    if not os.path.exists(ref_path):
        print(f"原音频不存在: {ref_path}, 跳过")
        continue
    
    # 读取音频
    ref, sr_ref = sf.read(ref_path)
    deg, sr_deg = sf.read(recon_path)
    
    # 保证采样率一致
    if sr_ref != sr_deg:
        raise ValueError(f"采样率不一致: {ref_path}, {recon_path}")
    
    # 对齐长度
    min_len = min(len(ref), len(deg))
    ref = ref[:min_len]
    deg = deg[:min_len]
    
    # 计算 PESQ/STOI
    pesq_score = pesq(sr, ref, deg, 'wb')
    stoi_score = stoi(ref, deg, sr, extended=False)
    
    pesq_list.append(pesq_score)
    stoi_list.append(stoi_score)
    
    results.append([recon_file, pesq_score, stoi_score])
    print(f"{recon_file}: PESQ={pesq_score:.3f}, STOI={stoi_score:.3f}")

# 写入 CSV
csv_file = "results/eval_results1000.csv"
with open(csv_file, mode='w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(["file", "PESQ", "STOI"])
    writer.writerows(results)
    # 写平均
    writer.writerow(["average", sum(pesq_list)/len(pesq_list), sum(stoi_list)/len(stoi_list)])

print(f"评估完成，结果保存到 {csv_file}")

