import os, soundfile as sf, csv
import numpy as np
from pystoi import stoi
from pesq import pesq

ref_dir = "data/LibriSpeech/samples"
deg_dir = "results/recon"
out_csv = "results/metrics.csv"

# 确保输出目录存在
os.makedirs(os.path.dirname(out_csv), exist_ok=True)

rows = []
processed_count = 0

for fn in sorted(os.listdir(ref_dir)):
    if not fn.endswith(".wav"): 
        continue
        
    ref_path = os.path.join(ref_dir, fn)
    deg_path = os.path.join(deg_dir, fn)
    
    print(f"\n处理文件: {fn}")
    
    # 检查处理后的文件是否存在
    if not os.path.exists(deg_path):
        print(f"警告: {deg_path} 不存在，跳过")
        continue
        
    try:
        # 读取音频文件
        ref, sr = sf.read(ref_path)
        deg, sr2 = sf.read(deg_path)
        
        print(f"采样率: 参考={sr}, 处理={sr2}")
        print(f"音频长度: 参考={len(ref)}, 处理={len(deg)}")
        print(f"音频形状: 参考={ref.shape}, 处理={deg.shape}")
        
        # 确保音频是单声道
        if len(ref.shape) > 1:
            print("参考音频是多声道，转换为单声道")
            ref = ref[:, 0]
        if len(deg.shape) > 1:
            print("处理音频是多声道，转换为单声道")
            deg = deg[:, 0]
            
        # 确保长度一致
        min_len = min(len(ref), len(deg))
        if min_len == 0:
            print("警告: 音频长度为0，跳过")
            continue
            
        ref = ref[:min_len]
        deg = deg[:min_len]
        
        # 确保采样率一致
        if sr != sr2:
            print(f"警告: 采样率不一致 {sr} vs {sr2}")
        
        # 计算PESQ
        try:
            p = pesq(sr, ref, deg, 'wb')   # PESQ
            print(f"PESQ计算成功: {p}")
        except Exception as e:
            print(f"PESQ计算失败: {str(e)}")
            p = float('nan')
        
        # 计算STOI
        try:
            s = stoi(ref, deg, sr)         # STOI
            print(f"STOI计算成功: {s}")
        except Exception as e:
            print(f"STOI计算失败: {str(e)}")
            s = float('nan')
        
        rows.append([fn, p, s])
        processed_count += 1
        print(f"文件处理完成")
        
    except Exception as e:
        print(f"处理文件 {fn} 时出错: {str(e)}")
        import traceback
        traceback.print_exc()
        continue

# 写入CSV文件
with open(out_csv, 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(["file", "PESQ", "STOI"])
    writer.writerows(rows)

print(f"\n评测完成，处理了 {processed_count} 个文件，结果保存在 {out_csv}")

# 如果有处理文件但CSV为空，检查数据
if processed_count > 0 and len(rows) == 0:
    print("警告: 处理了文件但没有结果数据")
    print("rows内容:", rows)


