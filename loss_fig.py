import pandas as pd
import matplotlib.pyplot as plt

# 读取两个训练记录
loss_600 = pd.read_csv(r"results\loss.600.csv")
loss_1000 = pd.read_csv(r"results\loss.csv")
plt.figure(figsize=(8,4))
plt.plot(loss_600['step'], loss_600['loss'], label='600 steps',color='red',linewidth=2)
plt.plot(loss_1000['step'], loss_1000['loss'], label='1000 steps')
plt.xlabel('Step')
plt.ylabel('Loss')
plt.title('Training Loss Comparison')
plt.legend()
plt.grid(True)
plt.show()
