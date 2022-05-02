import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# 数据
data_path = 'new/1/4/WHOLE_%d.xlsx' % (0)  # LSTM_DATA/1.xlsx
# data_path = '7/added_%d.xlsx' % (1)  # LSTM_DATA/1.xlsx
# 读取文件
df = pd.read_excel(data_path)
data = df.to_numpy()
data = np.transpose(data)
data = np.delete(data,[14,15],1)
df = pd.DataFrame(data)
height,width = data.shape
print(data.shape)
fid = np.linspace(0,25,26)
F = []
for j in range(26):
    F.append( 'f'+np.str(np.int(fid[j])))
# F = np.array(F)
# # print(F)
# print(df)
new_df = df.corr()
import seaborn as sns
#引入seaborn库
plt.figure(1)

ax = sns.heatmap(new_df,annot=False, linewidths=2, square=True,xticklabels=F,yticklabels=F)#绘制new_df的矩阵热力图
label_y = ax.get_yticklabels()
plt.show()#显示图片
print(1)    # 提出的是2 5 6