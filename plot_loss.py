import  pandas as pd
import matplotlib.pyplot as plt
import numpy as np
id = 1
path =  "new/1/5/4/train_%d_%s_%s.xlsx" % (id,'test23_LR','loss')  # 1-1  0-3
df = pd.read_excel(path)
data = df.to_numpy()
# data = data[:399]
# data = data[20,:]
# z = data[::-1]
l = len(data)
print(l)
index = np.linspace(1,l,l)
# print(index)
plt.subplots(1,1,figsize=(14,7),)
plt.subplots_adjust(hspace=0.5)

font1 = {'family' : 'Times New Roman',
'weight' : 'normal',
'size'   : 15,
}
font2 = {'weight' : 'normal',
'size'   : 26,
}
font3 = {'weight' : 'normal',
'size'   : 15,
}
plt.subplot(1,1,1)
plt.plot(index,data)
plt.rcParams['font.sans-serif']=['simhei']   # 指定默认字体
plt.rcParams['axes.unicode_minus']=False   #这两行需要手动设置


plt.xlabel("EPOCH",font2)
plt.ylabel("loss",font2)
plt.title("LSTM神经网络在指定参数下的loss",font2)

plt.tick_params(labelsize=18)
# plt.legend(prop=font3)
plt.show()