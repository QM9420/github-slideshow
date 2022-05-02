import  pandas as pd
import matplotlib.pyplot as plt
import numpy as np
time = np.linspace(2004,2020,17)
# print(time)
# path = 'H_CData/%d.xlsx'%(1)  #大缺失
for i in range(1):
    # if(i==0):
    #     id =58
    # else:
    #     id =23
    path =  "new/1/5/4/train_%d_%s.xlsx" % (0,'loss')
    df = pd.read_excel(path)
    data = df.to_numpy()
    # data = data[20,:]
    # z = data[::-1]
    l = len(data)
    print(l)
    time = time[(17-l)::]
    print(len(time))
    print(data)
    # print(z)

    plt.rcParams['font.sans-serif']=['simhei']   # 指定默认字体
    plt.rcParams['axes.unicode_minus']=False   #这两行需要手动设置
    plt.xlabel("时间")
    plt.ylabel("每股收益(单位：元)")
    plt.title("少量缺失情况")
    # plt.text(time, data, data, ha='center', va='bottom', fontsize=10)  # 设置数据标签位置及大小
    if(id==58):
        plt.plot(time,z,markersize=3,alpha= 1,label='有少量缺失')
    else:
        plt.plot(time, z, markersize=3, alpha=1, label='无缺失')
    time = np.linspace(2004, 2020, 17)
plt.legend()
plt.show()