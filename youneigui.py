import pandas as pd
import  numpy as np


def wrong_id(id = 0):
    # 数据
    data_path = '7/w%d.xlsx' % (id)  # LSTM_DATA/1.xlsx

    # 读取文件
    df = pd.read_excel(data_path)
    data = df.to_numpy()
    # height, width = data.shape
    return data

def true_data(id):
    # 数据
    data_path = '7/%d.xlsx' % (id)  # LSTM_DATA/1.xlsx

    # 读取文件
    df = pd.read_excel(data_path)
    data = df.to_numpy()
    height, width = data.shape
    return data[:height-1,:]

def test_data(id):
    # 数据
    data_path = '7/test%d.xlsx' % (2)  # LSTM_DATA/1.xlsx

    # 读取文件
    df = pd.read_excel(data_path)
    data = df.to_numpy()
    height, width = data.shape
    return data[:height-2,id]

wrong = wrong_id()
# print(wrong)

# 长度
lens = 115
wrong = wrong[:115]
tj,tk = 0, 0

for i in range(lens):
    wid = wrong[i]-1
    wid = np.int(wid)
    target_data = test_data(wid)
    target_data = target_data[:27]
    flag = 0
    # print(target_data.shape)
    for j in range(1,63):
        t_data = true_data(j)
        height,width = t_data.shape
        tj = j
        for k in range(width):
            temp = t_data[:,k]
            # print(temp.shape)
            tk = k
            if( (temp == target_data).all()):
                flag = 1
                break
        if(flag == 1):
            break
    print("%d的位置是%d.xlsx的第%d行"%(wid,tj,tk))


