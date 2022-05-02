from imblearn.combine import SMOTEENN
from imblearn.over_sampling import BorderlineSMOTE
import pandas as pd
import numpy as np
from LSTM import  *
from collections import Counter
from imblearn.under_sampling import EditedNearestNeighbours


# 数据
# data_path_p = '7/positive.xlsx'  # LSTM_DATA/1.xlsx
# data_path_n = '7/negative.xlsx'
# 读取文件
# df = pd.read_excel(data_path_p)
# data_p = df.to_numpy()
# pheight,pwidth = data_p.shape
# df = pd.read_excel(data_path_n)
# data_n = df.to_numpy()
# nheight,nwidth = data_n.shape
# print("positive:%d  %d"%(pheight,pwidth))  # positive:56  102
# print("negative:%d  %d"%(nheight,nwidth))  # negative:56  333

# 数据
# data_path = 'new/1/4/WHOLE_%d.xlsx' % (2)  # LSTM_DATA/1.xlsx
data_path = 'new/1/5/WHOLE_%d.xlsx' % (21)  # LSTM_DATA/1.xlsx
# 读取文件
df = pd.read_excel(data_path)
data = df.to_numpy()
height,width = data.shape
x,y = create_dataset(data)
# print(dataset_x.shape)
# print(dataset_y.shape)

def write_excel(WDate,index):
    data = pd.DataFrame(WDate)
    write_path = "new/1/5/added_%d.xlsx"%(index)

    writer = pd.ExcelWriter(write_path)
    # 数据写入
    data.to_excel(writer,'page_1',float_format='%.6f',index=False)
    writer.save()
    writer.close()
    print('%d文件已完成写入'%index)


# smote_enn = SMOTEENN(random_state=0)
# X, Y = smote_enn.fit_resample(x, y)
# X =np.array(X)
# Y = np.array(Y)
# X = np.transpose(X)
# print(X.shape)
# Y = Y.reshape(1,-1)
# print(Y.shape)
# data = np.vstack((X,Y))

# print(data.shape)
# write_excel(data,1)
# print (sorted(Counter(Y).items()))  #[(0.0, 297), (1.0, 327)]
# print(X.shape)
# print(Y.shape)

# from imblearn.under_sampling import EditedNearestNeighbours
# enn = EditedNearestNeighbours(random_state=0)
# X,Y= enn.fit_sample(X, y)


smote_enn = SMOTEENN(random_state=4)
X, Y = smote_enn.fit_resample(x, y)
print (sorted(Counter(Y).items()))

# bsmote = BorderlineSMOTE(kind='borderline-1',random_state=2)
# X,Y = bsmote.fit_resample(x,y)
# print (sorted(Counter(Y).items()))
from imblearn.combine import SMOTETomek
smote_tomek = SMOTETomek(random_state=0)
X,Y= smote_tomek.fit_sample(X, Y)

# enn = EditedNearestNeighbours()
# X,Y = enn.fit_resample(X, Y)

print (sorted(Counter(Y).items()))
X =np.array(X)
Y = np.array(Y)
X = np.transpose(X)
print(X.shape)
Y = Y.reshape(1,-1)
print(Y.shape)
data = np.vstack((X,Y))
print(data.shape)
# write_excel(data,22)