import xgboost as xb
import  numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import SelectFromModel

def feature_rank():

    # 合成路径`
    i = 1
    data_path = 'new/1/5/WHOLE_%d.xlsx' % (i)  # LSTM_DATA/1.xlsx

    # 读取文件
    df = pd.read_excel(data_path)
    data = df.to_numpy()
    height,width = data.shape
    x = data[0:height-1,:]
    x = np.transpose(x)
    print(x.shape)
    y = data[-1,:]
    # y = y.astype(np.int)
    print(y)

    # XGBOOST 模型
    model = xb.XGBClassifier(eval_metric='error',)#eval_metric='error',objective ='binary:logitraw'
    model.fit(x,y)

    print(model.feature_importances_)
    # plt.bar(range(len(model.feature_importances_)),model.feature_importances_)
    xb.plot_importance(model,importance_type='weight') # cover gain weight
    plt.show()

feature_rank()
# 写入函数
def write_excel(WDate,index):
    data = pd.DataFrame(WDate)
    write_path = 'Data_5/del_data_%d.xlsx'%(index)
    writer = pd.ExcelWriter(write_path)
    # header参数表示列的名称，index表示行的标签
    data.to_excel(writer,'page_1',float_format='%.6f',index=False)
    writer.save()
    writer.close()
    print('%d文件已完成写入'%index)


def data_del():   # 2 8 22 23
    # 合成路径`
    i = 1
    data_path = 'Data_5/del_data_%d.xlsx' % (i)  # LSTM_DATA/1.xlsx

    # 读取文件
    df = pd.read_excel(data_path)
    data = df.to_numpy()
    height, width = data.shape
    # print(data.shape) (33, 497)
    data = np.delete(data,[1,7,21,22,25 ],axis=0)
    # print(data.shape) (24, 497)
    write_excel(data,1)
# data_del()

def test():

    # 合成路径`
    i = 1
    # data_path = 'Data_6/WHOLE_%d.xlsx' % (i)  # LSTM_DATA/1.xlsx
    # data_path = 'new/1/5/added_%d.xlsx' % (1)  # LSTM_DATA/1.xlsx
    data_path = 'new/1/5/WHOLE_%d.xlsx' % (1)  # LSTM_DATA/1.xlsx
    # 读取文件
    df = pd.read_excel(data_path)
    data = df.to_numpy()
    height,width = data.shape
    x = data[0:height-1,:]
    x = np.transpose(x)
    # print(x.shape)
    y = data[-1,:]
    # y = y.astype(np.int)
    # print(y.shape)

    # split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3,random_state=1)

    # XGBOOST 模型
    model = xb.XGBClassifier()
    eval_set = [(X_test,y_test)]
    print(X_train.shape)
    print(y_train.shape)
    model.fit(X_train, y_train,eval_metric="error")#early_stopping_rounds=10 ,, eval_set=eval_set, verbose=True,
    y_pred = model.predict(X_test)
    predictions = [round(value) for value in y_pred]
    accuracy = accuracy_score(y_test, predictions)
    print("Accuracy: %.2f%%" % (accuracy * 100.0))

    # xb.plot_importance(model,height=0.5,importance_type='cover',max_num_features=64) #cover gain weight
    #
    # xb.plot_importance(model, height=0.5, importance_type='gain', max_num_features=64)  # cover gain weight

    xb.plot_importance(model, height=0.5, importance_type='weight', max_num_features=64)  # cover gain weight
    plt.show()
# test()


