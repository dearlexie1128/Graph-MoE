import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import pandas as pd
import numpy as np


import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import pandas as pd
import numpy as np
def adjust_labels(labels, batch_size):
    num_batches = len(labels) // batch_size
    adjusted_labels = labels[:num_batches * batch_size]
    return adjusted_labels

def loader_SWat(root, batch_size, window_size, stride_size,train_split,label=False):
    data = pd.read_csv(root,low_memory=False)
    data.columns = data.columns.str.strip()
    #将时间戳列转换为 datetime 格式，并将其设置为数据集的索引
    Timestamp = pd.to_datetime(data["Timestamp"])
    data["Timestamp"] = Timestamp
    data = data.set_index("Timestamp")
    #将标签列中的"Normal"标签转换为0，其他标签转换为1
    labels = [ int(l!= 'Normal' ) for l in data["Normal/Attack"].values]
    #将数据中的逗号替换为点号，将标签列删除，并将数据类型转换为浮点数
    for i in list(data): 
        data[i]=data[i].apply(lambda x: str(x).replace("," , "."))

    data = data.drop(["Normal/Attack"] , axis = 1)
    data = data.astype(float)
    n_sensor = len(data.columns)
    #%%
    #选择前51列进行标准化处理，并将标准化后的数据转换为 DataFrame 格式，删除包含缺失值的列
    feature = data.iloc[:,:51]
    scaler = StandardScaler()
    norm_feature = scaler.fit_transform(feature)
    #print("norm_feature",norm_feature)
    norm_feature = pd.DataFrame(norm_feature, columns= data.columns, index = Timestamp)
    #print("norm_feature",norm_feature)
    #shanchu包含缺失值的列
    norm_feature = norm_feature.dropna(axis=1)
    #训练集0.8，验证集0.2，测试集0.2
    print("train_split",train_split)
    train_df = norm_feature.iloc[:int(train_split*len(data))]
    train_label = labels[:int(train_split*len(data))]
    print('trainset size',train_df.shape, 'anomaly ration', sum(train_label)/len(train_label))
    #num_batches = len(labels) // batch_size
    #adjusted_labels = labels[:num_batches * batch_size]
    val_df = norm_feature.iloc[int(0.6*len(data)):int(train_split*len(data))]
    #print('val_df size',val_df.shape)
    val_label = labels[int(0.6*len(data)):int(train_split*len(data))]
    # vnum_batches = len(val_df) // batch_size
    # vadjusted_num_samples = vnum_batches * batch_size
    # val_df = val_df.iloc[:vadjusted_num_samples]
    # print('val_df size',val_df.shape)
    # val_label = val_label[:vadjusted_num_samples]
    
    #val_label = adjust_labels(val_label, batch_size)
    #print("len(val_label)",len(val_label))
    
    test_df = norm_feature.iloc[int(train_split*len(data)):]
    #print('test_df size',test_df.shape)
    test_label = labels[int(train_split*len(data)):]
    # num_batches = len(test_df) // batch_size
    # adjusted_num_samples = num_batches * batch_size
    # test_df = test_df.iloc[:adjusted_num_samples]
    # test_label = test_label[:adjusted_num_samples]
    #print("len(test_label)",len(test_label))
    print('testset size',test_df.shape, 'anomaly ration', sum(test_label)/len(test_label))
   
    train_loader = DataLoader(SWat_dataset(train_df,train_label, window_size, stride_size), batch_size=batch_size, shuffle=False, drop_last=True)
    val_loader = DataLoader(SWat_dataset(val_df,val_label, window_size, stride_size), batch_size=batch_size, shuffle=False, drop_last=True)
    test_loader = DataLoader(SWat_dataset(test_df, test_label, window_size, stride_size), batch_size=batch_size, shuffle=False, drop_last=True)
    # data_num = 0
    # for data111 in test_loader:
    #     data_num = data_num+1
    #     print(data111[0].size())
    # print(len(test_loader.dataset.label))
    #print("data_num", data_num)
    return train_loader, val_loader, test_loader, n_sensor

def loader_SWat_OCC(root, batch_size, window_size, stride_size,train_split,label=False):
    data = pd.read_csv("/home/chenweidong/MTGFLOW-main/Data/input/SWaT_Dataset_Normal_v1.csv",sep = ',', low_memory=False)
    Timestamp = pd.to_datetime(data["Timestamp"])
    data["Timestamp"] = Timestamp
    data = data.set_index("Timestamp")
    labels = [ int(l!= 'Normal' ) for l in data["Normal/Attack"].values]
    for i in list(data): 
        data[i]=data[i].apply(lambda x: str(x).replace("," , "."))
    data = data.drop(["Normal/Attack"] , axis = 1)
    data = data.astype(float)
    n_sensor = len(data.columns)
    #%%
    feature = data.iloc[:,:51]
    scaler = StandardScaler()

    norm_feature = scaler.fit_transform(feature)
    norm_feature = pd.DataFrame(norm_feature, columns= data.columns, index = Timestamp)
    norm_feature = norm_feature.dropna(axis=1)
    train_df = norm_feature.iloc[:]
    train_label = labels[:]
    print('trainset size',train_df.shape, 'anomaly ration', sum(train_label)/len(train_label))

    val_df = norm_feature.iloc[int(train_split*len(data)):]
    val_label = labels[int(train_split*len(data)):]
    #val_label = adjust_labels(val_label, batch_size)
    data = pd.read_csv('/home/chenweidong/MTGFLOW-main/Data/input/SWaT_Dataset_Attack_v0.csv',sep = ';', low_memory=False)
    Timestamp = pd.to_datetime(data["Timestamp"])
    data["Timestamp"] = Timestamp
    data = data.set_index("Timestamp")
    labels = [ int(l!= 'Normal' ) for l in data["Normal/Attack"].values]
    for i in list(data): 
        data[i]=data[i].apply(lambda x: str(x).replace("," , "."))
    data = data.drop(["Normal/Attack"] , axis = 1)
    data = data.astype(float)
    n_sensor = len(data.columns)
 
    feature = data.iloc[:,:51]
    scaler = StandardScaler()
    norm_feature = scaler.fit_transform(feature)
    norm_feature = pd.DataFrame(norm_feature, columns= data.columns, index = Timestamp)
    norm_feature = norm_feature.dropna(axis=1)
    

    test_df = norm_feature.iloc[int(0.8*len(data)):]
    test_label = labels[int(0.8*len(data)):]
    #test_label = adjust_labels(test_label, batch_size)
    #print('testset size',test_df.shape, 'anomaly ration', sum(test_label)/len(test_label))
    
    train_loader = DataLoader(SWat_dataset(train_df,train_label, window_size, stride_size), batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(SWat_dataset(val_df,val_label, window_size, stride_size), batch_size=batch_size, shuffle=False, drop_last=True)
    test_loader = DataLoader(SWat_dataset(test_df,test_label, window_size, stride_size), batch_size=batch_size, shuffle=False, drop_last=True)
    return train_loader, val_loader, test_loader, n_sensor


class SWat_dataset(Dataset):
    def __init__(self, df, label, window_size=60, stride_size=10) -> None:
        super(SWat_dataset, self).__init__()
        self.df = df
        self.window_size = window_size
        self.stride_size = stride_size

        self.data, self.idx, self.label = self.preprocess(df,label)
        self.columns = np.append(df.columns, ["Label"])
        self.timeindex = df.index[self.idx]
        #("self.timeindex",self.timeindex)
        #print("len(self.timeindex)",len(self.timeindex))
    def preprocess(self, df, label):
        #delat_time：计算每个窗口的时间差。
        #idx_mask：检查窗口的时间差是否等于 window_size 秒，过滤掉那些不满足条件的窗口。
        #start_index：满足条件的窗口的起始索引。
        start_idx = np.arange(0,len(df)-self.window_size,self.stride_size)
        end_idx = np.arange(self.window_size, len(df), self.stride_size)
        
        delat_time =  df.index[end_idx]-df.index[start_idx]

        idx_mask = delat_time==pd.Timedelta(self.window_size,unit='s')

        start_index = start_idx[idx_mask]
        #如果窗口内没有异常点，则标签为0；否则标签为1
        label = [0 if sum(label[index:index+self.window_size]) == 0 else 1 for index in start_index ]
        return df.values, start_idx[idx_mask], np.array(label)

    def __len__(self):

        length = len(self.idx)

        return length

    def __getitem__(self, index):
        #  N X K X L X D 
        """
        """
        #print("index:",index)
        start = self.idx[index]
        end = start + self.window_size
        #将数据重塑为 [window_size, number_of_features, 1] 的形状
        data = self.data[start:end].reshape([self.window_size,-1, 1])
        # print("start:",start)
        # start_time = self.timeindex[start].strftime('%Y-%m-%d %H:%M:%S')
        # print("start_time:",start_time)
        #将数据的第0维度和第1维度进行互换,变成[number_of_features, window_size, 1]
        return torch.FloatTensor(data).transpose(0,1), self.label[index],index

