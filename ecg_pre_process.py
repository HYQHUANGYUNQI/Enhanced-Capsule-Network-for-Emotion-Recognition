import scipy.io as sio
import os
import numpy as np
import time
import pickle
import re
import argparse
import sys
import pandas as pd
from sklearn import preprocessing

np.random.seed(0)

def norm_dataset(dataset_1D):
    norm_dataset_1D = preprocessing.normalize(dataset_1D, norm='l2', axis=1)
    return norm_dataset_1D


def windows(data, size): #
    start = 0
    while ((start+size) < data.shape[0]):  
        yield int(start), int(start + size)  
        start += size


def segment_signal(data, label, label_index, window_size):
    # get data file name and label file name
    for (start, end) in windows(data, window_size):
        # print(data.shape)
        if ((len(data[start:end]) == window_size)):
            if (start == 0):
                segments = data[start:end]
                segments = np.vstack([segments, data[start:end]])

                labels = np.array(label[label_index])
                labels = np.append(labels, np.array(label[label_index]))
            else:
                segments = np.vstack([segments, data[start:end]])
                labels = np.append(labels, np.array(label[label_index]))  # labels = np.append(labels, stats.mode(label[start:end])[0][0])
    return segments, labels

def segment_baseline(base, window_size):
    # get data file name
    for (start, end) in windows(base, window_size):
        # print(data.shape)
        if ((len(base[start:end]) == window_size)):
            if (start == 0):
                segments_ = base[start:end]
                t1 = base[start:end]
                segments_ = np.vstack([segments_, base[start:end]])
            else:
                t2 = base[start:end]
                segments_ = np.vstack([segments_, base[start:end]])
        print(segments_)
    return segments_

if __name__ == '__main__':
    begin = time.time()
    print("time begin:", time.localtime())
    label_class = "arousal"  # sys.argv[1]     # arousal/valence/dominance
    subs = ['1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20','21','22','23']  # sys.argv[2]
    for sub in subs:  # 取出的是被试的编号,为字符串类型,这样便于后面和其他字符串进行拼接;
        # debase = 'no'  # 这里控制是否需要删除baseline信号;
        debase = 'yes'  # 这里控制是否需要删除baseline信号;
        dataset_dir = './data/dreamer_data_ecg/'
        dataset_dir1 = dataset_dir+'stimuli(m,2)/'+ 'ecg' + sub+'_stimuli'
        data_file1 = sio.loadmat(dataset_dir1 + ".mat" )
        dataset_dir2 = dataset_dir+'label_'+label_class+'(18,1)/'+sub+'_'+label_class+'_label'
        data_file2 = sio.loadmat(dataset_dir2 + ".mat")
        dataset_dir3 = dataset_dir+'baseline(m,2)/'+ 'ecg' + sub+'_baseline'
        data_file3 = sio.loadmat(dataset_dir3 + ".mat")
        output_dir = './data/dreamer_preprocessed_data_ecg/' + debase + '_'+label_class + '/'


        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        window_size = 256  # 采样率为256Hz，取1s时间窗长；

        # load label
        # label_in = data_file2['label']  # 加载数据的标签；
        label_in = data_file2['data_label']  # 加载数据的标签；

        label_inter = np.empty([0])  # 建立一个空的标签
        # data_inter_cnn = np.empty([0, window_size, 9, 9])  # 被试的数量不确定,将其维度设定为0,另外cnn输入的为图片,是三维输入,所以将其转化为(128,(9,9));
        data_inter_rnn = np.empty([0, window_size, 2])  # rnn输入的为线性时间序列,故直接就是集合128帧14个通道的数据作为一个样本；
        baseline_inter = np.empty([0, window_size, 2])  # baseline信号一样,即第一个维度控制了样本的个数,无论是将trial信号还是baseline信号平均分割成多少样本
        base_signal_ = np.zeros([window_size, 2])  # 建立baseline信号的中间变量;
        base_signal = np.empty([18, window_size, 2])  # 实际上真正需要的极限信号是18个,即每段trial信号对应一个时长1秒的baseline信号；
        # load dataset
        dataset_in = data_file1
        '''
        读入的Matlab格式的数据data_file1.keys()包含dict_keys(['__header__', '__version__', '__globals__', 'data_stimuli'])
        即真正需要的数据是存在'data_stimuli'下的,而只有它是不以双下划线开头的；
        '''
        key_list = [key for key in data_file1.keys() if key.startswith('__') == False]  # 列表解析 #读取data_stimuli这一项
        print(key_list) #打印出读取到的列表 #['data_stimuli1', 'data_stimuli10', 'data_stimuli11', 'data_stimuli12', 'data_stimuli13', 'data_stimuli14', 'data_stimuli15', 'data_stimuli16', 'data_stimuli17', 'data_stimuli18', 'data_stimuli2', 'data_stimuli3', 'data_stimuli4', 'data_stimuli5', 'data_stimuli6', 'data_stimuli7', 'data_stimuli8', 'data_stimuli9']
        key_list_rearrange = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  # 将键值列表重排列,一共18个,算是初始化;
        for key in key_list: #遍历刚刚读取到的列表中的key
            index = int(re.findall('\d+', key)[0]) - 1  # re是匹配字符串的模块,\d即匹配数字；findall()的用法，可传两个参数；
            key_list_rearrange[index] = key #？
        print(key_list_rearrange) #重新排序这个键名 #['data_stimuli1', 'data_stimuli2', 'data_stimuli3', 'data_stimuli4', 'data_stimuli5', 'data_stimuli6', 'data_stimuli7', 'data_stimuli8', 'data_stimuli9', 'data_stimuli10', 'data_stimuli11', 'data_stimuli12', 'data_stimuli13', 'data_stimuli14', 'data_stimuli15', 'data_stimuli16', 'data_stimuli17', 'data_stimuli18']


        key_list2 = [key for key in data_file3.keys() if key.startswith('__') == False]
        print(key_list2) #['data_baseline1', 'data_baseline10', 'data_baseline11', 'data_baseline12', 'data_baseline13', 'data_baseline14', 'data_baseline15', 'data_baseline16', 'data_baseline17', 'data_baseline18', 'data_baseline2', 'data_baseline3', 'data_baseline4', 'data_baseline5', 'data_baseline6', 'data_baseline7', 'data_baseline8', 'data_baseline9']
        key_list_rearrange2 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        for key in key_list2:
            index = int(re.findall('\d+', key)[0]) - 1
            key_list_rearrange2[index] = key
        print(key_list_rearrange2) #['data_baseline1', 'data_baseline2', 'data_baseline3', 'data_baseline4', 'data_baseline5', 'data_baseline6', 'data_baseline7', 'data_baseline8', 'data_baseline9', 'data_baseline10', 'data_baseline11', 'data_baseline12', 'data_baseline13', 'data_baseline14', 'data_baseline15', 'data_baseline16', 'data_baseline17', 'data_baseline18']

        #把baseline数据由（15616, 2）--> (61, 256, 2)，再vstack
        for key in key_list_rearrange2:
            if key.startswith('__') == False:  # if array is EEG then do as follow
                print("Processing ",  key, "..........") #Processing  data_baseline1 ..........
                baseline = data_file3[key]
                label_index = int(re.findall('\d+', key)[0]) - 1  # get number from str key
                # data = norm_dataset(data)  # normalization
                print('shape of this ECG: ', baseline.shape) #shape of this EEG:  (15616, 14)
                baseline = segment_baseline(baseline, window_size)
                base_segment_set = baseline.reshape(int(baseline.shape[0] / window_size), window_size, 2) #（15616, 2）--> (61, 256, 2)
                print('segment number of this ECG: ', base_segment_set.shape[0])
                baseline_inter = np.vstack([baseline_inter, base_segment_set]) # shape: (1098, 256, 2)
                print(baseline_inter.shape)

        for k in range(1, 19): #18个trial
            for i in range(1, 62):
                base_signal_ += baseline_inter[k * i - 1] #baseline_inter[0]+[1]+[2]+...[61] ; shape:(256,2)
            base_signal_ = base_signal_ // 61
            base_signal[k-1] = base_signal_
        print(base_signal.shape) #每个trial对应1s的基线信号

        count = 1
        # traversing 18 EEGs of one experiment/session 在一个experiment中遍历18个trial
        #去除基线 + 标准化 + 去掉最后一秒信号 + shape：(?, 256, 2) + 把18个片段合在一起
        for key in key_list_rearrange:
             if key.startswith('__') == False:  # if array is EEG than do as follow
              print("Processing ", count, key, "..........")
              count = count + 1
              data = dataset_in[key]
              data = data.astype(float)
              label_index = int(re.findall('\d+', key)[0]) - 1  # get number from str key

              if debase =='yes':
                  for m in range(0, data.shape[0]//256):
                      data[m * 256:(m + 1) * 256, 0:2] = data[m * 256:(m + 1) * 256, 0:2] - base_signal[label_index] #除去基线信号
              data = norm_dataset(data)  # normalization
              print('shape of this ECG: ', data.shape)
              data, label = segment_signal(data, label_in, label_index, window_size)
              
              data_rnn = data.reshape(int(data.shape[0] / window_size), window_size, 2)
              # append new data and label
              data_inter_rnn = np.vstack([data_inter_rnn, data_rnn])
              label_inter = np.append(label_inter, label)
              #print("total cnn size:", data_inter_cnn.shape)
              print("total rnn size:", data_inter_rnn.shape)
              print("total label size:", label_inter.shape)

        # shuffle data
        # index = np.array(range(0, len(label_inter)))
        # np.random.shuffle(index)
        # #shuffled_data_cnn = data_inter_cnn[index]
        # shuffled_data_rnn = data_inter_rnn[index]
        # shuffled_label = label_inter[index]

        #output_data_cnn = output_dir + sub + "_cnn_dataset.pkl"
        output_data_rnn = output_dir + sub + "_rnn_dataset.pkl"
        output_label = output_dir + sub + '_'+label_class+'_labels.pkl'  # arousal/valence/dominance

        #with open(output_data_cnn, "wb") as fp:
            #pickle.dump(shuffled_data_cnn, fp, protocol=4)
        with open(output_data_rnn, "wb") as fp:
            pickle.dump(data_inter_rnn, fp, protocol=4)
        with open(output_label, "wb") as fp:
            pickle.dump(label_inter, fp)
        # break
        end = time.time()
        print("end time:", time.asctime(time.localtime(time.time())))
        print("time consuming:", (end - begin))