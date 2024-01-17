# -*- coding: utf-8 -*-
"""
Keras implementation of Multi-level Features Guided Capsule Network (MLF-CapsNet).
This file trains a MLF-CapsNet on DEAP/DREAMER dataset with the parameters as mentioned in paper.
We have developed this code using the following GitHub repositories:
- Xifeng Guo's CapsNet code (https://github.com/XifengGuo/CapsNet-Keras)

Usage:
       python capsulenet-multi-gpu.py --gpus 2

"""
from keras import backend as K
from keras import layers, models, optimizers,regularizers
from keras.layers import Lambda, SpatialDropout2D, BatchNormalization, Activation
from capsulelayers import CapsuleLayer, PrimaryCap, Length
import pandas as pd
import time
import pickle
import tensorflow as tf
import numpy as np
import os

import sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

'''
注意：如果直接运行程序的话有时会出现libcudart.so.8.0: cannot open shared object file: No such file or directory
的错误，此时需要：1、source activate tensoflow; 2、 sudo ldconfig /usr/local/cuda-8.0/lib64
完成以上两步即可正常运行；
运行方法：直接ctrl+shift+F10即可.
'''

K.set_image_data_format('channels_last')


def deap_load(data_file, dimension, debaseline):
    rnn_suffix = ".mat_win_128_rnn_dataset.pkl"  # 为cnn数据集准备的文件后缀；
    label_suffix = ".mat_win_128_labels.pkl"  # 这是label的后缀；
    arousal_or_valence = dimension
    with_or_without = debaseline  # 'yes','not'
    dataset_dir = "./deap_shuffled_data/" + with_or_without + "_" + arousal_or_valence + "/"

    # load training set
    with open(dataset_dir + data_file + rnn_suffix, "rb") as fp:
        rnn_datasets = pickle.load(fp)
    with open(dataset_dir + data_file + label_suffix, "rb") as fp:
        labels = pickle.load(fp)
        labels = np.transpose(labels)

    labels = np.asarray(pd.get_dummies(labels), dtype=np.int8)


    # shuffle data
    index = np.array(range(0, len(labels)))
    np.random.shuffle(index)
    rnn_datasets = rnn_datasets[index]  # .transpose(0,2,1)
    labels = labels[index]

    datasets = rnn_datasets.reshape(-1, 128, 32, 1).astype('float32')
    labels = labels.astype('float32')

    return datasets, labels

'''
入口参数  sub：即被试的编号; dimension：即所研究的是那个维度的信息(arousal/dominance); debaseline：即是否删除baseline信号；
'''

def dreamer_load(sub, dimension, debaseline):
    if debaseline == 'yes':  # 即删除baseline信号;
        dataset_suffix = "_rnn_dataset.pkl"
        label_suffix = "_labels.pkl"
        dataset_dir_ecg = './data/dreamer_preprocessed_data_ecg/' + 'yes_' + dimension + '/'
        dataset_dir_eeg = './data/dreamer_preprocessed_data_eeg/' + 'yes_' + dimension + '/'
    else:
        dataset_suffix = "_rnn_dataset.pkl"  # 即未删除baseline信号;
        label_suffix = "_labels.pkl"
        dataset_dir_ecg = './data/dreamer_preprocessed_data_ecg/' + 'no_' + dimension + '/'
        dataset_dir_eeg = './data/dreamer_preprocessed_data_ecg/' + 'no_' + dimension + '/'

    # load training set
    with open(dataset_dir_eeg + sub + dataset_suffix, "rb") as fp:  # 加载trial对应的eeg数据；
        datasets_eeg = pickle.load(fp)
    with open(dataset_dir_ecg + sub + dataset_suffix, "rb") as fp:  # 加载trial对应的eeg数据；
        datasets_ecg = pickle.load(fp)
    with open(dataset_dir_eeg + sub + '_' + dimension + label_suffix, "rb") as fp:  # 加载eeg数据对应的标签；
        labels = pickle.load(fp)
        labels = np.transpose(labels)  # 将标签取转置,本来label的数据是（18,1）.

    labels = labels > 3  # 设置阈值为3,
    labels = np.asarray(pd.get_dummies(labels), dtype=np.int8)


    # shuffle data
    index = np.array(range(0, len(labels)))
    np.random.shuffle(index)
    datasets_eeg = datasets_eeg[index]  # .transpose(0,2,1)
    datasets_ecg = datasets_ecg[index]
    labels = labels[index]

    datasets_eeg = datasets_eeg.reshape(-1, 128, 14, 1).astype('float32')
    datasets_ecg = datasets_ecg.reshape(-1, 256, 2, 1).astype('float32')
    labels = labels.astype('float32')

    return datasets_eeg, datasets_ecg, labels

def tilex(input):
    output = tf.tile(input, multiples=[1, 1, 14, 1])
    return output

def concatenatex(input1):
    output = tf.concat(input1, 3)
    return output

def CapsNet(input_shape_eeg, input_shape_ecg,  n_class, routings, model_version, lam_regularize):
    """
    A Capsule Network .
    :param input_shape: data shape, 3d, [width, height, channels]
    :param n_class: number of classes
    :param routings: number of routing iterations
    :return: Two Keras Models, the first one used for training, and the second one for evaluation.
            `eval_model` can also be used for training.
    """
    # 实例化tensor向量
    x_eeg = layers.Input(shape=input_shape_eeg)
    x_ecg = layers.Input(shape=input_shape_ecg)

    # 扩展ecg维度；-->(?, 256, 28, 1)
    conv1_ecg_tiled = Lambda(tilex)(x_ecg)
    print(conv1_ecg_tiled.shape)

    # 对ecg特征提取/形状变换；-->(?, 128，14，256)
    shaped_ecg = layers.Conv2D(filters=256, kernel_size=2, strides=2, padding='valid', activation='relu', name='conv1',kernel_regularizer=regularizers.l2(lam_regularize))(conv1_ecg_tiled) #kernel_size=9

    # 对eeg特征提取/形状变换；-->(?, 128, 14, 256)
    shaped_eeg = layers.Conv2D(filters=256, kernel_size=1, strides=1, padding='valid', activation='relu', name='conv2',kernel_regularizer=regularizers.l2(lam_regularize))(x_eeg) #kernel_size=9

    # 拼接特征;-->(?, 128, 14, 512)
    conv_concatenated = [shaped_eeg, shaped_ecg]
    conv_concatenated = Lambda(concatenatex)(conv_concatenated)

    # 融合特征特征提取/减通道数;-->(?, 120, 6, 256)
    conv_final = layers.Conv2D(filters=256, kernel_size=9, strides=1, padding='valid', activation='relu', name='conv3',kernel_regularizer=regularizers.l2(lam_regularize))(conv_concatenated) #kernel_size=9

    # 抑制过拟合
    conv_final = layers.SpatialDropout2D(rate=0.6, data_format='channels_last')(conv_final)

    if model_version == 'v0':
        primarycaps = PrimaryCap(conv_final, dim_capsule=8, n_channels=32, kernel_size=9, strides=2, padding='valid',lam_regularize = lam_regularize,model_version =model_version )
    else:
        primarycaps = PrimaryCap(conv_final, dim_capsule=8, n_channels=32, kernel_size=9, strides=1, padding='same',lam_regularize = lam_regularize,model_version =model_version )          #kernel_size=9
        # PrimaryCap的输出：output tensor, shape=[None, num_capsule, dim_capsule]
    # Layer 3: Capsule layer. Routing algorithm works here.
    digitcaps = CapsuleLayer(num_capsule=n_class, dim_capsule=16, routings=routings,
                             name='digitcaps', lam_regularize = lam_regularize)(primarycaps)
    # digitcaps = layers.Dropout(rate=0.5)(digitcaps)

    print(digitcaps)
    print(digitcaps.shape)
    # Layer 4: This is an auxiliary layer to replace each capsule with its length. Just to match the true label's shape.
    # If using tensorflow, this will not be necessary. :)
    out_caps = Length(name='capsnet')(digitcaps)

    # Models for training and evaluation (prediction)
    y = layers.Input(shape=(n_class,))

    train_model = models.Model(input=[x_eeg, x_ecg, y], output=out_caps)
    eval_model = models.Model(input=[x_eeg, x_ecg], output=out_caps)

    return train_model, eval_model

# 损失函数
def margin_loss(y_true, y_pred):
    """
    Margin loss for Eq.(4). When y_true[i, :] contains not just one `1`, this loss should work too. Not test it.
    :param y_true: [None, n_classes]
    :param y_pred: [None, num_capsule]
    :return: a scalar loss value.
    """
    L = y_true * K.square(K.maximum(0., 0.9 - y_pred)) + \
        0.5 * (1 - y_true) * K.square(K.maximum(0., y_pred - 0.1))

    return K.mean(K.sum(L, 1))


def train(model, data, args, fold):
    """
    Training a CapsuleNet
    :param model: the CapsuleNet model
    :param data: a tuple containing training and testing data, like `((x_train, y_train), (x_test, y_test))`
    :param args: arguments
    :return: The trained model
    """
    # unpacking the data
    (x_train_eeg, x_train_ecg, y_train), (x_test_eeg, x_test_ecg, y_test) = data

    # callbacks
    log = callbacks.CSVLogger(args.save_dir + '/' + 'log_fold'+str(fold)+'.csv') #save_dir/log_fold str(fold).csv
    tb = callbacks.TensorBoard(log_dir=args.save_dir + '/tensorboard-logs_fold'+str(fold),
                               batch_size=args.batch_size, histogram_freq=args.debug)
    checkpoint = callbacks.ModelCheckpoint(args.save_dir + '/weights-{epoch:02d}_fold'+str(fold)+'.h5', monitor='val_acc',
                                           save_best_only=True, save_weights_only=True, verbose=1)
    lr_decay = callbacks.LearningRateScheduler(schedule=lambda epoch: args.lr * (1.0 ** epoch))

    #EarlyStop = callbacks.EarlyStopping(monitor='val_capsnet_acc', patience=5)
    # compile the model
    model.compile(optimizer=optimizers.Adam(lr=args.lr),
                  loss=margin_loss,
                  metrics={'capsnet': 'accuracy'})

    # Training without data augmentation:
    # model.fit([x_train_eeg, x_train_ecg, y_train], [y_train, x_train_eeg, x_train_ecg], batch_size=args.batch_size, epochs=args.epochs,
    #           validation_data=[[x_test_eeg, x_test_ecg, y_test], [y_test, x_test_eeg, x_test_ecg]], callbacks=[log, tb, checkpoint, lr_decay])


    '''
    # Training with validation set
    model.fit([x_train, y_train], y_train ,  batch_size=args.batch_size, epochs=args.epochs,verbose = 1,
              validation_split= 0.1 , callbacks=[log, tb, checkpoint, lr_decay])
    '''

    # Training without validation set #开始训练了
    model.fit([x_train_eeg, x_train_ecg, y_train], y_train, batch_size=args.batch_size, epochs=args.epochs,
                # validation_data=(x_test_eeg, x_test_ecg, y_test),
                callbacks=[log, tb, lr_decay])

    #from utils import plot_log
    #plot_log(args.save_dir + '/log.csv', show=True)

    return model



time_start_whole = time.time()

# # ***************  运行deap数据  *****************
# dataset_name = 'deap' # 'deap' # dreamer
# subjects = ['s25']  # ['s01',...,'s21','s22','s23','s24','s26','s28','s29','s30','s31','s32']
# #subjects = ['s01'] #'s01','s02','s03','s04','s05','s06','s07','s08','s09','s10','s11','s12','s13','s14','s15','s16','s17','s18','s19','s20',
# dimensions = ['dominance']
# debaseline = 'yes'  # yes or no

# ***************  运行dreamer数据  *****************
dataset_name = 'dreamer'
subjects = ['1']  # '2','3'...'23',一共23个被试数据,中括号里面放进的被试被分别拿出来训练测试,而不是混在一起打乱测试甚至留一测试;
dimensions = ['arousal']
debaseline = 'yes'  # yes or no

# ***************  下方开始运行主程序  *****************
tune_overfit = 'tune_overfit'
model_version = 'v2'  # v0:'CapsNet', v1:'MLF-CapsNet(w/o)', v2:'MLF-CapsNet'


if __name__ == "__main__":
    print(dimensions)
    print(subjects)
    for dimension in dimensions:
        for subject in subjects:
            import numpy as np
            import tensorflow as tf
            import os
            from keras import callbacks
            from keras.utils.vis_utils import plot_model
            # from keras.utils import multi_gpu_model

            # setting the hyper parameters
            import argparse
            parser = argparse.ArgumentParser(description="Capsule Network on " + dataset_name)
            parser.add_argument('--epochs', default=100, type=int)  # v0:20, v2:40,30 maybe better
            parser.add_argument('--batch_size', default=40, type=int)  # 暂时先修改一下,原始值为100
            parser.add_argument('--lam_regularize', default=0.0, type=float,
                                help="The coefficient for the regularizers")
            parser.add_argument('-r', '--routings', default=3, type=int,
                                help="Number of iterations used in routing algorithm. should > 0")
            parser.add_argument('--debug', default=0, type=int,
                                help="Save weights by TensorBoard")
            parser.add_argument('--save_dir', default='./result_'+ dataset_name + '/sub_dependent_'+ model_version +'/') # other
            parser.add_argument('-t', '--testing', action='store_true',
                                help="Test the trained model on testing dataset")
            parser.add_argument('-w', '--weights', default=None,
                                help="The path of the saved weights. Should be specified when testing")
            parser.add_argument('--lr', default=0.00001, type=float,
                                help="Initial learning rate")  # v0:0.0001, v2:0.00001
            parser.add_argument('--gpus', default=1, type=int)
            args = parser.parse_args()

            print(time.asctime(time.localtime(time.time())))
            print(args)
            if not os.path.exists(args.save_dir):
                os.makedirs(args.save_dir)

            if dataset_name == 'dreamer':          # load dreamer data
                datasets_eeg, datasets_ecg, labels = dreamer_load(subject, dimension, debaseline)
            else:  # load deap data
                datasets, labels = deap_load(subject, dimension, debaseline)

            args.save_dir = args.save_dir + '/' + debaseline + '/' + subject + '_' + dimension + str(args.epochs)
            if not os.path.exists(args.save_dir):
                os.makedirs(args.save_dir)

            fold = 10
            test_accuracy_allfold = np.zeros(shape=[0], dtype=float)
            train_used_time_allfold = np.zeros(shape=[0], dtype=float)
            test_used_time_allfold = np.zeros(shape=[0], dtype=float)
            for curr_fold in range(fold):
                fold_size = datasets_eeg.shape[0] // fold
                indexes_list = [i for i in range(len(datasets_eeg))] #[0...3727]
                #indexes = np.array(indexes_list)
                split_list = [i for i in range(curr_fold * fold_size, (curr_fold + 1) * fold_size)]
                split = np.array(split_list)
                x_test_eeg = datasets_eeg[split]
                x_test_ecg = datasets_ecg[split]
                y_test = labels[split]

                split = np.array(list(set(indexes_list) ^ set(split_list)))
                x_train_eeg = datasets_eeg[split]
                x_train_ecg = datasets_ecg[split]
                y_train = labels[split]

                train_sample = y_train.shape[0]
                print("training examples:", train_sample)
                test_sample = y_test.shape[0]
                print("test examples    :", test_sample)

                # define model
                #with tf.device('/cpu:0'):
                with tf.device('/gpu:0'):
                    model, eval_model = CapsNet(input_shape_eeg=x_train_eeg.shape[1:], input_shape_ecg=x_train_ecg.shape[1:],
                                                                  n_class=len(np.unique(np.argmax(y_train, 1))), #np.unique(np.argmax(y_train, 1))=[0, 1] 表示分类情况就两种，高过label
                                                                  routings=args.routings,
                                                                  model_version= model_version,
                                                                  lam_regularize = args.lam_regularize)
                model.summary()
                eval_model.summary()
                # plot_model(model, to_file=args.save_dir+'/model_fold'+str(curr_fold)+'.png', show_shapes=True)  # 暂时无用

                # define muti-gpu model
                # multi_model = multi_gpu_model(model, gpus=args.gpus)  # 暂时不用这个多GPU***************************

                # train
                train_start_time = time.time()
                # # train(model=multi_model, data=((x_train, y_train), (x_test, y_test)), args=args,fold=curr_fold)
                train(model=model, data=((x_train_eeg, x_train_ecg, y_train), (x_test_eeg, x_test_ecg, y_test)), args=args, fold=curr_fold)
                train_used_time_fold = time.time() - train_start_time
                model.save_weights(args.save_dir + '/trained_model_fold'+str(curr_fold)+'.h5')
                print('Trained model saved to \'%s/trained_model_fold%s.h5\'' % (args.save_dir,curr_fold))
                print('Train time: ', train_used_time_fold)

                #test
                print('-' * 30 + 'fold  ' + str(curr_fold) + '  Begin: test' + '-' * 30)
                test_start_time = time.time()
                y_pred = eval_model.predict([x_test_eeg, x_test_ecg], batch_size=40)  # batch_size = 100,原始值为100,现修改为45.
                test_used_time_fold = time.time() - test_start_time
                test_acc_fold = np.sum(np.argmax(y_pred, 1) == np.argmax(y_test, 1)) / y_test.shape[0]
                # print('shape of y_pred: ',y_pred.shape[0])
                # print('y_pred: ', y_pred)
                # print('y_test: ', y_test)
                print('(' + time.asctime(time.localtime(time.time())) + ') Test acc:', test_acc_fold, 'Test time: ',test_used_time_fold )
                print('-' * 30 + 'fold  ' + str(curr_fold) + '  End: test' + '-' * 30)
                test_accuracy_allfold = np.append(test_accuracy_allfold, test_acc_fold)
                train_used_time_allfold = np.append(train_used_time_allfold, train_used_time_fold)
                test_used_time_allfold = np.append(test_used_time_allfold, test_used_time_fold)

                K.clear_session()

            summary = pd.DataFrame({'fold': range(1,fold+1), 'Test accuracy': test_accuracy_allfold, 'train time': train_used_time_allfold, 'test time': test_used_time_allfold})
            hyperparam = pd.DataFrame({'average acc of 10 folds': np.mean(test_accuracy_allfold), 'average train time of 10 folds': np.mean(train_used_time_allfold), 'average test time of 10 folds': np.mean(test_used_time_allfold),'epochs': args.epochs, 'lr':args.lr, 'batch size': args.batch_size},index=['dimension/sub'])
            writer = pd.ExcelWriter(args.save_dir + '/'+'summary'+ '_'+subject+'.xlsx')
            summary.to_excel(writer, 'Result', index=False)
            hyperparam.to_excel(writer, 'HyperParam', index=False)
            writer.save()
            print('10 fold average accuracy: ', np.mean(test_accuracy_allfold))
            print('10 fold average train time: ', np.mean(train_used_time_allfold))
            print('10 fold average test time: ', np.mean(test_used_time_allfold))


