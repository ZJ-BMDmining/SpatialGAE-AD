from __future__ import division

import pickle as pkl
import numpy as np
import pandas as pd
import sys
import time
import os
from sklearn import metrics
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.metrics.cluster import adjusted_rand_score
import scipy
import scipy.stats

from keras.callbacks import EarlyStopping, TensorBoard, ModelCheckpoint
from keras.layers import Input, Dropout, Lambda
from keras.models import Model
from keras.optimizers import Adam
from keras.regularizers import l2, l1
from keras import backend as K
import keras
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from graph_attention_layer import GraphAttention
from sklearn.model_selection import train_test_split
from keras.callbacks import ModelCheckpoint
import argparse
import tensorflow as tf
import csv


parser = argparse.ArgumentParser()

parser.add_argument('--dataset_str',default='ACT',type=str, help='name of dataset, ROSMAP MSBB ACT Mayo ANM1 ANM2')
parser.add_argument('--phenotype',default= "Abeta_IHC",type=str, help='CERAD Braak Thal Abeta_IHC Tau_IHC patient')
parser.add_argument('--isCCA',default=True,type=bool, help='use CCA fusion or not')
parser.add_argument('--F1', default=100, type=int, help='number of neurons in the 1-st layer of encoder')
parser.add_argument('--F2', default=50, type=int, help='number of neurons in the 2-nd layer of encoder')
parser.add_argument('--n_attn_heads', default=4, type=int, help='number of heads for attention')
parser.add_argument('--dropout_rate', default=0.4, type=float, help='dropout rate of neurons in autoencoder')
parser.add_argument('--l2_reg', default=0, type=float, help='coefficient for L2 regularizition')
parser.add_argument('--learning_rate', default=5e-4, type=float, help='learning rate for training')
parser.add_argument('--pre_lr', default=2e-4, type=float, help='learning rate for pre-training')
parser.add_argument('--pre_epochs', default=1000, type=int, help='number of epochs for pre-training')
parser.add_argument('--epochs', default=3000, type=int, help='number of epochs for pre-training')
parser.add_argument('--c1', default=1, type=float, help='weight of reconstruction loss')
parser.add_argument('--c2', default=1, type=float, help='weight of clustering loss')
parser.add_argument('--k', default=None, type=int, help='number of neighbors to construct the cell graph')

args = parser.parse_args()
phenotype = args.phenotype

if args.k == 1:
    dropout_rate = 0. # To avoid absurd results
else:
    dropout_rate = args.dropout_rate

if not os.path.exists('logs/'):
    os.makedirs('logs/')
if not os.path.exists('result/'):
    os.makedirs('result/')


args = parser.parse_args()

if not os.path.exists('result/'):
    os.makedirs('result/')
    
if args.dataset_str == "ROSMAP":
    data_name =  "ROSMAP_4738genes"
    if args.phenotype == "CERAD":
        Y_index = 1
        classific = 5
    elif args.phenotype == "Braak":
        Y_index = 0
        classific = 7
elif args.dataset_str == "MSBB":
    data_name = "MSBB_19440genes"
    if args.phenotype == "CERAD":
        Y_index = 0
        classific = 5
    elif args.phenotype == "Braak":
        Y_index = 1
        classific = 7
elif args.dataset_str == "ACT":
    data_name = "ACT_17574genes"
    if args.phenotype == "CERAD":
        Y_index = 0
        classific = 5
    elif args.phenotype == "Braak":
        Y_index = 1
        classific = 7
    elif args.phenotype == "Tau_IHC":
        Y_index = 2
    elif args.phenotype == "Abeta_IHC":
        Y_index = 3
elif args.dataset_str == "Mayo":
    data_name = "Mayo_13933genes"
    if args.phenotype == "Thal":
        Y_index = 1
        classific = 6
    elif args.phenotype == "Braak":
        Y_index = 0
        classific = 7

elif args.dataset_str == "ANM1":
    data_name = "ANM1"
    Y_index = 0
    classific = 3

elif args.dataset_str == "ANM2":
    data_name = "ANM2"
    Y_index = 0
    classific = 3

# Paths
data_path = './data/'+args.dataset_str+'/'+data_name+'.csv'
if args.phenotype == "Tau_IHC" or args.phenotype == "Abeta_IHC":
    path_label = './data/'+args.dataset_str+"/label_norm.csv"
else:
    path_label = './data/'+args.dataset_str+"/label_"+args.dataset_str+".csv"
if args.isCCA == True:
    path_matrix = './data/'+args.dataset_str+'/'+data_name+'_graph.csv' ## feature matrix is not only
path_graph = './data/'+args.dataset_str+'/graphmatrix_'+data_name+'.csv'

GAT_autoencoder_path = 'logs/GATae_'+args.dataset_str+'.h5'
model_path = 'logs/model_'+args.dataset_str+'.h5'
pred_path = 'result/pred_'+args.dataset_str+'.txt'
intermediate_path = 'logs/model_'+args.dataset_str+'_'

# Read data
Y = pd.read_csv(path_label, sep=',',header=0,index_col=0).values[:,Y_index]
if args.isCCA == True:
    X = pd.read_csv(path_matrix, sep=',',header=0,index_col=0).values
else:
    X = pd.read_csv(data_path, sep=',',header=0,index_col=0).values
    # features = normalization(features)

if args.dataset_str == "ANM1" or args.dataset_str == "ANM2":
    X = X.T

A = pd.read_csv(path_graph, sep=',',header=0,index_col=None).values
# Parameters
N = X.shape[0]                  # Number of nodes in the graph
F = X.shape[1]                  # Original feature dimension

# Loss functions
def mae(y_true, y_pred):
    return K.mean(K.abs(y_pred-y_true))

def DAEGC_class_loss_1(y_pred):
    return K.mean(K.exp(-1 * A * K.sigmoid(K.dot(y_pred, K.transpose(y_pred)))))

def maie_class_loss(y_true, y_pred):
    loss_E = mae(y_true, y_pred)
    return loss_E

class PrintLossCallback(tf.keras.callbacks.Callback):
    def __init__(self, print_frequency=100):
        self.print_frequency = print_frequency

    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % self.print_frequency == 0:
            print(f"Epoch {epoch + 1}/{self.params['epochs']} - loss: {logs['loss']:.4f}")

# Model definition
X_in = Input(shape=(F,))
A_in = Input(shape=(N,))

dropout1 = Dropout(dropout_rate)(X_in)
graph_attention_1 = GraphAttention(args.F1,
                                   attn_heads=args.n_attn_heads,
                                   attn_heads_reduction='concat',
                                   dropout_rate=dropout_rate,
                                   activation='elu',
                                   kernel_regularizer=l1(args.l2_reg),
                                   attn_kernel_regularizer=l1(args.l2_reg))([dropout1, A_in])

dropout2 = Dropout(dropout_rate)(graph_attention_1)
graph_attention_2 = GraphAttention(args.F2,
                                   attn_heads=args.n_attn_heads,
                                   attn_heads_reduction='concat',
                                   dropout_rate=dropout_rate,
                                   activation='elu',
                                   kernel_regularizer=l1(args.l2_reg),
                                   attn_kernel_regularizer=l1(args.l2_reg))([dropout2, A_in])

dropout3 = Dropout(dropout_rate)(graph_attention_2)
graph_attention_3 = GraphAttention(args.F1,
                                   attn_heads=args.n_attn_heads,
                                   attn_heads_reduction='concat',
                                   dropout_rate=dropout_rate,
                                   activation='elu',
                                   kernel_regularizer=l1(args.l2_reg),
                                   attn_kernel_regularizer=l1(args.l2_reg))([dropout3, A_in])

dropout4 = Dropout(dropout_rate)(graph_attention_3)
graph_attention_4 = GraphAttention(F,
                                   attn_heads=args.n_attn_heads,
                                   attn_heads_reduction='average',
                                   dropout_rate=dropout_rate,
                                   activation='elu',
                                   kernel_regularizer=l1(args.l2_reg),
                                   attn_kernel_regularizer=l1(args.l2_reg))([dropout4, A_in])

# Build GAT autoencoder modelon_4)
GAT_autoencoder = Model(inputs=[X_in, A_in], outputs=graph_attention_4)
optimizer = Adam(lr=args.pre_lr)
GAT_autoencoder.compile(optimizer=optimizer,
              loss=maie_class_loss)

# Callbacks
es_callback = EarlyStopping(monitor='loss', min_delta=0.1, patience=50)
tb_callback = TensorBoard(batch_size=N)
mc_callback = ModelCheckpoint(GAT_autoencoder_path,
                              monitor='loss',
                              save_best_only=True,
                              save_weights_only=True)

# Train GAT_autoencoder model
print('Train GAE model')
start_time = time.time()
print_frequency = 100
callbacks = [PrintLossCallback(print_frequency=print_frequency)]
GAT_autoencoder.fit([X, A],X,epochs=args.pre_epochs,batch_size=N,
                    verbose=0,shuffle=False,callbacks=callbacks)
end_time = time.time()
run_time = (end_time - start_time) / 60
print('hidden features: run time is %.2f '%run_time, 'minutes')

# Construct a model for hidden layer
hidden_model = Model(inputs=GAT_autoencoder.input,outputs=graph_attention_2)
hidden = hidden_model.predict([X, A], batch_size=N)
hidden = hidden.astype(float)
print('hidden features shape is: ',end=' ')
print(hidden.shape)

X_train, X_valid, y_train, y_valid = train_test_split(hidden, Y, test_size=0.3)

print('Train MLP')
if args.phenotype == "Tau_IHC" or args.phenotype == "Abeta_IHC":
    model = Sequential()
    input = hidden.shape[1]
    model.add(Dense(100, input_shape=(input,))) 
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
    model.add(Dense(64))
    model.add(Activation('relu'))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer=Adam())

    start_time = time.time()
    print_frequency = 100
    callbacks = [PrintLossCallback(print_frequency=print_frequency)]
    model.fit(X_train, y_train, epochs=args.epochs, batch_size=20, verbose=0,
                        shuffle=False,callbacks=callbacks)
    end_time = time.time()
    run_time = (end_time - start_time) / 60
    print('MLP: run time is %.2f '%run_time, 'minutes')

    yhat = model.predict(hidden)
    if args.isCCA == True:
        filename = './result/AD-SpatialGAE_'+args.dataset_str+'_'+phenotype+' '+accuracy+'.csv'
    else:
        filename = './result/MultiGAE_'+args.dataset_str+'_'+phenotype+' '+accuracy+'.csv'

    with open(filename,'w',newline='') as f:
        writer = csv.writer(f)
        writer.writerows(yhat)

    v = np.var(Y)
    print("mse:",metrics.mean_squared_error(Y, yhat))
    # print("1-R^2:",metrics.mean_squared_error(Y, yhat)/v)
    # # 计算均方根误差（RMSE）
    # rmse = np.sqrt(mean_squared_error(Y, yhat))
    # print("RMSE:", rmse)

    # # 计算平均绝对误差（MAE）
    # mae = mean_absolute_error(Y, yhat)
    # print("MAE:", mae)

    # # 计算R平方
    # r2 = r2_score(Y, yhat)
    # print("R-squared:", r2)

else: 
    model = Sequential()
    input = hidden.shape[1]
    model.add(Dense(100, input_shape=(input,))) 
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
    model.add(Dense(64))
    model.add(Activation('relu'))
    model.add(Dense(classific, activation='softmax'))
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    start_time = time.time()
    print_frequency = 100
    callbacks = [PrintLossCallback(print_frequency=print_frequency)]
    model.fit(X_train, y_train, epochs=args.epochs, batch_size=20,
                        validation_data=(X_valid, y_valid), verbose=0,
                        shuffle=False,callbacks = callbacks)
    end_time = time.time()
    run_time = (end_time - start_time) / 60
    print('MLP: run time is %.2f '%run_time, 'minutes')

    yhat = model.predict(hidden)
    y_pred = np.argmax(yhat, axis=1)
    accuracy = metrics.accuracy_score(Y, y_pred)
    precision = metrics.precision_score(Y, y_pred, average='weighted')
    recall = metrics.recall_score(Y, y_pred, average='macro')
    f1_score = metrics.f1_score(Y, y_pred, average='weighted')

    if args.isCCA == True:
        filename = './result/AD-SpatialGAE_'+args.dataset_str+'_'+phenotype+' '+str(accuracy)+'.csv'
    else:
        filename = './result/MultiGAE_'+args.dataset_str+'_'+phenotype+' '+str(accuracy)+'.csv'

    with open(filename,'w',newline='') as f:
        writer = csv.writer(f)
        writer.writerows(yhat)

    print("accuracy:",accuracy)
    print("precision:",precision)
    print("recall:",recall)
    print("f1_score:",f1_score)
    
    
