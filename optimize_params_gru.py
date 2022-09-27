import numpy as np
import pandas as pd
import time
from math import sqrt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import ticker
from matplotlib.pyplot import *
from scipy import stats

start = time.time()

df = pd.read_csv("train.csv")
df["Mag"] = df["X"]**2 + df["Y"]**2 + df["Z"]**2
samples_t = np.array(df.iloc[:,0])


def dev_ids(df):
    df['dev_id'] = df.index
    return df["dev_id"]
    

def device_sample_counter(df):        
    cnts = df.Device.value_counts()
    cnts = cnts.sort_index()
    cnts = pd.DataFrame(cnts)
    return cnts


def device_sample_endpoint(cnts):
    cnts_end = [0]    
    k = 0
    for i in range(len(cnts)):
        k += cnts.iloc[i,0]
        cnts_end.append(k)
    return cnts_end


def get_sequences(dev_id, jump, seq_len):
    
    begin = device_endpoints[dev_id]
    end = device_endpoints[dev_id+1]
    
    sequences = [begin]
    
    for i in range(begin,end-1):
        if samples_t[i+1] - samples_t[i] >= jump:
            sequences.append(i+1)
    if dev_id != 386:        
        sequences.append(end-1)
    else:
        sequences.append(end)
            
    seq_length = []
    
    possible_seqs = []
    
    for i in range(len(sequences)-1):
        seq_length.append(sequences[i+1]-sequences[i])
        if sequences[i+1]-sequences[i] >= seq_len:
            possible_seqs.append((sequences[i],sequences[i+1]-1))
            
    return sequences, seq_length, possible_seqs


dev_sample_count = device_sample_counter(df)
device_endpoints = device_sample_endpoint(dev_sample_count)
dev_ids = dev_ids(dev_sample_count)


time_jump = 250 # in miliseconds
seqs_at_least = 500

poss_devs = []
for i in range(len(dev_sample_count)):
    dev_label = dev_sample_count.iloc[i,1]
    seqs, seq_lengths, possible_seq = get_sequences(i,time_jump,seqs_at_least)
    poss_devs.append((dev_label, len(possible_seq)))
    
    
all_poss = 0    
for i in range(len(poss_devs)):
    all_poss += poss_devs[i][1]
#print(all_poss)


df_new = []

for j in range(len(dev_sample_count)):
    seqs, seq_lengths, possible_seq = get_sequences(j,time_jump,seqs_at_least)
    for i in range(len(possible_seq)):         
        poss_arr = df.iloc[possible_seq[i][0]:possible_seq[i][1],]
        df_new.append(poss_arr)


df = pd.concat(df_new)


from sklearn.preprocessing import RobustScaler

scale_columns = ['X', 'Y', 'Z', 'Mag']

scaler = RobustScaler()

scaler = scaler.fit(df[scale_columns])

df.loc[:, scale_columns] = scaler.transform(df[scale_columns].to_numpy())

sequences = []
devices = []

sequence_len = 50
increment = 50
validation_percent = 0.2


for i in range(0,  df.shape[0] - sequence_len, increment):  

    x_data = df['X'].values[i: i + sequence_len]

    y_data = df['Y'].values[i: i + sequence_len]

    z_data = df['Z'].values[i: i + sequence_len]

    mag_data = df['Mag'].values[i: i + sequence_len]
    
    dev_id = stats.mode(df['Device'][i: i + sequence_len])[0][0]

    sequences.append([x_data, y_data, z_data, mag_data])

    devices.append(dev_id)


from sklearn.model_selection import train_test_split

devices_shapeform = np.asarray(pd.get_dummies(devices), dtype = np.float32)
sequences_shapeform = np.asarray(sequences, dtype= np.float32).reshape(-1, sequence_len, 4)

X_train, X_test, y_train, y_test = train_test_split(sequences_shapeform, devices_shapeform, test_size = validation_percent, random_state = 1337)

from keras.models import Sequential
from keras.layers import GRU, Dense, Dropout
from keras.optimizers import Adam


def optimize_lr(learning_rate):
    
    test_acc = []
    test_l = []
    time_list = []
    
    for lr in learning_rate:
        epoch_num = 3
        batch_size = 1024
        
        ts = time.time()
        
        model = Sequential()
        
        model.add(GRU(units = 128, input_shape = (X_train.shape[1], X_train.shape[2])))
        
        model.add(Dropout(0.5)) 
        
        model.add(Dense(units = 64, activation='relu'))
        
        model.add(Dense(y_train.shape[1], activation = 'softmax'))
        
        opt = Adam(learning_rate=lr)
        
        model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
        
        model.summary()
        
        fitted_model = model.fit(X_train, y_train, epochs = epoch_num, validation_split = validation_percent, batch_size = batch_size, verbose = 1)
        
        test_loss, test_accuracy = model.evaluate(X_test, y_test, batch_size = batch_size, verbose = 1)
        test_acc.append(test_accuracy)
        test_l.append(test_loss)
        te = time.time()
        
        time_list.append((te-ts)/3)
        print("Test Accuracy :", test_accuracy)
        print("Test Loss :", test_loss)
        
        
    fig, ax = plt.subplots()
    ax2 = ax.twinx()   
    ax.plot(range(len(learning_rate)), test_acc, "g", label = "Test accuracy", marker="o")
    ax2.plot(range(len(learning_rate)), time_list, "b--", label = "Duration", marker="o")
    plt.title("Test accuracy and loss for changing learning rate")
    ax.xaxis.set_ticks(range(len(learning_rate)))
    ax.xaxis.set_ticklabels(learning_rate)
    ax.legend(bbox_to_anchor=(1.1,1), loc="upper left", borderaxespad=0)
    ax2.legend(bbox_to_anchor=(1.1,0.8), loc="lower left", borderaxespad=0)
    ax.set_ylabel('Test Accuracy')
    ax2.set_ylabel('Time per epoch (seconds)')
    ax.set_xlabel('Learning Rate') 
    ax.grid(color='grey', linestyle='-', linewidth=0.25, alpha=0.5)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)    
    ax2.yaxis.set_major_formatter(ticker.StrMethodFormatter('{x:,.0f}'))
    plt.show()
    
    return True

def optimize_bsize(batch_sizes):
    
    test_acc = []
    test_l = []
    time_list = []
    
    for batch in batch_sizes:
        epoch_num = 3
        lr = 0.01
        batch_size = batch
        
        ts = time.time()
        
        model = Sequential()
        
        model.add(GRU(units = 128, input_shape = (X_train.shape[1], X_train.shape[2])))
        
        model.add(Dropout(0.5)) 
        
        model.add(Dense(units = 64, activation='relu'))
        
        model.add(Dense(y_train.shape[1], activation = 'softmax'))
        
        opt = Adam(learning_rate=lr) # Using best learning rate based on previous result.
        
        model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
        
        model.summary()
        
        fitted_model = model.fit(X_train, y_train, epochs = epoch_num, validation_split = validation_percent, batch_size = batch_size, verbose = 1)
        
        test_loss, test_accuracy = model.evaluate(X_test, y_test, batch_size = batch_size, verbose = 1)
        test_acc.append(test_accuracy)
        test_l.append(test_loss)
        te = time.time()
        
        time_list.append((te-ts)/3)     
        print("Test Accuracy :", test_accuracy)
        print("Test Loss :", test_loss)
        

    fig, ax = plt.subplots()
    ax2 = ax.twinx()   
    ax.plot(range(len(batch_sizes)), test_acc, "g", label = "Test accuracy", marker="o")
    ax2.plot(range(len(batch_sizes)), time_list, "b", label = "Duration", marker="o")
    plt.title("Test accuracy and loss for changing batch size")
    ax.xaxis.set_ticks(range(len(batch_sizes)))
    ax.xaxis.set_ticklabels(batch_sizes)
    ax.legend(bbox_to_anchor=(1.1,1), loc="upper left", borderaxespad=0)
    ax2.legend(bbox_to_anchor=(1.1,0.8), loc="lower left", borderaxespad=0)
    ax.set_ylabel('Test Accuracy')
    ax2.set_ylabel('Time per epoch (seconds)')
    ax.set_xlabel('Batch Size') 
    ax.grid(color='grey', linestyle='-', linewidth=0.25, alpha=0.5)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    ax2.yaxis.set_major_formatter(ticker.StrMethodFormatter('{x:,.0f}'))
    plt.show()
    
    return True

def optimize_dropout(drop_rates):
    
    test_acc = []
    test_l = []
    time_list = []
    
    for drop in drop_rates:
        epoch_num = 3
        lr = 0.01
        batch_size = 256
        
        ts = time.time()
        
        model = Sequential()
        
        model.add(GRU(units = 128, input_shape = (X_train.shape[1], X_train.shape[2])))
        
        model.add(Dropout(drop)) 
        
        model.add(Dense(units = 64, activation='relu'))
        
        model.add(Dense(y_train.shape[1], activation = 'softmax'))
        
        opt = Adam(learning_rate=lr) # Using best learning rate based on previous result.
        
        model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
        
        model.summary()
        
        fitted_model = model.fit(X_train, y_train, epochs = epoch_num, validation_split = validation_percent, batch_size = batch_size, verbose = 1)
        
        test_loss, test_accuracy = model.evaluate(X_test, y_test, batch_size = batch_size, verbose = 1)
        test_acc.append(test_accuracy)
        test_l.append(test_loss)
        te = time.time()
        
        time_list.append((te-ts)/3)      
        print("Test Accuracy :", test_accuracy)
        print("Test Loss :", test_loss)
    
    fig, ax = plt.subplots()
    ax2 = ax.twinx()   
    ax.plot(range(len(drop_rates)), test_acc, "g", label = "Test accuracy", marker="o")
    ax2.plot(range(len(drop_rates)), time_list, "b", label = "Duration", marker="o")
    plt.title("Test accuracy and loss for changing dropout rate")
    ax.xaxis.set_ticks(range(len(drop_rates)))
    ax.xaxis.set_ticklabels(drop_rates)
    ax.legend(bbox_to_anchor=(1.1,1), loc="upper left", borderaxespad=0)
    ax2.legend(bbox_to_anchor=(1.1,0.8), loc="lower left", borderaxespad=0)
    ax.set_ylabel('Test Accuracy')
    ax2.set_ylabel('Time per epoch (seconds)')
    ax.set_xlabel('Dropout Rate') 
    ax.grid(color='grey', linestyle='-', linewidth=0.25, alpha=0.5)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    ax2.yaxis.set_major_formatter(ticker.StrMethodFormatter('{x:,.0f}'))
    plt.show()
    
    return True


def optimize_layers(input_neuron):
    
    test_acc = []
    test_l = []
    time_list = []
    
    for neuron in input_neuron:
        epoch_num = 3
        lr = 0.01
        batch_size = 256
        
        ts = time.time()
        
        model = Sequential()
        
        model.add(GRU(units = int(neuron), input_shape = (X_train.shape[1], X_train.shape[2])))
        
        model.add(Dropout(0.1)) 
        
        model.add(Dense(units = int(neuron*2), activation='relu'))
        
        model.add(Dense(y_train.shape[1], activation = 'softmax'))
        
        opt = Adam(learning_rate=lr) # Using best learning rate based on previous result.
        
        model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
        
        model.summary()
        
        fitted_model = model.fit(X_train, y_train, epochs = epoch_num, validation_split = validation_percent, batch_size = batch_size, verbose = 1)
        
        test_loss, test_accuracy = model.evaluate(X_test, y_test, batch_size = batch_size, verbose = 1)
        test_acc.append(test_accuracy)
        test_l.append(test_loss)
        te = time.time()
        
        time_list.append((te-ts)/3)       
        print("Test Accuracy :", test_accuracy)
        print("Test Loss :", test_loss)
    
    fig, ax = plt.subplots()
    ax2 = ax.twinx()   
    ax.plot(range(len(input_neuron)), test_acc, "g", label = "Test accuracy", marker="o")
    ax2.plot(range(len(input_neuron)), time_list, "b", label = "Duration", marker="o")
    plt.title("Test accuracy and loss for changing input schema")
    ax.xaxis.set_ticks(range(len(input_neuron)))
    ax.xaxis.set_ticklabels(input_neuron)
    ax.legend(bbox_to_anchor=(1.1,1), loc="upper left", borderaxespad=0)
    ax2.legend(bbox_to_anchor=(1.1,0.8), loc="lower left", borderaxespad=0)
    ax.set_ylabel('Test Accuracy')
    ax2.set_ylabel('Time per epoch (seconds)')
    ax.set_xlabel('Number of input neurons') 
    ax.grid(color='grey', linestyle='-', linewidth=0.25, alpha=0.5)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    ax2.yaxis.set_major_formatter(ticker.StrMethodFormatter('{x:,.0f}'))
    plt.show()
    
    return True

lrate = [0.1, 0.01, 0.001, 0.0001] # Giant Steps
optimize_lr(lrate)

lrate2 = [0.01, 0.02, 0.03, 0.04] # Small Steps Upwards
optimize_lr(lrate2)

lrate3 = [0.01, 0.009, 0.008, 0.007] # Small Steps Downwards
optimize_lr(lrate3)

b_sizes = [64,128,256,512,1024]
optimize_bsize(b_sizes)

drops = [0.1, 0.2, 0.3, 0.4, 0.5]
#optimize_dropout(drops)

neurons = [64,128,256]
#ptimize_layers(neurons)

        
end = time.time()
print(end-start)