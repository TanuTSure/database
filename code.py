import pandas as pd
import random
import os
import collections
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from tensorflow.contrib import rnn
from pythainlp import word_tokenize
from sklearn.preprocessing import LabelEncoder , OneHotEncoder
import time
import matplotlib.pyplot as plt
import MySQLdb
import database as db

directory = 'C:\\Users\\Tanut\\Desktop\\database\\'
dataset_dir = directory + 'dataset/'
save_path = directory + "save/"

batch_size = 10
pad_size = (200,500)

train_step = 1
num_class = 3
num_hidden = 10
timesteps = 200
num_input = 500

###############################################################################

def load_data(dataset_dir) :
    
    y_test = []
    y_train = []
    X_test = []
    X_train = []
    
    for dataset in os.listdir(dataset_dir):
        
        data_path = os.path.join(dataset_dir, str(dataset))
        
        for data_class in os.listdir(data_path):        
            
            class_path = os.path.join(data_path, str(data_class))
    
            for  file in os.listdir(class_path):
                file_path = os.path.join(class_path, str(file))
                file_path = file_path.replace('\\' , '//')
                print(data_class)
                print(file)
             
                if dataset == 'train':
                    data = pd.read_csv(str(file_path))
                    X_train.append(data.iloc[:,-1].values.astype('str').reshape(-1,1)) 
                    y_train.append(str(data_class))
            
                elif dataset == 'test':
                    data = pd.read_csv(str(file_path))
                    X_test.append(data.iloc[:,-1].values.astype('str').reshape(-1,1)) 
                    y_test.append(str(data_class))
    
    return X_test , y_test , X_train , y_train

###############################################################################
    
def build_dataset(list_X , list_y):
    
    X = []
    y = []
    
    y_array = np.array(list_y).reshape(-1,1)
    labelencoder = LabelEncoder()
    y_array[:, 0] = labelencoder.fit_transform(y_array[:, 0])
    onehotencoder = OneHotEncoder(categorical_features = [0])
    y_array = onehotencoder.fit_transform(y_array).toarray()
    
    for index in range(0,y_array.shape[0]):
        y.append(y_array[index,:].astype('int'))
        
    for x in list_X :
        pad = np.zeros(pad_size , dtype='int').astype('str')
        for line in range(0,x.shape[0]):
            word_cut = word_tokenize(x[line , 0], engine='deepcut')
            for index in range(0,len(word_cut)):
                pad[line , index] = word_cut[index]
        X.append(pad)
        
    return X , y

###############################################################################
    
def word_embedding(list_train , list_test):
    db_1 = MySQLdb.connect("localhost","root","","tb_database",use_unicode=True,charset='utf8')
#    X_concate = list_train[0]
#    for index in range(1,len(list_train)):
#        X_concate = np.concatenate((X_concate, list_train[index]), axis=0)
#    for index in range(0,len(list_test)):
#        X_concate = np.concatenate((X_concate, list_test[index]), axis=0)
#       
#    X_test = []
#    X_train = []
#    words = X_concate.reshape(-1,)
#    print("counting word")
#    count = collections.Counter(words).most_common()
#    dictionary = dict()
#    print("collect word in dict")
#    for word, _ in count:
#        dictionary[word] = len(dictionary)
#    reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
#    
#    
    for x in list_train :
        for row in range(0,x.shape[0]):
            for column in (0,x.shape[1]):
                x[row,column]=db.database_get_index(x[row,column])          
            
        X_train.append(x)
    for x in list_test :
        for row in range(0,x.shape[0]):
            for column in range(0,x.shape[1]):
                x[row,column]=db.database_get_index(x[row,column])  
        
        X_test.append(x)    
    db_1.close()    
    return X_train , X_test

###############################################################################
def random_data(X,y) :
    X_rand = []
    y_rand = []  
    
    for time in range(batch_size) :
        rand = random.randint(0,len(y)-1)
        X_rand.append(X[rand])
        y_rand.append(y[rand])
        
    return X_rand,y_rand
###############################################################################
def placeholder_inputs():
    
    # tf Graph inputs
    X = tf.placeholder(tf.float32, [None, timesteps, num_input])
    y = tf.placeholder(tf.float32, [None, num_class])
    
    return X , y
###############################################################################    
def network(x) :
        
        # Define weights matrices
    weights = {
   
        'out': tf.Variable(tf.random_normal([num_hidden, num_class]))
    }
    
    # Define bias vectors
    biases = {
            
        'out': tf.Variable(tf.constant(0.1, shape=[num_class,]))
    }    
    # Unstack to get a list of 'timesteps' tensors of shape (batch_size, n_input)
    x = tf.unstack(x, timesteps , 1)
#------------------------------------------------------------------------------
    # basic LSTM Cell 1
    lstm_cell1 = tf.nn.rnn_cell.BasicLSTMCell(num_hidden, forget_bias=1.0, state_is_tuple=True , name = 'lstm1')
    outputs1, final_states1 = rnn.static_rnn(lstm_cell1, x, dtype=tf.float32)
    #final_state_output1 = tf.matmul(outputs1[-1], weights['out']) + biases['out']
#------------------------------------------------------------------------------
    # basic LSTM Cell 2
    lstm_cell2 = tf.nn.rnn_cell.BasicLSTMCell(num_hidden, forget_bias=1.0, state_is_tuple=True , name = 'lstm2')
    outputs2, final_states2 = rnn.static_rnn(lstm_cell2, outputs1, dtype=tf.float32)
    final_state_output2 = tf.matmul(outputs2[-1], weights['out']) + biases['out']
#------------------------------------------------------------------------------
    
    return final_state_output2
###############################################################################     
def train(X_train , y_train , X_test , y_test , train_step) :
    
    loss_op_list = []
    accuracy_list = []
    amout_of_testset = len(y_test)
       
    filename = is_saved_model_exist(save_path)
    
    if filename == "" :
        
        x , y_ = placeholder_inputs()
        logits = network(x)
        
        tf.add_to_collection("logits", logits)
        tf.add_to_collection("x", x)
        tf.add_to_collection("y_", y_)
    
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            saver = tf.train.Saver()
            saver.save(sess , save_path + "graph" )
            saver.export_meta_graph(save_path + "graph.meta")
            
    with tf.Session() as sess:
        
        timestart = time.clock()
        sess.run(tf.global_variables_initializer())
        saver = tf.train.import_meta_graph(save_path + "graph.meta")
        saver.restore(sess, tf.train.latest_checkpoint(save_path))
        timestamp = time.clock()
        print("import garph has used : " , timestamp-timestart , ("sec."))
        
        timestart = time.clock()
        logits = tf.get_collection("logits")[0]
        x = tf.get_collection("x")[0]
        y_ = tf.get_collection("y_")[0]
        timestamp = time.clock()
        print("import logits has used : " , timestamp-timestart , ("sec."))
        
        timestart = time.clock()
        # Define loss and optimizer
        loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
        logits=logits, labels=y_))
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1)
        train_op = optimizer.minimize(loss_op)
            
        # Evaluate model (with test logits, for dropout to be disabled)
        correct_pred = tf.equal(tf.argmax(logits, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
        timestamp = time.clock()
        print("define loss , op and eval has used : " , timestamp-timestart , ("sec."))
        
        for step in range(1,train_step+1):
            
            timestart = time.clock()
            X_input , y_input = random_data(X_train,y_train)                
            X_input = np.array(X_input).reshape(batch_size,timesteps,num_input)
            y_input = np.array(y_input).reshape(batch_size,num_class)

            sess.run(train_op , feed_dict={x: X_input, y_: y_input})
                      
            print("Step : " , step)
            #X_test , y_test = random_data(X_test,y_test)                
            X_test = np.array(X_test).reshape(amout_of_testset,timesteps,num_input)
            y_test = np.array(y_test).reshape(amout_of_testset,num_class)
            
#            print("correct pred : " , sess.run(correct_pred, feed_dict={x: X_test, y_: y_test}))
            
            accuracy_list.append(sess.run(accuracy, feed_dict={x: X_test, y_: y_test}))
            loss_op_list.append(sess.run(loss_op, feed_dict={x: X_test, y_: y_test}))
            
            print("accuracy : " , accuracy_list[-1])
            print("loss op : " , loss_op_list[-1])
            
            print(sess.run(tf.argmax(logits, 1) , feed_dict={x : X_test}))
            print(sess.run(tf.argmax(y_, 1) , feed_dict={y_ : y_test}))
            
            timestamp = time.clock()
            print("train step has used : " , timestamp-timestart , ("sec."))
            
            if step%100 == 0:
                saver.save(sess, save_path + "graph" , global_step=step)
                
    return accuracy_list , loss_op_list
            
###############################################################################
def is_saved_model_exist(save_path):
    fileList = os.listdir(save_path)
    for file in fileList:
        if "checkpoint" in file:
            print ('found checkpoint')
            return file
    print ('checkpoint not found')  
    return ""
###############################################################################
def split_train_test(X_load , y_load , ratio) :
        
    y_test = []
    X_test = []
    
    amount_test = round(ratio*len(y_load))
    print(amount_test)
        
    for _ in range(amount_test) :
        ran = random.randint(0,len(y_load)-1)
        y_test.append(y_load[ran])
        X_test.append(X_load[ran])
        X_load.pop(ran)
        y_load.pop(ran)
        
    X_train = X_load
    y_train = y_load
    
    return X_train , X_test , y_train , y_test
############################################################################### 

X_test , y_test , X_train , y_train = load_data(dataset_dir)
X_test , y_test = build_dataset(X_test , y_test)
X_train , y_train = build_dataset(X_train , y_train)
X_train , X_test , dictionary = word_embedding(X_train , X_test)

accuracy , loss_op = train(X_train , y_train , X_test , y_test , train_step)

step = np.arange(1, train_step+1 , 1)
step = step.reshape((len(step), 1))
plt.plot(step , accuracy , color = 'blue')
plt.title('trend of model accuracy')
plt.xlabel('step')
plt.ylabel('accuracy')
plt.show()

plt.plot(step , loss_op , color = 'red')
plt.title('trend of loss op')
plt.xlabel('step')
plt.ylabel('loss op')
plt.show()
