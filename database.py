import pandas as pd
#import unicodedata
from pythainlp import word_tokenize
import numpy as np
import MySQLdb
import os
#x=pd.read_csv('C:\\Users\\codia\\Desktop\\database\\1.txt',encoding = 'utf8').values.astype('str')
#y=x[:,1]
directory = 'C:\\Users\\codia\\Desktop\\database\\'
dataset_dir = directory + 'dataset\\'
def insert_db(name):
    # Open database connection
    db = MySQLdb.connect("localhost","root","","tb_database",use_unicode=True,charset='utf8')
    name="'"+name + "');"
    # prepare a cursor object using cursor() method
    cursor = db.cursor()
    
    # Prepare SQL query to INSERT a record into the database.
    sql = """INSERT INTO thai_word(word)
         VALUES ("""+name
    try:
       # Execute the SQL command
       cursor.execute(sql)
       # Commit your changes in the database
       db.commit()
    except:
       # Rollback in case there is any error
       db.rollback()
    
    # disconnect from server
    db.close()
#
def database_get_index(name_data):
# Open database connection
    db = MySQLdb.connect("localhost","root","","tb_database",use_unicode=True,charset='utf8')
    name_data="'"+name_data + "'"
# prepare a cursor object using cursor() method
    cursor = db.cursor()
    # Prepare SQL query to INSERT a record into the database.
    sql = "SELECT * FROM `thai_word` WHERE `word` LIKE"+name_data
    try:
   # Execute the SQL command
       cursor.execute(sql)
   # Commit your changes in the database
       db.commit()
   # Fetch all the rows in a list of lists.
       results = cursor.fetchall()
       for row in results:
           index = row[0]
           return index
   # Execute the SQL command    
    except:
   # Rollback in case there is any error
       db.rollback()   
# disconnect from server
    db.close()
    return -1 # Not exit in database
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
#                    data=data[:,1]
                    X_train.append(data.iloc[:,-1].values.astype('str').reshape(-1,1)) 
                    y_train.append(str(data_class))
            
                elif dataset == 'test':
                    data = pd.read_csv(str(file_path))
#                    data=data[:,1]
                    X_test.append(data.iloc[:,-1].values.astype('str').reshape(-1,1)) 
                    y_test.append(str(data_class))
    
    return X_test , y_test , X_train , y_train
i=0
def input_database(y):
#y is data
    for i in y:
        check=database_get_index(i)
        if check==-1:       
            insert_db(i)
X_test , y_test , X_train , y_train=load_data(dataset_dir)
for x in X_train:
    i=i+1
    for y in x:
        word_cut = word_tokenize(''.join(y), engine='deepcut')
        print(word_cut)
        
        input_database(word_cut)
    print(i)