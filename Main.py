
#---------------------------------------------------Importing Libraries -----------------------------------#
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.models import load_model
from sklearn.decomposition import TruncatedSVD
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental import preprocessing
from keras.models import Sequential
from keras.layers import Activation, Dense
from sklearn.metrics import classification_report
from tensorflow.keras.layers import GRU, Embedding, SimpleRNN, Activation , LSTM , Dropout
import os
import pickle
import csv
import uuid
import pandas as pd 
import warnings
warnings.filterwarnings("ignore")


#--------------------------------------------- Read original Dataset ---------------------------------------#

def read_df(filepath):
     with open(filepath, 'rb') as infile:
         df=pickle.load(infile)
     return df

#------------------------------------------------------------------------------------------------------------#

# This function for returning cleand data 
def preprocessing(df):
    # We use uuid library to fill null values with random id 
    df.user_session.replace(np.nan, uuid.uuid1()  , inplace=True)
    # We choose the runail brand the mode beacuse the mode function not working in our case 
    df.brand.replace('nan' , 'runail'  , inplace=True)
    df.price.replace(0.0 , df.price.mean()  , inplace=True) 
    return df

 #------------------------------------------------------------------------------------------------------------#

 #This function create groupby for SVD modlling 
def groupbyforSVD(df): 
    events = {'purchase':1,'cart': 2,'view': 3, 'remove_from_cart':4}
    df['event'] = df.event_type.map(events)
    df=df.groupby(['user_id','product_id'])['event'].max()
    df= pd.DataFrame(df).reset_index()
    return df

#------------------------------------------------------------------------------------------------------------#

# This function take data fron groupbyforSVD function, chunk it and create pivot table for each chunk then
# these files will be merged using shell script  
def chunking(df):
        # Creating a smaller chunksize
        chunk_size = df.shape[0]//16
        # store the chunks in a list
        chunks = [x for x in range(0, df.shape[0], chunk_size)]
        for i in range(len(chunks)-1):
            print(chunks[i], chunks[i+1])
            # Extract the chunked dataframes
            t = df[chunks[i]:chunks[i+1]]
            # create a pivot table from the chunk
            temp = t.pivot_table(values='event', index='user_id', columns='product_id',fill_value=0)
            print("pivot complete")
            # save it as a file
            temp.to_pickle(f'file_{i}.pkl')
            print("saved file")
            # delete the dataframes so you don't run into memory issues
            del temp
            print("deleted df")
            print("-----------")


#------------------------------------------------------------------------------------------------------------#

# This function open saved file from shell file that contains pivot table and use it in SVD model 
def modeling_SVD():
    infile = open('file.pkl','rb')
    event_df = pickle.load(infile)
    #We are going to transpose the matrix
    X = event_df.T
    # Create SVD model with 300 components
    SVD = TruncatedSVD(n_components=300, random_state=42)
    event_matrix = SVD.fit_transform(X)
    # We use the correlation matrix to calculate the similarity
    corr_mat = np.corrcoef(event_matrix)
    event_product = event_df.columns
    return event_product , corr_mat 


#-----------------------------------------------------------------------------------------------------------#

# This function take cleand dataframe and convert to Sequance data for each user session and applying
# Sequential RNN model and save it and returning test data 
def rnn_model(df):
    # Mapping event to number 
    events = {'purchase':1,'cart': 2,'view': 3, 'remove_from_cart':4}
    df['event'] = df.event_type.map(events)
    sequence = df.groupby('user_session')['event'].apply(list)
    sequence = sequence.reset_index()
    # Filter the data to extract a Purchase when event=1 or not when event!=1 label.
    sequence['purchase'] = sequence['event'].apply(lambda x: 1 if 1 in x else 0)
    sequence = sequence[sequence['event'].map(len)> 1]
    productdf= pd.DataFrame(df.groupby('user_session')['product_id'].apply(list)).reset_index()
    productdf=productdf[productdf['product_id'].map(len)>1]
    #Add product list to the dataframe
    sequence['product']=productdf['product_id']
    #The sequence data should not contain the "purchase field" so it is filtered out
    sequence['event']= sequence.event.apply(lambda row: list(filter(lambda a: a != 1, row)))
    # Chossing sequance length upto 5 for event pattren and Discard remaining sequences.
    short_sequence_5 = sequence[sequence['event'].map(len) <= 5]
    event_sequence = short_sequence_5['event'].to_list()
    # Padding sequences to have same length 
    event = pad_sequences(event_sequence)
    short_sequence_5['event2']=event.tolist()
#-----------------------------------------Spliting Data -----------------------------------------#

    X=short_sequence_5[['user_session','product','event2']]
    y = np.array(pd.get_dummies(short_sequence_5['purchase'] , prefix='Purchase'))
    X_train, X_test, y_train, y_test = train_test_split(X , y,test_size=0.3)

#-------------------------Resizing is necessary since input to sequence models is (1,d)------------#

    Xe_train = np.array((X_train['event2'].tolist())) 
    Xe_train=Xe_train.reshape((Xe_train.shape[0], 1, Xe_train.shape[1]))
    Xe_test = np.array((X_test['event2'].tolist())) 
    Xe_test=Xe_test.reshape((Xe_test.shape[0], 1, Xe_test.shape[1]))

#------------------------------------------------Initializing RNN model---------------------------------#

    model = Sequential()
    model.add(SimpleRNN(units=40, return_sequences = True, input_shape = (1,5) ))
    model.add(SimpleRNN(2*40))
    model.add(Dense(2, activation='softmax'))
    model.compile(optimizer=tf.optimizers.Adam(learning_rate=0.0003), loss='mean_absolute_error',metrics=['acc'])
#-------------------------------------------------Fitting RNN Model--------------------------------------#

    model.fit(Xe_train, y_train,epochs=30 , batch_size=1000 , validation_data=(Xe_test, y_test)) 
    model_adam = model

#-------------------------------------------------Saving RNN Model--------------------------------------#

    model_adam.save("C:/Users/hano0/Desktop/DSI8/atom/RNNmodel.h5")
    print("Saved model to disk")

#------------------------------------Returning Test Data for Prediction ------------------------#
    return  Xe_test, y_test , X_test


def get_cleand_df(path):
     with open(path, 'rb') as infile:
         df=pickle.load(infile)
         return df

# __________________________________________________ Main ____________________________________________________#



# Path of files 
path='C:/Users/hano0/Desktop/DSI8/atom/shopping.pkl'
path1 ='C:/Users/hano0/Desktop/DSI8/atom/'
# Calling Functions 
df1=read_df(path)
df2=preprocessing(df1)

# -------------SVD Modelling---------------#

df_svd = groupbyforSVD(df2)
# Next we called chunking() function and runing shell file (concat.sh)  
# chunking(df_svd)
# import concat.sh 
# We will use saved file from shell in modeling_SVD()
event_product , corr_mat = modeling_SVD()

# -------------RNN Modelling---------------#

Xe_test, y_test , X_test = rnn_model(df2)
# loading saved RNN model
loaded_model=load_model("C:/Users/hano0/Desktop/DSI8/atom/RNNmodel.h5" )
print("Loaded model from disk")
# Prediction from RNN model
pred=loaded_model.predict_classes(Xe_test)
newdf= X_test[['user_session','product','event2' ]]
newdf['purchase_pred']= pred.tolist()
newdf.reset_index(inplace=True)
newdf.drop('index' , axis=1, inplace=True)



# #-----------------------Saving Data----------------#
with open('corrmat.pkl','wb') as out:
             pickle.dump(corr_mat,out)
             out.close()

with open('prediction.pkl','wb') as out:
             pickle.dump(newdf,out)
             out.close()

with open('cleand_data.pkl','wb') as out:
             pickle.dump(df2,out)
             out.close()
             
with open('event_column.pkl','wb') as out:
             pickle.dump(event_product,out)
             out.close()


print("-------------------------------") 
print("DONE") 
print("-------------------------------") 