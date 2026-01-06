import pandas as pd
import numpy as np
import tensorflow as tf
import warnings
import keras 
from keras.models import Sequential 
from keras.layers import Dense
from keras.callbacks import EarlyStopping
from keras.regularizers import l2
warnings.simplefilter('ignore', FutureWarning)
filepath='https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/DL0101EN/labs/data/concrete_data.csv'
concrete_data = pd.read_csv(filepath)
#print(concrete_data.head())
#print(concrete_data.shape)
#print(concrete_data.describe())
#print(concrete_data.isnull().sum())
concrete_data_columns=concrete_data.columns
X=concrete_data[concrete_data_columns[concrete_data_columns!='Strength']]
Y=concrete_data['Strength']
#test-train split
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.3,random_state=42)

n_cols = X_train.shape[1]
input_dim = n_cols
# 4. Normalize using TRAINING DATA ONLY
mean = X_train.mean()
std = X_train.std()
X_train = (X_train - mean) / std
#NORMALIZE TEST DATA
X_test  = (X_test - mean) / std

def regression_model(input_dim):
    model=Sequential()
    model.add(Dense(64,activation='relu',kernel_regularizer=l2(0.001),input_shape=(input_dim,)))
    model.add(Dense(64,activation='relu',kernel_regularizer=l2(0.001)))
    model.add(Dense(32,activation='relu',kernel_regularizer=l2(0.001)))
    model.add(Dense(1))
    model.compile(optimizer='adam',loss='mean_squared_error')
    return model
model = regression_model(input_dim)
#early stopping
early_stop=EarlyStopping(monitor='val_loss',patience=10,restore_best_weights=True)
# Fit the model
model.fit(X_train,Y_train, validation_data=(X_test,Y_test), epochs=200, verbose=2)
#Vlidation loss
val_loss = model.evaluate(X_test, Y_test, verbose=0)
rmse=np.sqrt(val_loss)
print(f"Test RMSE: {rmse: .2f}")
print(f"Test MSE: {val_loss: .2f}")
# Make predictions
predictions=model.predict(X_test)
print("\nPrediction vs actual values:")
for i in range(X_test.shape[0]):
    print(f"predicted :{predictions[i][0]:.2f} vs actual: {Y_test.iloc[i]: .2f}")

