import pandas as pd;
import numpy as np;
import matplotlib.pyplot as plt;
from imblearn.over_sampling import RandomOverSampler
from sklearn.preprocessing import StandardScaler
import copy
import seaborn as sns
import tensorflow as tf;
from sklearn.linear_model import LinearRegression

dataset_cols=["bike_count","hour","temp","humidity","wind","visibility","dew_pt_temp","radiation","rain","snow","functional"]
df=pd.read_csv("SeoulBikeData.csv",).drop(["Date","Seasons","Holiday"],axis=1);
df.columns=dataset_cols;
df["functional"]=(df["functional"]=="Yes").astype(int)
df=df[df["hour"]==12]
df=df.drop(["hour","wind","visibility","functional"],axis=1)

for label in df.columns[1:]:
    plt.scatter(df[label],df["bike_count"])
    plt.title(label)
    plt.ylabel("Bike Count at Noon")
    plt.xlabel(label)
    plt.show()

# train , validation and test dataset 
df_shuffled = df.sample(frac=1, random_state=42).reset_index(drop=True)

n = len(df_shuffled)
train_end = int(0.6 * n)
val_end = int(0.8 * n)

train_data = df_shuffled[:train_end]
val_data   = df_shuffled[train_end:val_end]
test_data  = df_shuffled[val_end:]

def get_xy(dataframe,y_label,x_labels=None):
    dataframe=copy.deepcopy(dataframe);
    if x_labels is None:
        X=dataframe[[c for c in dataframe.coloumns if c!=y_label ]].values
    else:
        if len(x_labels)==1:
            X=dataframe[x_labels[0]].values.reshape(-1,1);
        else:
            X=dataframe[x_labels].values
    y=dataframe[y_label].values.reshape(-1,1)
    data=np.hstack((X,y))
    return data,X,y

_,X_train,y_train=get_xy(train_data,"bike_count",x_labels=["temp"])
_,x_test,y_test=get_xy(test_data,"bike_count",x_labels=["temp"])
_,X_val,y_val=get_xy(val_data,"bike_count",x_labels=["temp"])

temp_reg=LinearRegression()
temp_reg.fit(X_train,y_train);

def plot_loss(history):
    plt.plot(history.history['loss'],label="loss")
    plt.plot(history.history['val_loss'],label="val_loss")
    plt.xlabel('Epoch')
    plt.ylabel('MSE')
    plt.legend()
    plt.grid(True)
    plt.show()

plt.scatter(X_train,y_train,label="Data",color="blue")
x=tf.linspace(-20,40,100)
plt.plot(x,temp_reg.predict(np.array(x).reshape(-1,1)),label="Fit",color="red",linewidth=3)
plt.legend()
plt.title("Bikes vs Temp")
plt.ylabel("Number of bikes")
plt.xlabel("Temp")
plt.show()

x_labels=list(df.columns[1:])
_,X_train_all,y_train_all=get_xy(train_data,"bike_count",x_labels)
_,x_test_all,y_test_all=get_xy(test_data,"bike_count",x_labels)
_,X_val_all,y_val_all=get_xy(val_data,"bike_count",x_labels)

all_reg=LinearRegression()
all_reg.fit(X_train_all,y_train_all)
print (all_reg.score(x_test_all,y_test_all))


# regression with neural net 
temp_normalizer=tf.keras.layers.Normalization(input_shape=(1,),axis=None)
temp_normalizer.adapt(X_train.reshape(-1))

temp_nn_model=tf.keras.Sequential([
    temp_normalizer,
    tf.keras.layers.Dense(1)
])
temp_nn_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.1),loss='mean_squared_error')

history=temp_nn_model.fit(
    X_train.reshape(-1),y_train,verbose=0,epochs=1000,validation_data=(X_val,y_val)
)

plot_loss(history)

plt.scatter(X_train,y_train,label="Data",color="blue")
x=tf.linspace(-20,40,100)
plt.plot(x,temp_nn_model.predict(np.array(x).reshape(-1,1)),label="Fit",color="red",linewidth=3)
plt.legend()
plt.title("Bikes vs Temp")
plt.ylabel("Number of bikes")
plt.xlabel("Temp")
plt.show()

temp_normalizer=tf.keras.layers.Normalization(input_shape=(1,),axis=None)
temp_normalizer.adapt(X_train.reshape(-1))

nn_model=tf.keras.Sequential([
    temp_normalizer,
    tf.keras.layers.Dense(128,activation="relu"),
    tf.keras.layers.Dense(64,activation="relu"),
    tf.keras.layers.Dense(1,activation="relu")
])
nn_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),loss='mean_squared_error')

history_nn=nn_model.fit(
    X_train,y_train,validation_data=(X_val,y_val),verbose=0,epochs=250
)
plot_loss(history_nn)