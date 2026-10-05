import numpy as np;
import pandas as pd;
import matplotlib.pyplot as plt;
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import RandomOverSampler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from sklearn.naive_bayes import GaussianNB 
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import tensorflow as tf;
cols=["fLength","fWidth","fSize","fConc","fConc1","fAsym","fM3Long","Fm3Trans","fAlpha","fDist","class"]
df=pd.read_csv("magic04.data",names=cols);
df["class"]=(df["class"]=="g").astype(int);
print(df.head());

for label in cols[:-1]:
    plt.hist(df[df["class"]==1][label],color='blue',label="gamma",alpha=0.7,density=True)
    plt.hist(df[df["class"]==0][label],color='red',label="hadron",alpha=0.7,density=True)
    plt.title(label)
    plt.ylabel("Probability")
    plt.xlabel(label)
    plt.legend()
    plt.show()

# train , validation and test dataset 
df_shuffled = df.sample(frac=1, random_state=42).reset_index(drop=True)

n = len(df_shuffled)
train_end = int(0.6 * n)
val_end = int(0.8 * n)

train_data = df_shuffled[:train_end]
val_data   = df_shuffled[train_end:val_end]
test_data  = df_shuffled[val_end:]


# scaling function 
def scale_dataset(dataframe , oversample=False):
    x=dataframe[dataframe.columns[:-1]].values
    y=dataframe[dataframe.columns[-1]].values

    scaler=StandardScaler()
    X=scaler.fit_transform(x)

    if oversample:
        ros=RandomOverSampler();
        X,y=ros.fit_resample(X,y)
    data=np.hstack((X,np.reshape(y,(-1,1))))

    return data,X,y;

train,X_train,y_train=scale_dataset(train_data,oversample=True);
val,X_val,y_val=scale_dataset(val_data,oversample=False);
test,X_test,y_test=scale_dataset(test_data,oversample=False);

print(len(y_train))
print(sum(y_train==1))

# knn

knn_model=KNeighborsClassifier(n_neighbors=3)
knn_model.fit(X_train,y_train);

y_pred=knn_model.predict(X_test);

# navier bayes model

nb_model=GaussianNB();
nb_model=nb_model.fit(X_train,y_train)
y_pred_nb=nb_model.predict(X_test)


# logistic regression 
lg_model=LogisticRegression();
lg_model=lg_model.fit(X_train,y_train)
y_pred_lg=lg_model.predict(X_test)

# svm  

svm_model=SVC();
svm_model=svm_model.fit(X_train,y_train);
y_pred_svm=svm_model.predict(X_test);
print(classification_report(y_test,y_pred_svm));

# neural networks implementation 

def plot_history(history):
    fig , (ax1,ax2)=plt.subplots(1,2,figsize=(10,4))

    ax1.plot(history.history['loss'],label='loss')
    ax1.plot(history.history['val_loss'],label='val_loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Binary crossentropy')
    ax1.legend()
    ax1.grid(True)
    


    ax2.plot(history.history['accuracy'],label='accuracy')
    ax2.plot(history.history['val_accuracy'],label='val_accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.grid(True)

    plt.show()

def train_model(X_train,y_train,num_nodes,dropout_prob,lr,batch_size,epochs):

    nn_model=tf.keras.Sequential([
    tf.keras.layers.Dense(num_nodes,activation='relu',input_shape=(10,)),
    tf.keras.layers.Dropout(dropout_prob),
    tf.keras.layers.Dense(num_nodes,activation="relu"),
    tf.keras.layers.Dropout(dropout_prob),
    tf.keras.layers.Dense(1,activation="sigmoid")
    ])
    nn_model.compile(optimizer=tf.keras.optimizers.Adam(lr),loss='binary_crossentropy',metrics=['accuracy'])

    history=nn_model.fit(
    X_train,y_train,epochs=epochs,batch_size=batch_size,validation_split=0.2,verbose=0
    )

    return nn_model,history
epochs=200;
least_val_loss=float('inf')
least_loss_model=None
for num_nodes in [16,32,64]:
    for dropout_prob in [0,0.2]:
        for lr in [0.01,0.005,0.001]:
            for batch_size in [32,64,128]:
                print(f"{num_nodes} nodes , {dropout_prob} dropout_prob , {lr} lr , batch_size{batch_size}")
                model,history=train_model(X_train,y_train,num_nodes,dropout_prob,lr,batch_size,epochs)
                plot_history(history)
                val_loss=model.evaluate(X_val,y_val)[0]
                if val_loss<least_val_loss:
                    least_val_loss=val_loss
                    least_loss_model=model