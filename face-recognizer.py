import numpy as np
from keras import layers
from keras.layers import Input, Dense, Activation, BatchNormalization, Flatten, Conv2D,Concatenate
from keras.layers import AveragePooling2D, MaxPooling2D
from keras.models import Model
from keras.optimizers import Adam
import cv2
import os
from keras.initializers import glorot_uniform
from keras import backend as K
import keras
import tensorflow as tf

npath=os.listdir('./negative/')
ppath=os.listdir('./positive/')
apath=os.listdir('./anchor/')
tpath=os.listdir('./test/')
xa1=[]
xt1=[]
xp=[]
xn=[]
for f in apath[1:]:
    a=cv2.imread('anchor/'+str(f))
    a=cv2.resize(a,(170,250))
    cv2.imwrite('anchor/'+str(f),a)
    xa1.append(a)
for f in ppath[1:]:
    a=cv2.imread('positive/'+str(f))
    a=cv2.resize(a,(170,250))
    cv2.imwrite('positive/'+str(f),a)
    xp.append(a)
for f in npath[1:]:
    a=cv2.imread('negative/'+str(f))
    a=cv2.resize(a,(170,250))
    cv2.imwrite('negative/'+str(f),a)
    xn.append(a)
for f in tpath[1:]:
    a=cv2.imread('test/'+str(f))
    a=cv2.resize(a,(170,250))
    cv2.imwrite('test/'+str(f),a)
    xt1.append(a)
    
xk=[]
xa=[]
for i in range(0,69):
    for j in range(0,69):
        if (i==j):
            continue
        xk.append([xa1[i],xp[i],xa[j]])
        xk.append([xa1[i],xp[i],xp[j]])
xk=np.array(xk)
xa=xk[:,0,:,:,:]
xp=xk[:,1,:,:,:]
xn=xk[:,2,:,:,:]
print(xa.shape,xp.shape,xn.shape)
    
xa=np.array(xa)/255
xp=np.array(xp)/255
xn=np.array(xn)/255
xt=np.array(xt)/255

y=[]
z=[]
for i in range(0,69):
    y.append(i)
y=np.array(y)
print(y)

model1 = keras.Sequential(
    [
        Conv2D(12, (3, 3), strides = (1, 1),padding='valid',activation='selu',kernel_initializer = glorot_uniform(seed=0)),
        BatchNormalization(axis = 3,trainable=False),
        AveragePooling2D((2, 2)),
        Conv2D(25, (3, 3), strides = (1, 1),padding='valid',activation='selu',kernel_initializer = glorot_uniform(seed=0)),
        BatchNormalization(axis = 3,trainable=False),
        AveragePooling2D((2, 2)),
        Conv2D(51, (3, 3), strides = (1, 1),padding='Valid',activation='selu',kernel_initializer = glorot_uniform(seed=0)),
        BatchNormalization(axis = 3,trainable=False),
        AveragePooling2D((2, 2)),
        AveragePooling2D((2, 2)),
        Conv2D(102, (4, 4), strides = (1, 1),padding='Valid',activation='selu',kernel_initializer = glorot_uniform(seed=0)),
        BatchNormalization(axis = 3,trainable=False),
        AveragePooling2D((2, 2)),
        Conv2D(204, (4, 4), strides = (1, 1),padding='Valid',activation='selu',kernel_initializer = glorot_uniform(seed=0)),
        BatchNormalization(axis = 3,trainable=False),
        AveragePooling2D((2, 2)),
        Flatten(),
        Dense(512, activation='tanh'),
        Dense(256,activation='sigmoid')
    ]
)

def final(input_shape):
    ia=Input(input_shape)
    ip=Input(input_shape)
    iN=Input(input_shape)
    i=Concatenate(axis=-1)([ia,ip,iN])
    va=model1(ia)
    vp=model1(ip)
    vn=model1(iN)
    y_pred=Concatenate(axis=-1)([va,vp,vn])
    model=Model(inputs=[ia,ip,iN],outputs=y_pred,name='fmodel')
    return model

model=final([250,170,3])
model.summary()

def kloss(y_true,y_pred):
    outa=y_pred[:,0:256]
    outp=y_pred[:,256:256*2]
    outn=y_pred[:,256*2:]
    loss1=K.sqrt(K.mean(K.square(outa-outp),axis=-1))
    loss2=K.sqrt(K.mean(K.square(outa-outn),axis=-1))
    l3=loss1-loss2+0.9
    l=K.mean(K.maximum(0.0,l3))
    return l

opt=Adam(learning_rate=0.0001,
    beta_1=0.9,
    beta_2=0.999,
    epsilon=1e-07,
    name="Adam")
model.compile(optimizer=opt,loss=kloss)

model.fit([xa,xp,xn],y,epochs=100)

ytest=model.predict([xt1,xt1,xt1])
yanchor=model.predict([xa1,xa1,xa1])

ytest=ytest[:,0:256]
yanchor=yanchor[:,0:256]

m=10
n=0
for i in range(0,13):
    for j in range(0,69):
        k=np.sqrt(np.mean(np.square(ytest[i,:]-yanchor[j,:])))
        print(k)
        if(k<m):
            n=j
            m=k
    print(n)
    cv2.imwrite('final/'+str(n)+'img.jpg',xa[n]*255)
    cv2.imwrite('final/'+str(n)+'img2.jpg',xt[i]*255)
    m=10
    n=0
