import numpy as np
from scipy.special import expit
import mnist_loader
from random import randint

X_tr, X_val, X_te = mnist_loader.load_data_wrapper()

hidden_size=30
input_size=784
classes=10
K=0
BATCH_SIZE=100
learning_rate=0.001

Wh=np.random.randn(hidden_size,input_size)*0.01
bh=np.random.randn(hidden_size,1)*0.01
Wo=np.random.randn(classes,hidden_size)*0.01
bo=np.random.randn(classes,1)*0.01



def lossFunc(X_tmp):
    global Wh
    global bh
    global Wo
    global bo
    global K
    loss=0
    dbo=np.zeros_like(bo)
    dWo=np.zeros_like(Wo)
    hg12=np.zeros_like(bh)
    dbh=np.zeros_like(bh)
    dh12tmp=np.zeros_like(bh)
    dh12=np.zeros_like(bh)
    dWh=np.zeros_like(Wh)
    K=0
    for i in xrange(len(X_tmp)):
        X,T = X_tmp[i]
        #forward propogation
        h12 = np.dot(Wh, X) + bh
        hg=np.maximum(0,h12)
        h3=np.dot(Wo,hg) + bo
        hg3=np.maximum(0,h3)
        y=hg3
        #loss+=0.5*(T-y)*(T-y)
        loss_tmp=np.sum(0.5*(T-y)*(T-y))
        loss_tmp=loss_tmp/10.0
        loss+=loss_tmp
        #backward propogation
        de=-(T-y)
        dhg3=hg3
        dhg3[dhg3<=0]=0
        dy=dhg3*de
        dbo+=dy
        dWo+=np.dot(dy,hg.T)
        dh12=np.dot(Wo.T,dy)
        dh12tmp=hg
        dh12tmp[dh12tmp <= 0]=0
        dh12tmp*=dh12
        dbh+=dh12tmp
        dWh+=np.dot(dh12tmp, X.T)
        #np.clip(dWh,-5,5,dWh)
        #np.clip(dWo,-2,5,dWo)
        #np.clip(dbh,-5,5,dbh)
        #np.clip(dbo,-5,5,dbo)
        if(i%BATCH_SIZE==0):
            Wh+=-learning_rate*dWh
            Wo+=-learning_rate*dWo
            bh+=-learning_rate*dbh
            bo+=-learning_rate*dbo
            dbo=np.zeros_like(bo)
            dWo=np.zeros_like(Wo)
            dbh=np.zeros_like(bh)
            dWh=np.zeros_like(Wh)
            K+=BATCH_SIZE
			
	#return loss,dWh,dWo,dbh,dbo
    return loss

X_tmp=X_tr
for ep in xrange(50):
    loss=lossFunc(X_tmp);	
    np.random.shuffle(X_tmp)	
    print "\nepoch number:%d" % (ep)
    correctly_classified=0
    for i in xrange(len(X_tr)):
        X,T = X_tr[i]
        #forward propogation
        h12 = np.dot(Wh, X) + bh
        hg=np.maximum(0,h12)
        h3=np.dot(Wo,hg) + bo
        hg3=np.maximum(0,h3)
        y=hg3
        pos_e=np.argmax(y)
        pos_g=np.argmax(T)
        if (pos_e==pos_g):
            correctly_classified+=1
    if(ep%10==0):
        print loss
    accuracy=float(correctly_classified)/len(X_tr)
    print 'training accuracy:%f' %(accuracy)

    correctly_classified=0
    for i in xrange(len(X_val)):
        X,T = X_val[i]
        #forward propogation
        h12 = np.dot(Wh, X) + bh
        hg=np.maximum(0,h12)
        h3=np.dot(Wo,hg) + bo
        hg3=np.maximum(0,h3)
        y=hg3
        pos_e=np.argmax(y)
        if (pos_e==T):
            correctly_classified+=1
    accuracy=float(correctly_classified)/len(X_val)
    print 'validation accuracy:%f' %(accuracy)
    
    correctly_classified=0
    for i in xrange(len(X_te)):
        X,T = X_te[i]
        #forward propogation
        h12 = np.dot(Wh, X) + bh
        hg=np.maximum(0,h12)
        h3=np.dot(Wo,hg) + bo
        hg3=np.maximum(0,h3)
        y=hg3
        pos_e=np.argmax(y)
        if (pos_e==T):
            correctly_classified+=1
    accuracy=float(correctly_classified)/len(X_te)
    print 'test set accuracy:%f' %(accuracy)
