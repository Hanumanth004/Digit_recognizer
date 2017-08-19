import numpy as np
from scipy.special import expit
import mnist_loader
from random import randint
import time


X_tr, X_val, X_te = mnist_loader.load_data_wrapper()

hidden_size=30
input_size=784
classes=10
K=0
BATCH_SIZE=100
learning_rate=0.1

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
        hg=expit(h12)
        h3=np.dot(Wo,hg) + bo
        hg3=expit(h3)
        y=hg3
        #loss+=0.5*(T-y)*(T-y)
        loss_tmp=np.sum(0.5*(T-y)*(T-y))
        loss_tmp=loss_tmp/10.0
        loss+=loss_tmp
        #backward propogation
        de=-(T-y)
        dhg3=hg3*(1-hg3)
        dy=dhg3*de
        dbo+=dy
        dWo+=np.dot(dy,hg.T)
        dh12=np.dot(Wo.T,dy)
        dh12tmp=hg*(1-hg)*dh12
        dbh+=dh12tmp
        dWh+=np.dot(dh12tmp, X.T)
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
			
	#np.clip(dWh,-2,2,dWh)
	#np.clip(dWo,-2,2,dWo)
	#np.clip(dbh,-2,2,dbh)
	#np.clip(dbo,-2,2,dbo)
	#return loss,dWh,dWo,dbh,dbo
    return loss

X_tmp=X_tr
total_time=0.0
for ep in xrange(50):
    start=time.time()	
    loss=lossFunc(X_tmp)
    end=time.time();
    total_time+=(end-start)
    print 'iteration number %d time taken %f' % (ep, (end-start))
    np.random.shuffle(X_tmp)
    print "\nepoch number:%d" % (ep)
    correctly_classified=0
    if(ep%5==0):
        print loss
    for i in xrange(len(X_tr)):
        X,T = X_tr[i]
        #forward propogation
        h12 = np.dot(Wh, X) + bh
        hg=expit(h12)
        h3=np.dot(Wo,hg) + bo
        hg3=expit(h3)
        y=hg3
        pos_e=np.argmax(y)
        pos_g=np.argmax(T)
        if (pos_e==pos_g):
            correctly_classified+=1
    if(ep%5==0):
        print loss
    accuracy=float(correctly_classified)/len(X_tr)
    print 'training accuracy:%f' %(accuracy)
    correctly_classified=0
    for i in xrange(len(X_val)):
        X,T = X_val[i]
        #forward propogation
        h12 = np.dot(Wh, X) + bh
        hg=expit(h12)
        h3=np.dot(Wo,hg) + bo
        hg3=expit(h3)
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
        hg=expit(h12)
        h3=np.dot(Wo,hg) + bo
        hg3=expit(h3)
        y=hg3
        pos_e=np.argmax(y)
        if (pos_e==T):
            correctly_classified+=1
    accuracy=float(correctly_classified)/len(X_te)
    print 'test set accuracy:%f' %(accuracy)


average_time=total_time/10.0

print "total time:%f" % (average_time)
