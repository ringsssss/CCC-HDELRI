from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split 
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score 
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score,precision_recall_curve,roc_curve,auc
import pandas as pd
import numpy as np
from feature_extract import getData,getData_case

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, LSTM,Embedding 
from keras.optimizers import Adam
from keras.optimizers import SGD
from itertools import chain

from pai4sk import BoostingMachine

def cv3(seed,path):
    import time
    m_acc = []
    m_pre = []
    m_recall = []
    m_F1 = []
    m_AUC = []
    m_AUPR = []
    for times in range(5):
        itime = time.time()
        print("*************%d**************"%(times+1))
        a = np.array(getData(seed,path))

        feature = a[:,:-1]
        d = feature.shape[1]   #feature dimension
        labels = a[:,-1]

        sum_acc = 0
        sum_pre = 0
        sum_recall = 0
        sum_f1 = 0
        sum_AUC = 0
        sum_AUPR = 0
        kf = StratifiedKFold(n_splits=5, random_state=None, shuffle=True)
        for train_index,test_index in kf.split(feature,labels):
            X_train, X_test = feature[train_index], feature[test_index]
            y_train, y_test = labels[train_index], labels[test_index]

            X_train_lstm = X_train.reshape(X_train.shape[0], 1, X_train.shape[1])
            X_test_lstm = X_test.reshape(X_test.shape[0], 1, X_test.shape[1])

            model = Sequential()


            model.add(LSTM(d, return_sequences=True))  
            model.add(LSTM(d))
            model.add(Dense(256, activation='elu'))  # Full connection layer
            model.add(Dense(128, activation='elu'))
            model.add(Dense(64, activation='elu'))
            model.add(Dense(1, activation='sigmoid'))
            
            model.compile(optimizer=Adam(lr=1e-4),
                        loss='binary_crossentropy',
                        metrics=['accuracy'])
            
            model.fit(X_train_lstm, y_train,
                    epochs=200,
                    batch_size=128,verbose=0)
            prob_lstm = model.predict(X_test_lstm)
            prob_lstm = np.array(list(chain.from_iterable(prob_lstm)))

            model1 = BoostingMachine(objective='logloss',num_round=4000,min_max_depth=1,max_max_depth=24)
            model1.fit(X_train, y_train)

            pred = []
            prob_snap = model1.predict_proba(X_test)[:,1]


            pred = []
            prob = prob_lstm*0.6 + prob_snap*0.4  #Ensemble Learning
            for k in prob:
                if k > 0.8:
                    pred.append(1)
                else:
                    pred.append(0)
            pred = np.array(pred)

            sum_acc += accuracy_score(y_test, pred)
            sum_pre += precision_score(y_test, pred)
            sum_recall += recall_score(y_test, pred)
            sum_f1 += f1_score(y_test, pred)

            fpr, tpr, thresholds = roc_curve(y_test, prob)
            prec, rec, thr = precision_recall_curve(y_test, prob)
            sum_AUC += auc(fpr,tpr)
            sum_AUPR += auc(rec,prec)

        m_acc.append(sum_acc/5)
        m_pre.append(sum_pre/5)
        m_recall.append(sum_recall/5)
        m_F1.append(sum_f1/5)
        m_AUC.append(sum_AUC/5)
        m_AUPR.append(sum_AUPR/5)
        print("precision:%.4f+%.4f" % (np.mean(m_pre), np.std(np.array(m_pre))))
        print("recall:%.4f+%.4f" % (np.mean(m_recall), np.std(np.array(m_recall))))
        print("accuracy:%.4f+%.4f" % (np.mean(m_acc), np.std(np.array(m_acc))))
        print("F1 score:%.4f+%.4f" % (np.mean(m_F1), np.std(np.array(m_F1))))
        print("AUC:%.4f+%.4f" % (np.mean(m_AUC), np.std(np.array(m_AUC))))
        print("AUPR:%.4f+%.4f" % (np.mean(m_AUPR), np.std(np.array(m_AUPR))))
        print('One time 5-Folds computed. Time: {}m'.format((time.time() - itime)/60))

    print("precision:%.4f+%.4f"%(np.mean(m_pre),np.std(np.array(m_pre))))
    print("recall:%.4f+%.4f"%(np.mean(m_recall),np.std(np.array(m_recall))))
    print("accuracy:%.4f+%.4f"%(np.mean(m_acc),np.std(np.array(m_acc))))
    print("F1 score:%.4f+%.4f"%(np.mean(m_F1),np.std(np.array(m_F1))))
    print("AUC:%.4f+%.4f"%(np.mean(m_AUC),np.std(np.array(m_AUC))))
    print("AUPR:%.4f+%.4f"%(np.mean(m_AUPR),np.std(np.array(m_AUPR))))
    print('One time 5-Folds computed. Time: {}m'.format((time.time() - itime)/60))

    

if __name__ == "__main__":
    cv3(1,'./mouse/')

