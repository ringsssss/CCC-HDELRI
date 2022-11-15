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
from feature_extract import getData
from xgboost import XGBClassifier
import psutil
import os


def cv3(seed,path):
    import time
    m_acc = []
    m_pre = []
    m_recall = []
    m_F1 = []
    m_AUC = []
    m_AUPR = []
    for times in range(10):
        itime = time.time()
        print("*************%d**************"%(times+1))
        a = np.array(getData(seed,path))

        feature = a[:,:-1]
        labels = a[:,-1]

        sum_acc = 0
        sum_pre = 0
        sum_recall = 0
        sum_f1 = 0
        sum_AUC = 0
        sum_AUPR = 0
        kf = StratifiedKFold(n_splits=5, random_state=None, shuffle=True)
        for train_index,test_index in kf.split(feature,labels):
            #X_train,X_test,y_train,y_test = train_test_split(feature,labels,test_size=0.2,random_state=42, stratify=labels)
            X_train, X_test = feature[train_index], feature[test_index]
            y_train, y_test = labels[train_index], labels[test_index]

            # y_train = np.array(list(map(int,y_train)))
            # y_test = np.array(list(map(int,y_test)))


            model = XGBClassifier() 
            model.fit(X_train, y_train)

            #pred = model.predict(X_test)
            #print(pred.shape)
            pred = model.predict(X_test)
            prob = model.predict_proba(X_test)[:,1]
            #print(prob)


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
        print('One time 5-Folds computed. Time: {}m, mem: {}MB'.format((time.time() - itime)/60,psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024))

    print("precision:%.4f+%.4f"%(np.mean(m_pre),np.std(np.array(m_pre))))
    print("recall:%.4f+%.4f"%(np.mean(m_recall),np.std(np.array(m_recall))))
    print("accuracy:%.4f+%.4f"%(np.mean(m_acc),np.std(np.array(m_acc))))
    print("F1 score:%.4f+%.4f"%(np.mean(m_F1),np.std(np.array(m_F1))))
    print("AUC:%.4f+%.4f"%(np.mean(m_AUC),np.std(np.array(m_AUC))))
    print("AUPR:%.4f+%.4f"%(np.mean(m_AUPR),np.std(np.array(m_AUPR))))
    print('One time 5-Folds computed. Time: {}m, mem: {}MB'.format((time.time() - itime)/60,psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024))
    

if __name__ == "__main__":
    cv3(1,'./human/')
    cv3(1,'./mouse/')
    cv3(1,'./mouse-heart/')
    cv3(1,'./data4/')
